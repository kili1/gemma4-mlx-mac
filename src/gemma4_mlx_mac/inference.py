from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .downloads import get_cached_model_path
from .models import DEFAULT_MODEL_ID


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    stream: bool = False
    show_thinking: bool = False

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("Model id or local path is required.")
        return model


class InferenceNotReady(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedChatModel:
    model: Any
    tokenizer: Any
    path: str


@dataclass(frozen=True)
class GeneratedToken:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    finish_reason: str | None = None


class LoadedModelMemory(BaseModel):
    id: str
    path: str


class ModelMemoryStatus(BaseModel):
    loaded: bool
    loaded_count: int
    loaded_models: list[LoadedModelMemory]
    process_memory_bytes: int | None
    baseline_memory_bytes: int | None
    model_memory_bytes: int | None
    note: str


ModelLoader = Callable[[str], tuple[Any, Any]]
StreamGenerator = Callable[..., Iterable[Any]]
SamplerFactory = Callable[[float], Callable[..., Any]]
ModelPathResolver = Callable[[str], str | None]


class ChatService:
    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        stream_generator: StreamGenerator | None = None,
        sampler_factory: SamplerFactory | None = None,
        model_path_resolver: ModelPathResolver = get_cached_model_path,
    ) -> None:
        self._model_loader = model_loader
        self._stream_generator = stream_generator
        self._sampler_factory = sampler_factory
        self._model_path_resolver = model_path_resolver
        self._loaded_models: dict[str, LoadedChatModel] = {}
        self._baseline_memory_bytes = _process_resident_memory_bytes()
        self._lock = RLock()

    def create_completion(self, request: ChatCompletionRequest) -> dict:
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        try:
            for chunk in self.stream_tokens(request):
                text_parts.append(chunk.text)
                prompt_tokens = chunk.prompt_tokens
                completion_tokens = chunk.completion_tokens
                finish_reason = chunk.finish_reason or finish_reason
        except InferenceNotReady:
            raise
        except InferenceError:
            raise
        except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
            raise InferenceError(f"MLX generation failed: {exc}") from exc

        content = "".join(text_parts)
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def stream_tokens(self, request: ChatCompletionRequest) -> Iterable[GeneratedToken]:
        loaded = self._load_model(request.model)
        prompt = _render_prompt(loaded.tokenizer, _messages_for_request(request))
        thinking_filter = _ThinkingTagFilter()
        visible_thinking = _VisibleThinkingFormatter() if request.show_thinking else None
        started_at = time.monotonic()
        completion_tokens = 0
        prompt_tokens = 0

        try:
            for raw_chunk in self._stream_completion(loaded, prompt, request):
                text = thinking_filter.feed(str(getattr(raw_chunk, "text", raw_chunk)))
                if visible_thinking is not None:
                    text = visible_thinking.feed(text)
                prompt_tokens = int(getattr(raw_chunk, "prompt_tokens", prompt_tokens) or 0)
                reported_tokens = int(getattr(raw_chunk, "generation_tokens", 0) or 0)
                if reported_tokens > completion_tokens:
                    completion_tokens = reported_tokens
                elif text:
                    completion_tokens += 1
                elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                yield GeneratedToken(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed_seconds,
                    tokens_per_second=completion_tokens / elapsed_seconds,
                    finish_reason=getattr(raw_chunk, "finish_reason", None),
                )
            remaining_text = thinking_filter.flush()
            if visible_thinking is not None:
                remaining_text = (
                    f"{visible_thinking.feed(remaining_text)}{visible_thinking.flush()}"
                )
            if remaining_text:
                elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                yield GeneratedToken(
                    text=remaining_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed_seconds,
                    tokens_per_second=completion_tokens / elapsed_seconds,
                )
        except InferenceNotReady:
            raise
        except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
            raise InferenceError(f"MLX generation failed: {exc}") from exc

    def memory_status(self) -> ModelMemoryStatus:
        process_memory_bytes = _process_resident_memory_bytes()
        with self._lock:
            loaded_models = [
                LoadedModelMemory(id=model_id, path=loaded.path)
                for model_id, loaded in self._loaded_models.items()
            ]

        model_memory_bytes: int | None
        if not loaded_models:
            model_memory_bytes = 0 if process_memory_bytes is not None else None
            note = "No model is loaded yet. Send a chat message to load one into unified memory."
        elif process_memory_bytes is None or self._baseline_memory_bytes is None:
            model_memory_bytes = None
            note = "Model is loaded, but process memory could not be read on this system."
        else:
            model_memory_bytes = max(process_memory_bytes - self._baseline_memory_bytes, 0)
            note = (
                "Approximate model footprint based on process resident memory delta since "
                "the server started."
            )

        return ModelMemoryStatus(
            loaded=bool(loaded_models),
            loaded_count=len(loaded_models),
            loaded_models=loaded_models,
            process_memory_bytes=process_memory_bytes,
            baseline_memory_bytes=self._baseline_memory_bytes,
            model_memory_bytes=model_memory_bytes,
            note=note,
        )

    def _load_model(self, model_id: str) -> LoadedChatModel:
        with self._lock:
            loaded = self._loaded_models.get(model_id)
            if loaded is not None:
                return loaded

            model_path = self._resolve_model_path(model_id)
            loader = self._model_loader or _default_model_loader
            try:
                model, tokenizer = loader(model_path)
            except InferenceNotReady:
                raise
            except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
                raise InferenceError(f"Could not load model {model_id}: {exc}") from exc

            loaded = LoadedChatModel(model=model, tokenizer=tokenizer, path=model_path)
            self._loaded_models[model_id] = loaded
            return loaded

    def _resolve_model_path(self, model_id: str) -> str:
        local_model_path = _existing_model_path(model_id)
        if local_model_path is not None:
            return str(local_model_path)
        if _looks_like_path(model_id):
            raise InferenceNotReady(f"Model path does not exist: {model_id}")

        local_path = self._model_path_resolver(model_id)
        if local_path is None:
            raise InferenceNotReady(
                f"{model_id} is not downloaded. Download it in the Models tab before chatting."
            )
        return local_path

    def _stream_completion(
        self,
        loaded: LoadedChatModel,
        prompt: str | list[int],
        request: ChatCompletionRequest,
    ) -> Iterable[Any]:
        stream_generate = self._stream_generator or _load_mlx_lm()["stream_generate"]
        return stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt,
            max_tokens=request.max_tokens,
            sampler=self._make_sampler(request.temperature),
        )

    def _make_sampler(self, temperature: float) -> Callable[..., Any]:
        sampler_factory = self._sampler_factory
        if sampler_factory is None:
            sampler_factory = _load_mlx_lm()["make_sampler"]
        return sampler_factory(temperature)


def _load_mlx_lm() -> dict[str, Any]:
    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise InferenceNotReady(
            "MLX inference dependencies are not installed. Use the Chat install button or run "
            "`gemma4-mlx-mac install-mlx`."
        ) from exc
    return {
        "load": load,
        "stream_generate": stream_generate,
        "make_sampler": make_sampler,
    }


def _default_model_loader(model_path: str) -> tuple[Any, Any]:
    mlx_lm = _load_mlx_lm()
    try:
        return mlx_lm["load"](model_path)
    except ValueError as exc:
        if "parameters not in model" not in str(exc):
            raise
        return _load_model_without_strict_weight_matching(model_path)


def _load_model_without_strict_weight_matching(model_path: str) -> tuple[Any, Any]:
    try:
        from mlx_lm.utils import load_model, load_tokenizer
    except ImportError as exc:
        raise InferenceNotReady(
            "MLX-LM utility loaders are not available in this environment."
        ) from exc

    model_dir = Path(model_path)
    model, config = load_model(model_dir, strict=False)
    model.eval()
    tokenizer = load_tokenizer(model_dir, eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def _render_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str | list[int]:
    serializable_messages = [message.model_dump() for message in messages]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(
                serializable_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            rendered = apply_chat_template(
                serializable_messages,
                add_generation_prompt=True,
                return_dict=False,
            )
        if isinstance(rendered, str | list):
            return rendered
        if isinstance(rendered, dict) and "input_ids" in rendered:
            return rendered["input_ids"]

    return _fallback_prompt(messages)


def _messages_for_request(request: ChatCompletionRequest) -> list[ChatMessage]:
    if request.show_thinking:
        instruction = (
            "You MUST start every response exactly with `Thinking:` followed by 1-3 concise "
            "bullets that summarize high-level considerations only. Then write `Answer:` and "
            "the final answer. Do not omit either label. Do not reveal hidden chain-of-thought "
            "or token-by-token private reasoning."
        )
    else:
        instruction = (
            "Answer directly. Do not include `Thinking:` or `Answer:` labels. Do not reveal "
            "hidden chain-of-thought or step-by-step private reasoning."
        )
    return [ChatMessage(role="system", content=instruction), *request.messages]


def _fallback_prompt(messages: list[ChatMessage]) -> str:
    parts = [f"{message.role}: {message.content}" for message in messages]
    parts.append("assistant:")
    return "\n".join(parts)


class _ThinkingTagFilter:
    _opening = "<think>"
    _closing = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_thinking = False

    def feed(self, text: str) -> str:
        if not text:
            return ""

        combined = self._buffer + text
        self._buffer = ""
        output: list[str] = []
        cursor = 0

        while cursor < len(combined):
            lower = combined.lower()
            if self._inside_thinking:
                end = lower.find(self._closing, cursor)
                if end == -1:
                    tail_start = max(cursor, len(combined) - (len(self._closing) - 1))
                    self._buffer = combined[tail_start:]
                    return "".join(output)
                cursor = end + len(self._closing)
                self._inside_thinking = False
                continue

            start = lower.find(self._opening, cursor)
            if start == -1:
                safe_end = _safe_marker_boundary(combined, cursor, self._opening)
                output.append(combined[cursor:safe_end])
                self._buffer = combined[safe_end:]
                return "".join(output)

            output.append(combined[cursor:start])
            cursor = start + len(self._opening)
            self._inside_thinking = True

        return "".join(output)

    def flush(self) -> str:
        if self._inside_thinking:
            self._buffer = ""
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining


class _VisibleThinkingFormatter:
    _required_prefix = "thinking:"
    _fallback_prefix = (
        "Thinking:\n"
        "- No separate visible reasoning summary was returned by the model.\n\n"
        "Answer:\n"
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._decided = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._decided:
            return text

        self._buffer += text
        candidate = self._buffer.lstrip()
        if not candidate:
            return ""

        lower = candidate.lower()
        if lower.startswith(self._required_prefix):
            self._decided = True
            output = self._buffer
            self._buffer = ""
            return output

        if self._required_prefix.startswith(lower):
            return ""

        self._decided = True
        output = f"{self._fallback_prefix}{self._buffer}"
        self._buffer = ""
        return output

    def flush(self) -> str:
        if self._decided:
            return ""
        if not self._buffer.strip():
            return ""

        candidate = self._buffer.lstrip().lower()
        if candidate.startswith(self._required_prefix):
            output = self._buffer
        else:
            output = f"{self._fallback_prefix}{self._buffer}"

        self._buffer = ""
        self._decided = True
        return output


def _safe_marker_boundary(text: str, start: int, marker: str) -> int:
    lower = text.lower()
    max_tail = min(len(marker) - 1, len(text) - start)
    for tail_length in range(max_tail, 0, -1):
        tail_start = len(text) - tail_length
        if marker.startswith(lower[tail_start:]):
            return tail_start
    return len(text)


def _process_resident_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        rss_kb = int(result.stdout.strip().splitlines()[0])
    except (IndexError, OSError, subprocess.CalledProcessError, ValueError):
        return None
    return rss_kb * 1024


def _existing_model_path(value: str) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate
    return None


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("/", "~", "."))
        or value.count("/") > 1
        or "\\" in value
    )
