from __future__ import annotations

import inspect
import os
import re
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

_GEMMA_THOUGHT_OPEN = "<|channel>thought"
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_LEGACY_THOUGHT_OPEN = "<think>"
_LEGACY_THOUGHT_CLOSE = "</think>"
_STRIP_THOUGHTS_RE = re.compile(
    r"<\|channel>thought.*?<channel\|>|<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)


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
    adapter_path: str | None = None

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("Model id or local path is required.")
        return model

    @field_validator("adapter_path")
    @classmethod
    def normalize_adapter_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return str(Path(cleaned).expanduser()) if cleaned else None


class InferenceNotReady(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedChatModel:
    model: Any
    tokenizer: Any
    path: str
    adapter_path: str | None = None


@dataclass(frozen=True)
class GeneratedToken:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    finish_reason: str | None = None
    is_thinking: bool = False


class LoadedModelMemory(BaseModel):
    id: str
    path: str
    adapter_path: str | None = None


class ModelMemoryStatus(BaseModel):
    loaded: bool
    loaded_count: int
    loaded_models: list[LoadedModelMemory]
    process_memory_bytes: int | None
    baseline_memory_bytes: int | None
    model_memory_bytes: int | None
    note: str


ModelLoader = Callable[..., tuple[Any, Any]]
StreamGenerator = Callable[..., Iterable[Any]]
SamplerFactory = Callable[[float], Callable[..., Any]]
ModelPathResolver = Callable[[str], str | None]
ActiveAdapterResolver = Callable[[], str | None]


class ChatService:
    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        stream_generator: StreamGenerator | None = None,
        sampler_factory: SamplerFactory | None = None,
        model_path_resolver: ModelPathResolver = get_cached_model_path,
        active_adapter_resolver: ActiveAdapterResolver | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._stream_generator = stream_generator
        self._sampler_factory = sampler_factory
        self._model_path_resolver = model_path_resolver
        self._active_adapter_resolver = active_adapter_resolver
        self._loaded_models: dict[tuple[str, str | None], LoadedChatModel] = {}
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
        adapter_path = self._resolve_adapter_path(request.adapter_path)
        loaded = self._load_model(request.model, adapter_path=adapter_path)
        prompt = _render_prompt(
            loaded.tokenizer,
            _messages_for_request(request),
            enable_thinking=request.show_thinking,
        )
        parser = _GemmaThoughtStreamParser()
        formatter = _ThinkingOutputFormatter(show_thinking=request.show_thinking)
        started_at = time.monotonic()
        completion_tokens = 0
        prompt_tokens = 0

        try:
            for raw_chunk in self._stream_completion(loaded, prompt, request):
                raw_text = str(getattr(raw_chunk, "text", raw_chunk))
                segments = parser.feed(raw_text)
                emitted = formatter.format_segments(segments)
                prompt_tokens = int(getattr(raw_chunk, "prompt_tokens", prompt_tokens) or 0)
                reported_tokens = int(getattr(raw_chunk, "generation_tokens", 0) or 0)
                if reported_tokens > completion_tokens:
                    completion_tokens = reported_tokens
                elif emitted:
                    completion_tokens += 1
                elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                finish_reason = getattr(raw_chunk, "finish_reason", None)
                if emitted or finish_reason is not None:
                    yield GeneratedToken(
                        text=emitted,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_seconds=elapsed_seconds,
                        tokens_per_second=completion_tokens / elapsed_seconds,
                        finish_reason=finish_reason,
                        is_thinking=formatter.last_was_thinking,
                    )

            tail_text = formatter.format_segments(parser.flush()) + formatter.flush()
            if tail_text:
                elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                yield GeneratedToken(
                    text=tail_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed_seconds,
                    tokens_per_second=completion_tokens / elapsed_seconds,
                    is_thinking=formatter.last_was_thinking,
                )
        except InferenceNotReady:
            raise
        except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
            raise InferenceError(f"MLX generation failed: {exc}") from exc

    def memory_status(self) -> ModelMemoryStatus:
        process_memory_bytes = _process_resident_memory_bytes()
        with self._lock:
            loaded_models = [
                LoadedModelMemory(
                    id=model_id,
                    path=loaded.path,
                    adapter_path=loaded.adapter_path,
                )
                for (model_id, _adapter), loaded in self._loaded_models.items()
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

    def unload_models(self) -> None:
        with self._lock:
            self._loaded_models.clear()

    def _resolve_adapter_path(self, explicit_adapter: str | None) -> str | None:
        if explicit_adapter is not None:
            candidate = Path(explicit_adapter).expanduser()
            if not candidate.exists():
                raise InferenceNotReady(f"Adapter path does not exist: {explicit_adapter}")
            return str(candidate)
        if self._active_adapter_resolver is not None:
            return self._active_adapter_resolver()
        return None

    def _load_model(self, model_id: str, adapter_path: str | None = None) -> LoadedChatModel:
        cache_key = (model_id, adapter_path)
        with self._lock:
            loaded = self._loaded_models.get(cache_key)
            if loaded is not None:
                return loaded

            model_path = self._resolve_model_path(model_id)
            loader = self._model_loader or _default_model_loader
            try:
                model, tokenizer = _invoke_loader(loader, model_path, adapter_path)
            except InferenceNotReady:
                raise
            except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
                raise InferenceError(f"Could not load model {model_id}: {exc}") from exc

            loaded = LoadedChatModel(
                model=model,
                tokenizer=tokenizer,
                path=model_path,
                adapter_path=adapter_path,
            )
            self._loaded_models[cache_key] = loaded
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


def _invoke_loader(
    loader: ModelLoader,
    model_path: str,
    adapter_path: str | None,
) -> tuple[Any, Any]:
    if adapter_path is not None:
        return loader(model_path, adapter_path=adapter_path)
    try:
        params = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return loader(model_path)
    if "adapter_path" in params:
        return loader(model_path, adapter_path=None)
    return loader(model_path)


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


def _default_model_loader(
    model_path: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    mlx_lm = _load_mlx_lm()
    load_kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    try:
        return mlx_lm["load"](model_path, **load_kwargs)
    except ValueError as exc:
        if "parameters not in model" not in str(exc):
            raise
        if adapter_path is not None:
            return _load_model_without_strict_weight_matching(
                model_path, adapter_path=adapter_path
            )
        return _load_model_without_strict_weight_matching(model_path)


def _load_model_without_strict_weight_matching(
    model_path: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    try:
        from mlx_lm.utils import load_model, load_tokenizer
    except ImportError as exc:
        raise InferenceNotReady(
            "MLX-LM utility loaders are not available in this environment."
        ) from exc

    model_dir = Path(model_path)
    model, config = load_model(model_dir, strict=False)
    if adapter_path:
        try:
            from mlx_lm.tuner.utils import load_adapters
        except ImportError as exc:
            raise InferenceNotReady(
                "MLX-LM adapter loader is not available in this environment."
            ) from exc
        load_adapters(model, adapter_path)
    model.eval()
    tokenizer = load_tokenizer(model_dir, eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def _render_prompt(
    tokenizer: Any,
    messages: list[ChatMessage],
    enable_thinking: bool = False,
) -> str | list[int]:
    serializable_messages = [message.model_dump() for message in messages]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        for kwargs in (
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            },
            {
                "tokenize": False,
                "add_generation_prompt": True,
            },
            {
                "add_generation_prompt": True,
                "return_dict": False,
            },
        ):
            try:
                rendered = apply_chat_template(serializable_messages, **kwargs)
                break
            except TypeError:
                continue
        else:
            return _fallback_prompt(messages)

        if isinstance(rendered, str | list):
            return rendered
        if isinstance(rendered, dict) and "input_ids" in rendered:
            return rendered["input_ids"]

    return _fallback_prompt(messages)


def _messages_for_request(request: ChatCompletionRequest) -> list[ChatMessage]:
    """Strips prior raw thought channels from multi-turn assistant messages."""
    cleaned: list[ChatMessage] = []
    for message in request.messages:
        if message.role == "assistant":
            stripped = _STRIP_THOUGHTS_RE.sub("", message.content).strip()
            cleaned.append(ChatMessage(role="assistant", content=stripped))
        else:
            cleaned.append(message)
    return cleaned


def _fallback_prompt(messages: list[ChatMessage]) -> str:
    parts = [f"{message.role}: {message.content}" for message in messages]
    parts.append("assistant:")
    return "\n".join(parts)


class _GemmaThoughtStreamParser:
    """Separates Gemma 4 ``<|channel>thought...<channel|>`` and ``<think>...</think>`` blocks."""

    _openers = (_GEMMA_THOUGHT_OPEN, _LEGACY_THOUGHT_OPEN)
    _closers = {
        _GEMMA_THOUGHT_OPEN: _GEMMA_THOUGHT_CLOSE,
        _LEGACY_THOUGHT_OPEN: _LEGACY_THOUGHT_CLOSE,
    }

    def __init__(self) -> None:
        self._buffer = ""
        self._active_closer: str | None = None
        self._strip_leading_newline = False

    def feed(self, text: str) -> list[tuple[bool, str]]:
        if not text:
            return []

        combined = self._buffer + text
        self._buffer = ""
        emitted: list[tuple[bool, str]] = []
        cursor = 0

        while cursor < len(combined):
            if self._strip_leading_newline:
                if combined[cursor] == "\n":
                    cursor += 1
                self._strip_leading_newline = False
                if cursor >= len(combined):
                    break

            if self._active_closer is not None:
                closer = self._active_closer
                end = combined.lower().find(closer.lower(), cursor)
                if end == -1:
                    safe_end = _safe_marker_boundary(combined, cursor, (closer,))
                    if safe_end > cursor:
                        emitted.append((True, combined[cursor:safe_end]))
                    self._buffer = combined[safe_end:]
                    return emitted
                if end > cursor:
                    emitted.append((True, combined[cursor:end]))
                cursor = end + len(closer)
                self._active_closer = None
                continue

            match = _find_earliest_marker(combined, cursor, self._openers)
            if match is None:
                safe_end = _safe_marker_boundary(combined, cursor, self._openers)
                if safe_end > cursor:
                    emitted.append((False, combined[cursor:safe_end]))
                self._buffer = combined[safe_end:]
                return emitted

            start, opener = match
            if start > cursor:
                emitted.append((False, combined[cursor:start]))
            cursor = start + len(opener)
            if opener == _GEMMA_THOUGHT_OPEN:
                self._strip_leading_newline = True
            self._active_closer = self._closers[opener]

        return emitted

    def flush(self) -> list[tuple[bool, str]]:
        if not self._buffer:
            return []
        tail = self._buffer
        self._buffer = ""
        return [(self._active_closer is not None, tail)]


class _ThinkingOutputFormatter:
    """Formats parsed ``(is_thought, text)`` segments for chat output."""

    def __init__(self, show_thinking: bool) -> None:
        self._show_thinking = show_thinking
        self._started_thinking = False
        self._started_answer = False
        self.last_was_thinking = False

    def format_segments(self, segments: list[tuple[bool, str]]) -> str:
        parts: list[str] = []
        for is_thought, text in segments:
            if not text:
                continue
            if is_thought:
                if not self._show_thinking:
                    continue
                if not self._started_thinking:
                    parts.append("Thinking:\n")
                    self._started_thinking = True
                parts.append(text)
                self.last_was_thinking = True
                continue

            if self._show_thinking and self._started_thinking and not self._started_answer:
                parts.append("\n\nAnswer:\n")
                self._started_answer = True
            elif not self._started_answer:
                self._started_answer = True
            parts.append(text)
            self.last_was_thinking = False
        return "".join(parts)

    def flush(self) -> str:
        return ""


def _find_earliest_marker(
    text: str,
    start: int,
    markers: tuple[str, ...],
) -> tuple[int, str] | None:
    lower = text.lower()
    best: tuple[int, str] | None = None
    for marker in markers:
        pos = lower.find(marker.lower(), start)
        if pos != -1 and (best is None or pos < best[0]):
            best = (pos, marker)
    return best


def _safe_marker_boundary(text: str, start: int, markers: tuple[str, ...]) -> int:
    lower = text.lower()
    safe_end = len(text)
    for marker in markers:
        marker_lower = marker.lower()
        max_tail = min(len(marker_lower) - 1, len(text) - start)
        for tail_length in range(max_tail, 0, -1):
            tail_start = len(text) - tail_length
            if marker_lower.startswith(lower[tail_start:]):
                safe_end = min(safe_end, tail_start)
                break
    return safe_end


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
