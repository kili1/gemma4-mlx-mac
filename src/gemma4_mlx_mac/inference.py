from __future__ import annotations

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
        prompt = _render_prompt(loaded.tokenizer, request.messages)
        started_at = time.monotonic()
        completion_tokens = 0
        prompt_tokens = 0

        try:
            for raw_chunk in self._stream_completion(loaded, prompt, request):
                text = str(getattr(raw_chunk, "text", raw_chunk))
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
        except InferenceNotReady:
            raise
        except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
            raise InferenceError(f"MLX generation failed: {exc}") from exc

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


def _fallback_prompt(messages: list[ChatMessage]) -> str:
    parts = [f"{message.role}: {message.content}" for message in messages]
    parts.append("assistant:")
    return "\n".join(parts)


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
