from __future__ import annotations

import gc
import inspect
import os
import re
import subprocess
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
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
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]

    def text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = Field(default=None, ge=0.5, le=2.0)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
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

    def stop_sequences(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop] if self.stop else []
        return [item for item in self.stop if item]


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
    mlx_active_memory_bytes: int | None = None
    mlx_cache_memory_bytes: int | None = None
    mlx_peak_memory_bytes: int | None = None
    note: str


ModelLoader = Callable[..., tuple[Any, Any]]
StreamGenerator = Callable[..., Iterable[Any]]
SamplerFactory = Callable[..., Callable[..., Any]]
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
        max_loaded_models: int = 1,
    ) -> None:
        self._model_loader = model_loader
        self._stream_generator = stream_generator
        self._sampler_factory = sampler_factory
        self._model_path_resolver = model_path_resolver
        self._active_adapter_resolver = active_adapter_resolver
        self._max_loaded_models = max(1, max_loaded_models)
        self._loaded_models: OrderedDict[tuple[str, str | None], LoadedChatModel] = OrderedDict()
        self._baseline_memory_bytes = _process_resident_memory_bytes()
        self._lock = RLock()
        self._generate_lock = Lock()

    def prepare_request(self, request: ChatCompletionRequest) -> LoadedChatModel:
        """Validates paths and loads the requested model+adapter before streaming begins."""
        adapter_path = self._resolve_adapter_path(request.adapter_path)
        return self._load_model(request.model, adapter_path=adapter_path)

    def preload(self, model_id: str, adapter_path: str | None = None) -> LoadedChatModel:
        resolved_adapter = self._resolve_adapter_path(adapter_path)
        return self._load_model(model_id, adapter_path=resolved_adapter)

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
        loaded = self.prepare_request(request)
        prompt = _render_prompt(
            loaded.tokenizer,
            _messages_for_request(request),
            enable_thinking=request.show_thinking,
        )
        parser = _GemmaThoughtStreamParser()
        formatter = _ThinkingOutputFormatter(show_thinking=request.show_thinking)
        stop_filter = _StopSequenceFilter(request.stop_sequences())
        started_at = time.monotonic()
        completion_tokens = 0
        prompt_tokens = 0

        # Serialize Metal/MLX generation calls so concurrent HTTP requests never collide on GPU.
        with self._generate_lock:
            try:
                stopped_early = False
                for raw_chunk in self._stream_completion(loaded, prompt, request):
                    raw_text = str(getattr(raw_chunk, "text", raw_chunk))
                    segments = parser.feed(raw_text)
                    emitted = formatter.format_segments(segments)
                    emitted, matched_stop = stop_filter.feed(emitted)
                    prompt_tokens = int(getattr(raw_chunk, "prompt_tokens", prompt_tokens) or 0)
                    reported_tokens = int(getattr(raw_chunk, "generation_tokens", 0) or 0)
                    if reported_tokens > completion_tokens:
                        completion_tokens = reported_tokens
                    elif emitted:
                        completion_tokens += 1
                    elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                    finish_reason = (
                        "stop" if matched_stop else getattr(raw_chunk, "finish_reason", None)
                    )
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
                    if matched_stop:
                        stopped_early = True
                        break

                if not stopped_early:
                    tail_raw = formatter.format_segments(parser.flush()) + formatter.flush()
                    tail_emitted, matched_stop = stop_filter.feed(tail_raw)
                    if not matched_stop:
                        tail_emitted += stop_filter.flush()
                    if tail_emitted or matched_stop:
                        elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
                        yield GeneratedToken(
                            text=tail_emitted,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            elapsed_seconds=elapsed_seconds,
                            tokens_per_second=completion_tokens / elapsed_seconds,
                            finish_reason="stop" if matched_stop else None,
                            is_thinking=formatter.last_was_thinking,
                        )
            except InferenceNotReady:
                raise
            except Exception as exc:  # pragma: no cover - exact MLX exceptions vary by model.
                raise InferenceError(f"MLX generation failed: {exc}") from exc

    def memory_status(self) -> ModelMemoryStatus:
        process_memory_bytes = _process_resident_memory_bytes()
        mlx_active, mlx_cache, mlx_peak = _mlx_memory_bytes()
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
            has_memory = mlx_active is not None or process_memory_bytes is not None
            model_memory_bytes = 0 if has_memory else None
            note = "No model is loaded yet. Send a chat message to load one into unified memory."
        elif mlx_active is not None:
            model_memory_bytes = mlx_active
            note = (
                "Measured directly from MLX unified Metal memory "
                "(active allocations; cache shown separately)."
            )
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
            mlx_active_memory_bytes=mlx_active,
            mlx_cache_memory_bytes=mlx_cache,
            mlx_peak_memory_bytes=mlx_peak,
            note=note,
        )

    def unload_models(self) -> int:
        with self._lock:
            count = len(self._loaded_models)
            self._loaded_models.clear()
        if count > 0:
            _release_mlx_memory()
        return count

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
                self._loaded_models.move_to_end(cache_key)
                return loaded

            model_path = self._resolve_model_path(model_id)
            evicted = False
            while len(self._loaded_models) >= self._max_loaded_models:
                self._loaded_models.popitem(last=False)
                evicted = True
            if evicted:
                _release_mlx_memory()

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
        if request.seed is not None:
            _seed_mlx(request.seed)
        stream_generate = self._stream_generator or _load_mlx_lm()["stream_generate"]
        kwargs: dict[str, Any] = {
            "max_tokens": request.max_tokens,
            "sampler": self._make_sampler(request),
        }
        if request.repetition_penalty is not None:
            processors = _make_logits_processors(request.repetition_penalty)
            if processors is not None:
                kwargs["logits_processors"] = processors
        return stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt,
            **kwargs,
        )

    def _make_sampler(self, request: ChatCompletionRequest) -> Callable[..., Any]:
        sampler_factory = self._sampler_factory
        if sampler_factory is None:
            sampler_factory = _load_mlx_lm()["make_sampler"]
        try:
            params = inspect.signature(sampler_factory).parameters
        except (TypeError, ValueError):
            return sampler_factory(request.temperature)
        if "top_p" in params:
            if "temp" in params:
                return sampler_factory(temp=request.temperature, top_p=request.top_p)
            return sampler_factory(request.temperature, top_p=request.top_p)
        return sampler_factory(request.temperature)


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


def _make_logits_processors(repetition_penalty: float) -> list[Any] | None:
    try:
        from mlx_lm.sample_utils import make_logits_processors
    except ImportError:
        return None
    return make_logits_processors(repetition_penalty=repetition_penalty)


def _seed_mlx(seed: int) -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    try:
        mx.random.seed(seed)
    except Exception:
        pass


def _release_mlx_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        try:
            clear_cache()
            return
        except Exception:
            pass
    metal = getattr(mx, "metal", None)
    metal_clear = getattr(metal, "clear_cache", None) if metal is not None else None
    if callable(metal_clear):
        try:
            metal_clear()
        except Exception:
            pass


def _mlx_memory_bytes() -> tuple[int | None, int | None, int | None]:
    try:
        import mlx.core as mx
    except ImportError:
        return (None, None, None)

    def _read(name: str) -> int | None:
        fn = getattr(mx, name, None)
        if not callable(fn):
            metal = getattr(mx, "metal", None)
            fn = getattr(metal, name, None) if metal is not None else None
        if not callable(fn):
            return None
        try:
            return int(fn())
        except Exception:
            return None

    return (
        _read("get_active_memory"),
        _read("get_cache_memory"),
        _read("get_peak_memory"),
    )


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
    serializable_messages = [
        {"role": message.role, "content": message.text_content()} for message in messages
    ]
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
    """Strips prior raw thought channels and merges consecutive leading system messages."""
    cleaned: list[ChatMessage] = []
    leading_systems: list[str] = []
    collecting_system = True

    for message in request.messages:
        text = message.text_content()
        if collecting_system and message.role in {"system", "developer"}:
            if text.strip():
                leading_systems.append(text.strip())
            continue

        if collecting_system:
            collecting_system = False
            if leading_systems:
                cleaned.append(ChatMessage(role="system", content="\n\n".join(leading_systems)))

        if message.role == "assistant":
            stripped = _STRIP_THOUGHTS_RE.sub("", text).strip()
            cleaned.append(ChatMessage(role="assistant", content=stripped))
        else:
            cleaned.append(ChatMessage(role=message.role, content=text))

    if collecting_system and leading_systems:
        cleaned.append(ChatMessage(role="system", content="\n\n".join(leading_systems)))

    return cleaned


def _fallback_prompt(messages: list[ChatMessage]) -> str:
    parts = [f"{message.role}: {message.text_content()}" for message in messages]
    parts.append("assistant:")
    return "\n".join(parts)


class _StopSequenceFilter:
    """Buffers minimal tail characters needed to detect stop sequences across token boundaries."""

    def __init__(self, stop_sequences: list[str]) -> None:
        self._stops = tuple(dict.fromkeys(seq for seq in stop_sequences if seq))
        self._buffer = ""
        self._max_len = max((len(seq) for seq in self._stops), default=0)

    def feed(self, text: str) -> tuple[str, bool]:
        if not self._stops:
            return text, False
        if not text:
            return "", False

        combined = self._buffer + text
        earliest_pos: int | None = None
        for seq in self._stops:
            pos = combined.find(seq)
            if pos != -1 and (earliest_pos is None or pos < earliest_pos):
                earliest_pos = pos

        if earliest_pos is not None:
            self._buffer = ""
            return combined[:earliest_pos], True

        keep = self._max_len - 1
        if keep <= 0 or len(combined) <= keep:
            self._buffer = combined
            return "", False

        safe_end = len(combined) - keep
        self._buffer = combined[safe_end:]
        return combined[:safe_end], False

    def flush(self) -> str:
        tail = self._buffer
        self._buffer = ""
        return tail


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
