from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[a-zA-Z0-9.+_-]+/[a-zA-Z0-9.+_-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)

_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

_SAFE_AUDIO_FORMATS = frozenset({"wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"})


def _max_media_bytes() -> int:
    raw = os.environ.get("GEMMA4_MLX_MAX_MEDIA_MB", "").strip()
    try:
        mb = int(raw) if raw else 25
    except ValueError:
        mb = 25
    return max(1, mb) * 1024 * 1024


def _allow_remote_media() -> bool:
    return os.environ.get("GEMMA4_MLX_ALLOW_REMOTE_MEDIA") == "1"


_MEDIA_PART_TYPES = frozenset(
    {"image_url", "image", "input_audio", "audio_url", "audio", "video_url", "video"}
)


def message_has_media(content: str | list[dict[str, Any]]) -> bool:
    if isinstance(content, str):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        if str(part.get("type", "")).lower() in _MEDIA_PART_TYPES:
            return True
    return False


@dataclass
class PreparedMultimodalBatch:
    messages: list[dict[str, Any]]
    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    _temp_dir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def has_media(self) -> bool:
        return bool(self.images or self.audios or self.videos)

    def close(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> PreparedMultimodalBatch:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def prepare_multimodal_inputs(
    messages: list[dict[str, Any]],
    *,
    extra_images: list[str] | None = None,
    extra_audios: list[str] | None = None,
    extra_videos: list[str] | None = None,
) -> PreparedMultimodalBatch:
    """Normalizes OpenAI multimodal parts into Gemma 4 message parts and local file paths."""
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    file_counter = 0

    def ensure_temp_dir() -> Path:
        nonlocal temp_dir
        if temp_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="gemma4-media-")
        return Path(temp_dir.name)

    def write_bytes(payload: bytes, suffix: str) -> str:
        nonlocal file_counter
        limit = _max_media_bytes()
        if len(payload) > limit:
            raise ValueError(
                f"Attached media exceeds the {limit // (1024 * 1024)} MB limit."
            )
        file_counter += 1
        target = ensure_temp_dir() / f"media_{file_counter}{suffix}"
        target.write_bytes(payload)
        return str(target)

    def resolve_media_source(source: str, expected_prefix: str) -> str:
        cleaned = source.strip()
        if not cleaned:
            raise ValueError(f"Empty {expected_prefix} source.")

        data_match = _DATA_URI_RE.match(cleaned)
        if data_match:
            mime = data_match.group("mime").lower()
            if not mime.startswith(f"{expected_prefix}/"):
                raise ValueError(f"Expected a {expected_prefix}/* data URI, got {mime}.")
            try:
                raw = base64.b64decode(data_match.group("data"), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"Invalid base64 payload in {expected_prefix} data URI.") from exc
            ext = _MIME_TO_EXT.get(mime, f".{mime.split('/', 1)[1]}")
            return write_bytes(raw, ext)

        parsed = urlparse(cleaned)
        if parsed.scheme in {"http", "https"}:
            if not _allow_remote_media():
                raise ValueError(
                    f"Remote {expected_prefix} URLs are blocked in local-only mode. "
                    "Pass a local file path or base64 data URI, or set "
                    "GEMMA4_MLX_ALLOW_REMOTE_MEDIA=1."
                )
            return cleaned

        if parsed.scheme == "file":
            local_path = Path(unquote(parsed.path)).expanduser()
        else:
            local_path = Path(cleaned).expanduser()

        if not local_path.is_file():
            raise FileNotFoundError(f"Attached {expected_prefix} file does not exist: {local_path}")
        return str(local_path)

    def resolve_input_audio(audio_obj: dict[str, Any]) -> str:
        raw_b64 = str(audio_obj.get("data", "")).strip()
        if not raw_b64:
            raise ValueError("input_audio.data must be a non-empty base64 string.")
        fmt = str(audio_obj.get("format", "wav")).strip().lower().lstrip(".")
        if fmt not in _SAFE_AUDIO_FORMATS:
            raise ValueError(f"Unsupported input_audio format: {fmt!r}.")
        try:
            raw = base64.b64decode(raw_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 payload in input_audio.data.") from exc
        return write_bytes(raw, f".{fmt}")

    images: list[str] = []
    audios: list[str] = []
    videos: list[str] = []
    formatted_messages: list[dict[str, Any]] = []

    try:
        for message in messages:
            role = str(message["role"])
            content = message["content"]
            if isinstance(content, str):
                formatted_messages.append({"role": role, "content": content})
                continue

            parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = str(part.get("type", "")).lower()
                if kind == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append({"type": "text", "text": text})
                elif kind in {"image_url", "image"}:
                    raw_val = part.get("image_url") if kind == "image_url" else part.get("image")
                    url = _extract_url(raw_val)
                    images.append(resolve_media_source(url, "image"))
                    parts.append({"type": "image"})
                elif kind == "input_audio":
                    audio_obj = part.get("input_audio")
                    if not isinstance(audio_obj, dict):
                        raise ValueError(
                            "input_audio part must be an object with 'data' and 'format'."
                        )
                    audios.append(resolve_input_audio(audio_obj))
                    parts.append({"type": "audio"})
                elif kind in {"audio_url", "audio"}:
                    raw_val = part.get("audio_url") if kind == "audio_url" else part.get("audio")
                    url = _extract_url(raw_val)
                    audios.append(resolve_media_source(url, "audio"))
                    parts.append({"type": "audio"})
                elif kind in {"video_url", "video"}:
                    raw_val = part.get("video_url") if kind == "video_url" else part.get("video")
                    url = _extract_url(raw_val)
                    videos.append(resolve_media_source(url, "video"))
                    parts.append({"type": "video"})

            formatted_messages.append({"role": role, "content": parts})

        # Attach top-level CLI/API media arguments (--image/--audio/--video) to the last user turn.
        extra_parts: list[dict[str, Any]] = []
        for src in extra_images or []:
            images.append(resolve_media_source(src, "image"))
            extra_parts.append({"type": "image"})
        for src in extra_audios or []:
            audios.append(resolve_media_source(src, "audio"))
            extra_parts.append({"type": "audio"})
        for src in extra_videos or []:
            videos.append(resolve_media_source(src, "video"))
            extra_parts.append({"type": "video"})

        if extra_parts:
            target_idx = next(
                (
                    idx
                    for idx in range(len(formatted_messages) - 1, -1, -1)
                    if formatted_messages[idx]["role"] == "user"
                ),
                None,
            )
            if target_idx is None:
                formatted_messages.append({"role": "user", "content": extra_parts})
            else:
                existing = formatted_messages[target_idx]["content"]
                if isinstance(existing, str):
                    combined = [*extra_parts, {"type": "text", "text": existing}]
                else:
                    combined = [*extra_parts, *existing]
                formatted_messages[target_idx]["content"] = combined
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise

    return PreparedMultimodalBatch(
        messages=formatted_messages,
        images=images,
        audios=audios,
        videos=videos,
        _temp_dir=temp_dir,
    )


def _extract_url(raw_val: Any) -> str:
    if isinstance(raw_val, dict):
        return str(raw_val.get("url", ""))
    return str(raw_val or "")
