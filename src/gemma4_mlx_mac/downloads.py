from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .models import DEFAULT_MODEL_ID

SnapshotDownloader = Callable[..., str]


class ModelDownloadRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    revision: str | None = None
    cache_dir: str | None = None
    local_dir: str | None = None
    token: str | None = Field(
        default=None,
        description="Optional Hugging Face token. If omitted, huggingface_hub uses local auth.",
    )
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    force_download: bool = False

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model id is required.")
        return value.strip()


class ModelDownloadResult(BaseModel):
    model: str
    path: str
    revision: str | None = None
    source: str = "huggingface"
    files: int = 0
    message: str


class ModelDownloadError(RuntimeError):
    pass


class ModelDownloader:
    def __init__(self, snapshot_downloader: SnapshotDownloader | None = None) -> None:
        self._snapshot_downloader = snapshot_downloader

    def download(self, request: ModelDownloadRequest) -> ModelDownloadResult:
        snapshot_download = self._snapshot_downloader or _load_snapshot_download()

        try:
            path = snapshot_download(
                repo_id=request.model,
                revision=request.revision,
                cache_dir=request.cache_dir,
                local_dir=request.local_dir,
                token=request.token,
                allow_patterns=request.allow_patterns,
                ignore_patterns=request.ignore_patterns,
                force_download=request.force_download,
            )
        except Exception as exc:  # pragma: no cover - exact HF exceptions vary by version.
            raise ModelDownloadError(_friendly_huggingface_error(exc)) from exc

        return ModelDownloadResult(
            model=request.model,
            path=str(path),
            revision=request.revision,
            files=_count_files(path),
            message="Model snapshot is available locally.",
        )


def _load_snapshot_download() -> SnapshotDownloader:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency should be installed.
        raise ModelDownloadError(
            "huggingface-hub is not installed. Reinstall gemma4-mlx-mac to enable downloads."
        ) from exc
    return snapshot_download


def _count_files(path: str | Path) -> int:
    local_path = Path(path)
    if not local_path.exists():
        return 0
    if local_path.is_file():
        return 1
    return sum(1 for child in local_path.rglob("*") if child.is_file())


def _friendly_huggingface_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "401" in message or "unauthorized" in lowered or "gated" in lowered:
        return (
            "Hugging Face denied access to this model. Run `huggingface-cli login`, pass a "
            "token, and make sure you accepted the model license if it is gated."
        )
    if "404" in message or "not found" in lowered:
        return "Hugging Face could not find that model id or revision."
    if "connection" in lowered or "timeout" in lowered:
        return "Could not reach Hugging Face. Check your network connection and try again."
    return f"Hugging Face download failed: {message}"


def download_model_snapshot(**kwargs: Any) -> ModelDownloadResult:
    request = ModelDownloadRequest(**kwargs)
    return ModelDownloader().download(request)
