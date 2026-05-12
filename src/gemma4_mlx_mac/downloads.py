from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from os import devnull
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator
from tqdm.auto import tqdm

from .models import DEFAULT_MODEL_ID

SnapshotDownloader = Callable[..., str]
ProgressCallback = Callable[["ModelDownloadProgress"], None]
DownloadStatus = Literal["queued", "running", "succeeded", "failed"]


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


class ModelDownloadProgress(BaseModel):
    kind: Literal["bytes", "files"]
    completed: int = 0
    total: int | None = None
    description: str = ""


class ModelDownloadJob(BaseModel):
    id: str
    model: str
    revision: str | None = None
    status: DownloadStatus
    source: str = "huggingface"
    bytes_downloaded: int = 0
    bytes_total: int | None = None
    files_downloaded: int = 0
    files_total: int | None = None
    path: str | None = None
    error: str | None = None
    message: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @computed_field
    @property
    def percent(self) -> float | None:
        if self.status == "succeeded":
            return 100.0
        if self.bytes_total and self.bytes_total >= self.bytes_downloaded and self.bytes_total > 0:
            return round(min(self.bytes_downloaded / self.bytes_total, 1) * 100, 1)
        if self.files_total and self.files_total > 0:
            return round(min(self.files_downloaded / self.files_total, 1) * 100, 1)
        return None


class ModelDownloadError(RuntimeError):
    pass


class ModelDownloader:
    def __init__(self, snapshot_downloader: SnapshotDownloader | None = None) -> None:
        self._snapshot_downloader = snapshot_downloader

    def download(
        self,
        request: ModelDownloadRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ModelDownloadResult:
        snapshot_download = self._snapshot_downloader or _load_snapshot_download()
        tqdm_class = _progress_tqdm_class(progress_callback) if progress_callback else None

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
                tqdm_class=tqdm_class,
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


class ModelDownloadJobStore:
    def __init__(
        self,
        downloader: ModelDownloader | None = None,
        max_workers: int = 2,
    ) -> None:
        self._downloader = downloader or ModelDownloader()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, ModelDownloadJob] = {}
        self._lock = Lock()

    def start(self, request: ModelDownloadRequest) -> ModelDownloadJob:
        job = ModelDownloadJob(
            id=uuid4().hex,
            model=request.model,
            revision=request.revision,
            status="queued",
            message="Download queued.",
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id, request)
        return job

    def get(self, job_id: str) -> ModelDownloadJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[ModelDownloadJob]:
        with self._lock:
            return list(self._jobs.values())

    def _run(self, job_id: str, request: ModelDownloadRequest) -> None:
        self._update(
            job_id,
            status="running",
            started_at=datetime.now(UTC),
            message="Connecting to Hugging Face...",
        )

        def on_progress(progress: ModelDownloadProgress) -> None:
            current_job = self.get(job_id)
            if current_job is None or current_job.status != "running":
                return
            if progress.kind == "bytes":
                self._update(
                    job_id,
                    bytes_downloaded=progress.completed,
                    bytes_total=progress.total,
                    message=_progress_message(progress),
                )
            else:
                self._update(
                    job_id,
                    files_downloaded=progress.completed,
                    files_total=progress.total,
                    message=_progress_message(progress),
                )

        try:
            result = self._downloader.download(request, progress_callback=on_progress)
        except ModelDownloadError as exc:
            self._update(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(UTC),
                message=str(exc),
            )
            return

        current_job = self.get(job_id)
        current_files = current_job.files_downloaded if current_job else 0
        current_files_total = current_job.files_total if current_job else None
        final_files_total = max(result.files, current_files_total or 0) or None
        self._update(
            job_id,
            status="succeeded",
            path=result.path,
            files_downloaded=max(result.files, current_files),
            files_total=final_files_total,
            completed_at=datetime.now(UTC),
            message=result.message,
        )

    def _update(self, job_id: str, **values: object) -> ModelDownloadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(update=values)
            self._jobs[job_id] = updated
            return updated


def _load_snapshot_download() -> SnapshotDownloader:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency should be installed.
        raise ModelDownloadError(
            "huggingface-hub is not installed. Reinstall gemma4-mlx-mac to enable downloads."
        ) from exc
    return snapshot_download


def _progress_tqdm_class(progress_callback: ProgressCallback) -> type[tqdm]:
    null_writer = _NullWriter()

    class ProgressTqdm(tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._progress_kind: Literal["bytes", "files"] = (
                "bytes" if kwargs.get("unit") == "B" else "files"
            )
            kwargs.setdefault("file", null_writer)
            super().__init__(*args, **kwargs)
            self._emit_progress()

        def update(self, n: int | float | None = 1) -> bool | None:
            updated = super().update(n)
            self._emit_progress()
            return updated

        def refresh(self, *args: object, **kwargs: object) -> bool | None:
            refreshed = super().refresh(*args, **kwargs)
            self._emit_progress()
            return refreshed

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            super().set_description(desc, refresh=refresh)
            self._emit_progress()

        def _emit_progress(self) -> None:
            total = int(self.total) if self.total else None
            progress_callback(
                ModelDownloadProgress(
                    kind=self._progress_kind,
                    completed=max(int(self.n), 0),
                    total=total,
                    description=str(self.desc or ""),
                )
            )

    return ProgressTqdm


class _NullWriter:
    def write(self, _: str) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return devnull


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


def _progress_message(progress: ModelDownloadProgress) -> str:
    total_is_known = progress.total is not None and progress.total >= progress.completed
    if not total_is_known and progress.kind == "bytes":
        return f"Downloading {progress.completed} bytes; total is still being discovered."
    total = progress.total if total_is_known else "unknown"
    label = "bytes" if progress.kind == "bytes" else "files"
    return f"Downloading {progress.completed} / {total} {label}."


def download_model_snapshot(**kwargs: Any) -> ModelDownloadResult:
    request = ModelDownloadRequest(**kwargs)
    return ModelDownloader().download(request)
