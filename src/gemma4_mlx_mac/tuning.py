from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .adapters import AdapterRegistry
from .datasets import DatasetReport, validate_dataset_dir
from .downloads import get_cached_model_path
from .mlx_setup import is_mlx_available
from .models import DEFAULT_MODEL_ID

TuneStatus = Literal["queued", "running", "ready", "succeeded", "failed", "cancelled"]

_TRAIN_ITER_RE = re.compile(
    r"Iter\s+(?P<iter>\d+):\s+Train loss\s+(?P<loss>[0-9.]+)"
    r"(?:.*?Tokens/sec\s+(?P<tps>[0-9.]+))?"
    r"(?:.*?Peak mem\s+(?P<mem>[0-9.]+)\s*GB)?",
    re.IGNORECASE,
)
_VAL_ITER_RE = re.compile(
    r"Iter\s+(?P<iter>\d+):\s+Val loss\s+(?P<loss>[0-9.]+)",
    re.IGNORECASE,
)
_MAX_LOG_LINES = 200


class TuneRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    data_path: str
    adapter_name: str
    iters: int = Field(default=100, ge=1)
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0)
    fine_tune_type: Literal["lora", "dora", "full"] = "lora"
    num_layers: int | None = Field(default=None, ge=-1)
    grad_checkpoint: bool = False


class TuneJob(BaseModel):
    id: str
    status: TuneStatus
    request: TuneRequest
    dataset: DatasetReport
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_percent: int = Field(default=0, ge=0, le=100)
    current_step: str = "Queued"
    current_iter: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    tokens_per_second: float | None = None
    peak_memory_gb: float | None = None
    adapter_path: str | None = None
    command: list[str] = Field(default_factory=list)
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    message: str


@dataclass(frozen=True)
class ParsedTuneMetrics:
    iteration: int
    train_loss: float | None = None
    val_loss: float | None = None
    tokens_per_second: float | None = None
    peak_memory_gb: float | None = None


LineCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]
ProcessCallback = Callable[[subprocess.Popen[str] | None], None]
CommandRunner = Callable[[list[str], LineCallback, CancelCheck, ProcessCallback], int]
ModelPathResolver = Callable[[str], str | None]


def parse_mlx_lora_line(line: str) -> ParsedTuneMetrics | None:
    """Extracts iteration and loss/throughput/memory metrics from mlx_lm lora stdout."""
    train_match = _TRAIN_ITER_RE.search(line)
    if train_match:
        return ParsedTuneMetrics(
            iteration=int(train_match.group("iter")),
            train_loss=float(train_match.group("loss")),
            tokens_per_second=(
                float(train_match.group("tps")) if train_match.group("tps") else None
            ),
            peak_memory_gb=(
                float(train_match.group("mem")) if train_match.group("mem") else None
            ),
        )

    val_match = _VAL_ITER_RE.search(line)
    if val_match:
        return ParsedTuneMetrics(
            iteration=int(val_match.group("iter")),
            val_loss=float(val_match.group("loss")),
        )

    return None


def build_lora_command(
    request: TuneRequest,
    model_ref: str,
    adapter_path: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Builds the canonical ``mlx_lm lora`` training CLI invocation."""
    data_target = Path(request.data_path).expanduser()
    data_dir = data_target.parent if data_target.is_file() else data_target
    command = [
        python_executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model_ref,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(request.iters),
        "--batch-size",
        str(request.batch_size),
        "--learning-rate",
        str(request.learning_rate),
        "--fine-tune-type",
        request.fine_tune_type,
    ]
    if request.num_layers is not None:
        command.extend(["--num-layers", str(request.num_layers)])
    if request.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def _default_command_runner(
    command: list[str],
    on_line: LineCallback,
    is_cancelled: CancelCheck,
    set_process: ProcessCallback,
) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    set_process(process)
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                on_line(line)
            if is_cancelled():
                _terminate_process(process)
                break
        return process.wait()
    finally:
        set_process(None)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class TuneJobStore:
    def __init__(
        self,
        max_workers: int = 1,
        adapter_registry: AdapterRegistry | None = None,
        command_runner: CommandRunner | None = None,
        mlx_checker: Callable[[], bool] = is_mlx_available,
        model_path_resolver: ModelPathResolver = get_cached_model_path,
    ) -> None:
        self._jobs: dict[str, TuneJob] = {}
        self._cancelled: set[str] = set()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self._adapter_registry = adapter_registry or AdapterRegistry()
        self._command_runner = command_runner or _default_command_runner
        self._mlx_checker = mlx_checker
        self._model_path_resolver = model_path_resolver

    def create(self, request: TuneRequest) -> TuneJob:
        dataset = validate_dataset_dir(request.data_path)
        adapter_dir = self._adapter_registry.adapter_path(request.adapter_name)
        model_ref = self._resolve_model_ref(request.model)
        command = build_lora_command(request, model_ref, adapter_dir)
        created_at = datetime.now(UTC)
        total_examples = sum(file_report.examples for file_report in dataset.files)
        job = TuneJob(
            id=uuid4().hex,
            status="queued",
            request=request,
            dataset=dataset,
            created_at=created_at,
            updated_at=created_at,
            progress_percent=5,
            current_step="Dataset validated",
            adapter_path=str(adapter_dir),
            command=command,
            logs=[
                f"Dataset validated ({total_examples} examples).",
                f"Prepared adapter output path: {adapter_dir}",
            ],
            message="Dataset validated. Queued MLX-LM LoRA training job.",
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id, adapter_dir, command)
        return job

    def get(self, job_id: str) -> TuneJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TuneJob]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> TuneJob | None:
        process: subprocess.Popen[str] | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job
            self._cancelled.add(job_id)
            process = self._processes.get(job_id)
            updated = job.model_copy(
                update={
                    "status": "cancelled",
                    "completed_at": datetime.now(UTC),
                    "current_step": "Cancelled",
                    "message": "Fine-tune job was cancelled.",
                    "logs": [*job.logs, "Cancellation requested by user."][-_MAX_LOG_LINES:],
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated

        if process is not None:
            _terminate_process(process)
        return updated

    def _resolve_model_ref(self, model_id: str) -> str:
        local_candidate = Path(model_id).expanduser()
        if local_candidate.exists():
            return str(local_candidate)
        cached = self._model_path_resolver(model_id)
        return cached or model_id

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _set_process(self, job_id: str, process: subprocess.Popen[str] | None) -> None:
        with self._lock:
            if process is None:
                self._processes.pop(job_id, None)
            else:
                self._processes[job_id] = process

    def _run(self, job_id: str, adapter_dir: Path, command: list[str]) -> None:
        if self._is_cancelled(job_id):
            return

        if not self._mlx_checker():
            error_msg = (
                "MLX training dependencies are not installed. "
                "Run `gemma4-mlx-mac install-mlx` (requires Apple Silicon macOS)."
            )
            self._update(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC),
                current_step="MLX not installed",
                error=error_msg,
                message=error_msg,
                log=error_msg,
            )
            return

        adapter_dir.mkdir(parents=True, exist_ok=True)
        self._update(
            job_id,
            status="running",
            started_at=datetime.now(UTC),
            progress_percent=10,
            current_step="Starting mlx_lm lora",
            message="Running MLX-LM LoRA fine-tuning...",
            log=f"Running: {' '.join(command)}",
        )

        job_snapshot = self.get(job_id)
        total_iters = job_snapshot.request.iters if job_snapshot else 1

        def on_line(line: str) -> None:
            metrics = parse_mlx_lora_line(line)
            updates: dict[str, object] = {}
            if metrics is not None:
                pct = min(95, max(10, 10 + int((metrics.iteration / total_iters) * 85)))
                updates["current_iter"] = metrics.iteration
                updates["progress_percent"] = pct
                updates["current_step"] = f"Iter {metrics.iteration}/{total_iters}"
                if metrics.train_loss is not None:
                    updates["train_loss"] = metrics.train_loss
                    updates["message"] = (
                        f"Iter {metrics.iteration}/{total_iters} "
                        f"(train loss {metrics.train_loss:.3f})"
                    )
                if metrics.val_loss is not None:
                    updates["val_loss"] = metrics.val_loss
                if metrics.tokens_per_second is not None:
                    updates["tokens_per_second"] = metrics.tokens_per_second
                if metrics.peak_memory_gb is not None:
                    updates["peak_memory_gb"] = metrics.peak_memory_gb
            self._update(job_id, log=line, **updates)

        try:
            returncode = self._command_runner(
                command,
                on_line,
                lambda: self._is_cancelled(job_id),
                lambda proc: self._set_process(job_id, proc),
            )
        except Exception as exc:  # pragma: no cover - subprocess OS failures vary.
            error_msg = f"Failed to launch mlx_lm lora: {exc}"
            self._update(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC),
                current_step="Launch failed",
                error=error_msg,
                message=error_msg,
                log=error_msg,
            )
            return

        if self._is_cancelled(job_id):
            return

        if returncode == 0:
            self._update(
                job_id,
                status="succeeded",
                completed_at=datetime.now(UTC),
                progress_percent=100,
                current_step="Completed",
                message=f"Fine-tuning complete. Adapter saved to {adapter_dir}.",
                log=f"Training succeeded. Adapter saved to {adapter_dir}.",
            )
            return

        error_msg = f"mlx_lm lora exited with code {returncode}. Check logs for details."
        self._update(
            job_id,
            status="failed",
            completed_at=datetime.now(UTC),
            current_step="Training failed",
            error=error_msg,
            message=error_msg,
            log=error_msg,
        )

    def _update(self, job_id: str, log: str | None = None, **updates: object) -> TuneJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            logs = [*job.logs]
            if log is not None:
                logs.append(log)
                if len(logs) > _MAX_LOG_LINES:
                    logs = logs[-_MAX_LOG_LINES:]

            updated = job.model_copy(
                update={
                    **updates,
                    "logs": logs,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated
            return updated
