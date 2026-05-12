from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Lock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .datasets import DatasetReport, validate_dataset_dir
from .models import DEFAULT_MODEL_ID

TuneStatus = Literal["queued", "running", "ready", "succeeded", "failed", "cancelled"]


class TuneRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    data_path: str
    adapter_name: str
    iters: int = Field(default=100, ge=1)
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0)
    fine_tune_type: Literal["lora", "dora", "full"] = "lora"


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
    logs: list[str] = Field(default_factory=list)
    message: str


class TuneJobStore:
    def __init__(self, max_workers: int = 1, step_delay: float = 0.25) -> None:
        self._jobs: dict[str, TuneJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self._step_delay = step_delay

    def create(self, request: TuneRequest) -> TuneJob:
        dataset = validate_dataset_dir(request.data_path)
        created_at = datetime.now(UTC)
        job = TuneJob(
            id=uuid4().hex,
            status="queued",
            request=request,
            dataset=dataset,
            created_at=created_at,
            updated_at=created_at,
            progress_percent=5,
            current_step="Dataset validated",
            logs=[
                "Dataset validated.",
                "Queued fine-tune preparation job.",
            ],
            message="Dataset validated. Preparing the MLX fine-tune job.",
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return job

    def get(self, job_id: str) -> TuneJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TuneJob]:
        with self._lock:
            return list(self._jobs.values())

    def _run(self, job_id: str) -> None:
        started_at = datetime.now(UTC)
        self._update(
            job_id,
            status="running",
            started_at=started_at,
            progress_percent=18,
            current_step="Preparing trainer command",
            message="Preparing MLX-LM LoRA command and adapter output.",
            log="Preparing MLX-LM LoRA command and adapter output.",
        )
        self._sleep()
        self._update(
            job_id,
            progress_percent=42,
            current_step="Inspecting dataset",
            message="Counting examples and checking train/valid/test files.",
            log="Counting examples and checking train/valid/test files.",
        )
        self._sleep()
        self._update(
            job_id,
            progress_percent=68,
            current_step="Checking run settings",
            message="Checking iteration count, batch size, learning rate, and fine-tune type.",
            log="Checking iteration count, batch size, learning rate, and fine-tune type.",
        )
        self._sleep()
        self._update(
            job_id,
            progress_percent=88,
            current_step="Ready for trainer handoff",
            message="Fine-tune job metadata is ready for the MLX trainer.",
            log="Fine-tune job metadata is ready for the MLX trainer.",
        )
        self._sleep()
        self._update(
            job_id,
            status="ready",
            completed_at=datetime.now(UTC),
            progress_percent=100,
            current_step="Ready",
            message=(
                "Starter progress is complete. MLX training execution is not wired yet, "
                "so no model weights were changed."
            ),
            log="Starter progress complete; waiting for MLX training execution integration.",
        )

    def _sleep(self) -> None:
        if self._step_delay > 0:
            time.sleep(self._step_delay)

    def _update(self, job_id: str, log: str | None = None, **updates: object) -> TuneJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            logs = [*job.logs]
            if log is not None:
                logs.append(log)

            updated = job.model_copy(
                update={
                    **updates,
                    "logs": logs,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated
            return updated
