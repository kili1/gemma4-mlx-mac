from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .datasets import DatasetReport, validate_dataset_dir
from .models import DEFAULT_MODEL_ID

TuneStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


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
    message: str


class TuneJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, TuneJob] = {}

    def create(self, request: TuneRequest) -> TuneJob:
        dataset = validate_dataset_dir(request.data_path)
        job = TuneJob(
            id=uuid4().hex,
            status="queued",
            request=request,
            dataset=dataset,
            created_at=datetime.now(UTC),
            message="Dataset validated. MLX execution will run behind this job interface.",
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> TuneJob | None:
        return self._jobs.get(job_id)

    def list(self) -> list[TuneJob]:
        return list(self._jobs.values())
