from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

InstallStatus = Literal["queued", "running", "succeeded", "failed"]


class InstallCommand(BaseModel):
    argv: list[str]
    cwd: str | None = None

    @property
    def display(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


class MlxInstallJob(BaseModel):
    id: str
    status: InstallStatus
    command: list[str]
    cwd: str | None = None
    output: list[str] = Field(default_factory=list)
    returncode: int | None = None
    error: str | None = None
    message: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class MlxStatus(BaseModel):
    available: bool
    installing: bool
    install_job_id: str | None = None
    command: list[str]
    cwd: str | None = None
    message: str
    error: str | None = None
    job: MlxInstallJob | None = None


CommandBuilder = Callable[[], InstallCommand]


def is_mlx_available() -> bool:
    importlib.invalidate_caches()
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def build_mlx_install_command() -> InstallCommand:
    source_root = _find_source_root()
    uv = shutil.which("uv")
    if uv:
        command = [uv, "pip", "install", "--python", sys.executable]
        if source_root is not None:
            return InstallCommand(argv=[*command, "-e", ".[mlx]"], cwd=str(source_root))
        return InstallCommand(argv=[*command, "gemma4-mlx-mac[mlx]"])

    if source_root is not None:
        return InstallCommand(
            argv=[sys.executable, "-m", "pip", "install", "-e", ".[mlx]"],
            cwd=str(source_root),
        )
    return InstallCommand(argv=[sys.executable, "-m", "pip", "install", "gemma4-mlx-mac[mlx]"])


class MlxInstallerJobStore:
    def __init__(
        self,
        command_builder: CommandBuilder = build_mlx_install_command,
        max_workers: int = 1,
    ) -> None:
        self._command_builder = command_builder
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, MlxInstallJob] = {}
        self._lock = Lock()

    def status(self) -> MlxStatus:
        current_job = self.current_job()
        command = self._command_builder()
        available = is_mlx_available()
        installing = current_job is not None and current_job.status in {"queued", "running"}
        if available:
            message = "MLX inference dependencies are installed."
        elif installing:
            message = "Installing MLX inference dependencies..."
        else:
            message = "Install the MLX extra to enable local chat inference."

        return MlxStatus(
            available=available,
            installing=installing,
            install_job_id=current_job.id if current_job else None,
            command=command.argv,
            cwd=command.cwd,
            message=message,
            error=None if available or current_job is None else current_job.error,
            job=current_job,
        )

    def start(self) -> MlxInstallJob:
        running_job = self.current_job(statuses={"queued", "running"})
        if running_job is not None:
            return running_job

        command = self._command_builder()
        if is_mlx_available():
            job = MlxInstallJob(
                id=uuid4().hex,
                status="succeeded",
                command=command.argv,
                cwd=command.cwd,
                message="MLX inference dependencies are already installed.",
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
            with self._lock:
                self._jobs[job.id] = job
            return job

        job = MlxInstallJob(
            id=uuid4().hex,
            status="queued",
            command=command.argv,
            cwd=command.cwd,
            message="MLX install queued.",
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return job

    def get(self, job_id: str) -> MlxInstallJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[MlxInstallJob]:
        with self._lock:
            return list(self._jobs.values())

    def current_job(self, statuses: set[InstallStatus] | None = None) -> MlxInstallJob | None:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        for job in jobs:
            if statuses is None or job.status in statuses:
                return job
        return None

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return

        self._update(
            job_id,
            status="running",
            started_at=datetime.now(UTC),
            message="Installing MLX optional dependencies...",
        )
        self._append_output(job_id, f"$ {InstallCommand(argv=job.command).display}")

        try:
            process = subprocess.Popen(
                job.command,
                cwd=job.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._update(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(UTC),
                message="Could not start the MLX install command.",
            )
            return

        assert process.stdout is not None
        for line in process.stdout:
            clean_line = line.rstrip()
            if clean_line:
                self._append_output(job_id, clean_line)

        returncode = process.wait()
        if returncode != 0:
            self._update(
                job_id,
                status="failed",
                returncode=returncode,
                error=f"Install command exited with code {returncode}.",
                completed_at=datetime.now(UTC),
                message="MLX install failed.",
            )
            return

        if not is_mlx_available():
            self._update(
                job_id,
                status="failed",
                returncode=returncode,
                error="Install finished, but mlx_lm is still not importable.",
                completed_at=datetime.now(UTC),
                message="MLX install did not become available to this server.",
            )
            return

        self._update(
            job_id,
            status="succeeded",
            returncode=returncode,
            completed_at=datetime.now(UTC),
            message="MLX inference dependencies are installed.",
        )

    def _append_output(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            output = [*job.output, line][-200:]
            self._jobs[job_id] = job.model_copy(update={"output": output})

    def _update(self, job_id: str, **values: object) -> MlxInstallJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(update=values)
            self._jobs[job_id] = updated
            return updated


def _find_source_root() -> Path | None:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        pyproject = candidate / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            contents = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "gemma4-mlx-mac"' in contents:
            return candidate
    return None
