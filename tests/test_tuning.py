from __future__ import annotations

from pathlib import Path
from threading import Event
from time import sleep

import pytest

from gemma4_mlx_mac.adapters import AdapterRegistry
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID
from gemma4_mlx_mac.tuning import (
    TuneJobStore,
    TuneRequest,
    build_lora_command,
    parse_mlx_lora_line,
)

EXAMPLES_DATA = str(Path(__file__).resolve().parents[1] / "examples" / "data")


def test_parse_mlx_lora_train_and_val_lines() -> None:
    train = parse_mlx_lora_line(
        "Iter 10: Train loss 2.134, Learning Rate 1.000e-05, It/sec 1.420, "
        "Tokens/sec 420.500, Trained Tokens 3200, Peak mem 5.120 GB"
    )
    assert train is not None
    assert train.iteration == 10
    assert train.train_loss == pytest.approx(2.134)
    assert train.tokens_per_second == pytest.approx(420.5)
    assert train.peak_memory_gb == pytest.approx(5.12)

    val = parse_mlx_lora_line("Iter 10: Val loss 2.041, Val took 1.234s")
    assert val is not None
    assert val.iteration == 10
    assert val.val_loss == pytest.approx(2.041)

    assert parse_mlx_lora_line("Loading pretrained model...") is None


def test_build_lora_command_includes_flags(tmp_path) -> None:
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"text":"hi"}\n', encoding="utf-8")
    adapter_dir = tmp_path / "adapters" / "demo"

    request = TuneRequest(
        model=DEFAULT_MODEL_ID,
        data_path=str(data_file),
        adapter_name="demo",
        iters=50,
        batch_size=2,
        learning_rate=2e-4,
        fine_tune_type="lora",
        num_layers=8,
        grad_checkpoint=True,
    )
    cmd = build_lora_command(request, "/models/gemma4", adapter_dir, python_executable="python3")

    assert cmd[:4] == ["python3", "-m", "mlx_lm", "lora"]
    assert "--model" in cmd and "/models/gemma4" in cmd
    assert "--data" in cmd and str(tmp_path) in cmd
    assert "--adapter-path" in cmd and str(adapter_dir) in cmd
    assert "--num-layers" in cmd and "8" in cmd
    assert "--grad-checkpoint" in cmd
    assert "--val-batches" in cmd and "0" in cmd


def test_tune_job_store_runs_mlx_lora_and_tracks_metrics(tmp_path) -> None:
    registry = AdapterRegistry(tmp_path / "adapters")

    def fake_runner(command, on_line, _is_cancelled, _set_process):
        assert "-m" in command and "mlx_lm" in command and "lora" in command
        on_line("Iter 5: Train loss 2.450, Tokens/sec 310.2, Peak mem 4.10 GB")
        on_line("Iter 10: Val loss 1.980, Val took 0.8s")
        on_line("Iter 10: Train loss 1.820, Tokens/sec 325.0, Peak mem 4.15 GB")
        adapter_dir = registry.adapter_path("my-adapter")
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        return 0

    store = TuneJobStore(
        adapter_registry=registry,
        command_runner=fake_runner,
        mlx_checker=lambda: True,
        model_path_resolver=lambda _model_id: "/tmp/cached-gemma4",
    )
    job = store.create(
        TuneRequest(
            model=DEFAULT_MODEL_ID,
            data_path=EXAMPLES_DATA,
            adapter_name="my-adapter",
            iters=10,
        )
    )

    for _ in range(40):
        current = store.get(job.id)
        if current and current.status in {"succeeded", "failed", "cancelled"}:
            break
        sleep(0.01)

    finished = store.get(job.id)
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.progress_percent == 100
    assert finished.current_iter == 10
    assert finished.train_loss == pytest.approx(1.82)
    assert finished.val_loss == pytest.approx(1.98)
    assert finished.peak_memory_gb == pytest.approx(4.15)
    assert finished.adapter_path == str(tmp_path / "adapters" / "my-adapter")

    listed = registry.list_adapters()
    assert len(listed) == 1
    assert listed[0].id == "my-adapter"
    assert listed[0].has_weights is True


def test_tune_job_store_fails_cleanly_when_mlx_missing(tmp_path) -> None:
    store = TuneJobStore(
        adapter_registry=AdapterRegistry(tmp_path / "adapters"),
        mlx_checker=lambda: False,
    )
    job = store.create(
        TuneRequest(
            model=DEFAULT_MODEL_ID,
            data_path=EXAMPLES_DATA,
            adapter_name="demo",
            iters=5,
        )
    )

    for _ in range(40):
        current = store.get(job.id)
        if current and current.status == "failed":
            break
        sleep(0.01)

    failed = store.get(job.id)
    assert failed is not None
    assert failed.status == "failed"
    assert "install-mlx" in (failed.error or "")


def test_tune_job_store_supports_cancellation(tmp_path) -> None:
    started = Event()
    release = Event()

    def slow_runner(_command, on_line, is_cancelled, _set_process):
        started.set()
        for _ in range(50):
            if is_cancelled():
                return -15
            sleep(0.01)
        release.wait(timeout=1)
        return 0

    store = TuneJobStore(
        adapter_registry=AdapterRegistry(tmp_path / "adapters"),
        command_runner=slow_runner,
        mlx_checker=lambda: True,
    )
    job = store.create(
        TuneRequest(
            model=DEFAULT_MODEL_ID,
            data_path=EXAMPLES_DATA,
            adapter_name="cancel-me",
            iters=10,
        )
    )
    assert started.wait(timeout=1)
    cancelled = store.cancel(job.id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"


def test_adapter_registry_rejects_path_traversal(tmp_path) -> None:
    registry = AdapterRegistry(tmp_path / "adapters")
    with pytest.raises(ValueError, match="Adapter name"):
        registry.adapter_path("../escape")


def test_real_subprocess_runner_streams_stdout_and_succeeds(tmp_path) -> None:
    import sys

    from gemma4_mlx_mac.tuning import _default_command_runner

    lines: list[str] = []
    script = (
        "import sys; "
        "print('Iter 1: Train loss 2.500, Tokens/sec 100.0, Peak mem 3.20 GB', flush=True); "
        "print('Iter 2: Val loss 2.100, Val took 0.5s', file=sys.stderr, flush=True)"
    )
    rc = _default_command_runner(
        [sys.executable, "-c", script],
        lines.append,
        lambda: False,
        lambda _proc: None,
    )

    assert rc == 0
    assert len(lines) == 2
    assert "Train loss 2.500" in lines[0]
    assert "Val loss 2.100" in lines[1]


def test_real_subprocess_runner_terminates_cleanly_on_cancel(tmp_path) -> None:
    import sys

    from gemma4_mlx_mac.tuning import _default_command_runner

    registry = AdapterRegistry(tmp_path / "adapters")
    started = Event()

    def real_sleep_runner(command, on_line, is_cancelled, set_process):
        del command
        script = (
            "import time; "
            "print('Iter 1: Train loss 2.000', flush=True); "
            "time.sleep(30)"
        )

        def tracked_line(line: str) -> None:
            on_line(line)
            started.set()

        return _default_command_runner(
            [sys.executable, "-c", script],
            tracked_line,
            is_cancelled,
            set_process,
        )

    store = TuneJobStore(
        adapter_registry=registry,
        command_runner=real_sleep_runner,
        mlx_checker=lambda: True,
    )
    job = store.create(
        TuneRequest(
            model=DEFAULT_MODEL_ID,
            data_path=EXAMPLES_DATA,
            adapter_name="real-cancel",
            iters=10,
        )
    )
    assert started.wait(timeout=3)
    cancelled = store.cancel(job.id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"


def test_tune_job_store_handles_nonzero_exit_code(tmp_path) -> None:
    store = TuneJobStore(
        adapter_registry=AdapterRegistry(tmp_path / "adapters"),
        command_runner=lambda _cmd, on_line, _c, _p: (on_line("CUDA/Metal OOM"), 137)[1],
        mlx_checker=lambda: True,
    )
    job = store.create(
        TuneRequest(
            model=DEFAULT_MODEL_ID,
            data_path=EXAMPLES_DATA,
            adapter_name="oom-run",
            iters=5,
        )
    )

    for _ in range(40):
        current = store.get(job.id)
        if current and current.status == "failed":
            break
        sleep(0.01)

    failed = store.get(job.id)
    assert failed is not None
    assert failed.status == "failed"
    assert "exited with code 137" in (failed.error or "")
    assert "CUDA/Metal OOM" in failed.logs
