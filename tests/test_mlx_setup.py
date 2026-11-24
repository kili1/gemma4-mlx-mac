from __future__ import annotations

import sys
from time import sleep

from gemma4_mlx_mac import mlx_setup
from gemma4_mlx_mac.mlx_setup import InstallCommand, MlxInstallerJobStore, build_mlx_install_command


def test_build_mlx_install_command_prefers_uv_when_available(monkeypatch, tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text('name = "gemma4-mlx-mac"\n', encoding="utf-8")
    monkeypatch.setattr(mlx_setup, "_find_source_root", lambda: tmp_path)
    monkeypatch.setattr(
        mlx_setup.shutil,
        "which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    cmd = build_mlx_install_command()
    assert cmd.argv[:4] == ["/usr/bin/uv", "pip", "install", "--python"]
    assert cmd.argv[-2:] == ["-e", ".[mlx]"]
    assert cmd.cwd == str(tmp_path)


def test_build_mlx_install_command_falls_back_to_pip(monkeypatch) -> None:
    monkeypatch.setattr(mlx_setup, "_find_source_root", lambda: None)
    monkeypatch.setattr(mlx_setup.shutil, "which", lambda _name: None)

    cmd = build_mlx_install_command()
    assert cmd.argv == [sys.executable, "-m", "pip", "install", "gemma4-mlx-mac[mlx]"]
    assert cmd.cwd is None


def test_mlx_installer_job_store_runs_command_and_tracks_failure(monkeypatch) -> None:
    monkeypatch.setattr(mlx_setup, "is_mlx_available", lambda: False)
    store = MlxInstallerJobStore(
        command_builder=lambda: InstallCommand(
            argv=[sys.executable, "-c", "print('installing...'); raise SystemExit(2)"]
        ),
        max_workers=1,
    )
    try:
        job = store.start()
        for _ in range(50):
            current = store.get(job.id)
            if current and current.status in {"succeeded", "failed"}:
                break
            sleep(0.02)

        finished = store.get(job.id)
        assert finished is not None
        assert finished.status == "failed"
        assert finished.returncode == 2
        assert any("installing..." in line for line in finished.output)
    finally:
        store.shutdown()
