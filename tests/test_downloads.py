from pathlib import Path
from threading import Event
from time import sleep

import pytest

from gemma4_mlx_mac import downloads
from gemma4_mlx_mac.downloads import (
    ModelDownloader,
    ModelDownloadJobStore,
    ModelDownloadRequest,
    get_cached_model_path,
)
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID


def test_model_downloader_uses_huggingface_snapshot_download(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.update(kwargs)
        return str(model_dir)

    result = ModelDownloader(fake_snapshot_download).download(
        ModelDownloadRequest(model=DEFAULT_MODEL_ID, revision="main")
    )

    assert calls["repo_id"] == DEFAULT_MODEL_ID
    assert calls["revision"] == "main"
    assert result.path == str(model_dir)
    assert result.files == 1


def test_model_download_request_normalizes_local_dir() -> None:
    request = ModelDownloadRequest(model=DEFAULT_MODEL_ID, local_dir="~/Models/gemma4")

    assert request.local_dir == str(Path("~/Models/gemma4").expanduser())


def test_model_downloader_isolates_subdir_per_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "result"
    model_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(model_dir)

    ModelDownloader(fake_snapshot_download).download(
        ModelDownloadRequest(
            model="mlx-community/gemma-4-12B-it-qat-4bit",
            local_dir=str(tmp_path / "shared"),
            subdir_per_model=True,
        )
    )

    assert captured["local_dir"] == str(
        (tmp_path / "shared" / "gemma-4-12B-it-qat-4bit").resolve()
    )


def test_model_downloader_aggregates_parallel_byte_progress(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    updates: list[tuple[str, int, int | None]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        tqdm_class = kwargs["tqdm_class"]
        bar_a = tqdm_class(total=100, unit="B")
        bar_b = tqdm_class(total=200, unit="B")
        bar_a.update(40)
        bar_b.update(50)
        bar_a.close()
        bar_b.close()
        return str(model_dir)

    ModelDownloader(fake_snapshot_download).download(
        ModelDownloadRequest(model=DEFAULT_MODEL_ID),
        progress_callback=lambda update: updates.append(
            (update.kind, update.completed, update.total)
        ),
    )

    assert ("bytes", 90, 300) in updates


def test_download_job_store_tracks_persists_and_deduplicates(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    state_file = tmp_path / "state.json"
    release = Event()
    started = Event()

    def fake_snapshot_download(**kwargs: object) -> str:
        started.set()
        release.wait(timeout=2)
        tqdm_class = kwargs["tqdm_class"]
        bar = tqdm_class(total=2)
        bar.update(2)
        bar.close()
        return str(model_dir)

    store = ModelDownloadJobStore(
        ModelDownloader(fake_snapshot_download),
        max_workers=2,
        state_file=state_file,
    )
    try:
        first = store.start(ModelDownloadRequest(model=DEFAULT_MODEL_ID))
        assert started.wait(timeout=1)
        duplicate = store.start(ModelDownloadRequest(model=DEFAULT_MODEL_ID))
        assert duplicate.id == first.id

        release.set()
        for _ in range(30):
            job = store.get(first.id)
            if job and job.status == "succeeded":
                break
            sleep(0.01)

        assert job is not None
        assert job.status == "succeeded"
        assert job.files_downloaded >= 1
    finally:
        store.shutdown()

    # Persisted custom paths survive across store restarts.
    reloaded_store = ModelDownloadJobStore(state_file=state_file)
    try:
        assert reloaded_store.downloaded_paths()[DEFAULT_MODEL_ID] == str(model_dir)
    finally:
        reloaded_store.shutdown()


def test_get_cached_model_path_requires_weights(monkeypatch, tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    config = snap / "config.json"
    config.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda **_kwargs: str(config),
    )

    # Partial snapshot with only config.json must NOT be treated as downloaded.
    assert get_cached_model_path(DEFAULT_MODEL_ID) is None

    (snap / "model.safetensors").write_bytes(b"weights")
    assert get_cached_model_path(DEFAULT_MODEL_ID) == str(snap)


def test_model_download_rejects_system_directory() -> None:
    store = ModelDownloadJobStore()
    try:
        with pytest.raises(ValueError, match="system directory"):
            store.start(ModelDownloadRequest(model=DEFAULT_MODEL_ID, local_dir="/etc/gemma4"))
        with pytest.raises(ValueError, match="system directory"):
            store.start(
                ModelDownloadRequest(model=DEFAULT_MODEL_ID, local_dir="/private/etc/gemma4")
            )
    finally:
        store.shutdown()


def test_snapshot_has_weights_checks_sharded_index(tmp_path: Path) -> None:
    snap = tmp_path / "sharded"
    snap.mkdir()
    (snap / "model.safetensors.index.json").write_text(
        '{"weight_map": {"w1": "model-00001.safetensors", "w2": "model-00002.safetensors"}}',
        encoding="utf-8",
    )
    (snap / "model-00001.safetensors").write_bytes(b"1")
    assert downloads._snapshot_has_weights(snap) is False

    (snap / "model-00002.safetensors").write_bytes(b"2")
    assert downloads._snapshot_has_weights(snap) is True
