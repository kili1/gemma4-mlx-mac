from pathlib import Path
from time import sleep

from gemma4_mlx_mac.downloads import (
    ModelDownloader,
    ModelDownloadJobStore,
    ModelDownloadRequest,
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


def test_model_downloader_reports_progress(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    updates: list[tuple[str, int, int | None]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        tqdm_class = kwargs["tqdm_class"]
        bar = tqdm_class(total=100, unit="B")
        bar.update(40)
        bar.close()
        return str(model_dir)

    ModelDownloader(fake_snapshot_download).download(
        ModelDownloadRequest(model=DEFAULT_MODEL_ID),
        progress_callback=lambda update: updates.append(
            (update.kind, update.completed, update.total)
        ),
    )

    assert ("bytes", 40, 100) in updates


def test_download_job_store_tracks_completion(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(**kwargs: object) -> str:
        tqdm_class = kwargs["tqdm_class"]
        bar = tqdm_class(total=2)
        bar.update(1)
        bar.update(1)
        bar.close()
        return str(model_dir)

    store = ModelDownloadJobStore(ModelDownloader(fake_snapshot_download), max_workers=1)
    job = store.start(ModelDownloadRequest(model=DEFAULT_MODEL_ID))

    for _ in range(20):
        job = store.get(job.id)
        if job and job.status == "succeeded":
            break
        sleep(0.01)

    assert job is not None
    assert job.status == "succeeded"
    assert job.files_downloaded >= 1
