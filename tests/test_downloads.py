from pathlib import Path

from gemma4_mlx_mac.downloads import ModelDownloader, ModelDownloadRequest
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
