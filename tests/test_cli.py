from typer.testing import CliRunner

from gemma4_mlx_mac import cli
from gemma4_mlx_mac.cli import app
from gemma4_mlx_mac.downloads import ModelDownloadResult
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.output


def test_cli_models_command() -> None:
    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "gemma-4-e2b-it-4bit" in result.output


def test_cli_download_command(monkeypatch) -> None:
    def fake_download(self, request, progress_callback=None):
        assert request.model == DEFAULT_MODEL_ID
        return ModelDownloadResult(
            model=request.model,
            path="/tmp/model",
            files=2,
            message="ok",
        )

    monkeypatch.setattr(cli.ModelDownloader, "download", fake_download)

    result = runner.invoke(app, ["download"])

    assert result.exit_code == 0
    assert "Downloaded" in result.output
