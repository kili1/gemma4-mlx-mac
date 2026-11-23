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
    assert "gemma-4-E2B-it-qat-4bit" in result.output
    assert "gemma-4-12B-it-qat-4bit" in result.output


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


def test_cli_install_mlx_already_available(monkeypatch) -> None:
    monkeypatch.setattr(cli, "is_mlx_available", lambda: True)

    result = runner.invoke(app, ["install-mlx"])

    assert result.exit_code == 0
    assert "already installed" in result.output


def test_cli_chat_passes_thinking_and_adapter(monkeypatch, tmp_path) -> None:
    adapter_dir = tmp_path / "my-adapter"
    adapter_dir.mkdir()
    captured: list[object] = []

    def fake_create_completion(self, request):
        captured.append(request)
        return {"choices": [{"message": {"content": "Thinking:\nhi\n\nAnswer:\ndone"}}]}

    monkeypatch.setattr(cli.ChatService, "create_completion", fake_create_completion)

    result = runner.invoke(
        app,
        ["chat", "Test prompt", "--thinking", "--adapter", str(adapter_dir)],
    )

    assert result.exit_code == 0
    assert "Answer:" in result.output
    assert len(captured) == 1
    assert captured[0].show_thinking is True
    assert captured[0].adapter_path == str(adapter_dir)
