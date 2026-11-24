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
    monkeypatch.setattr(cli, "is_mlx_vlm_available", lambda: True)

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


def test_cli_respects_env_default_model_and_serve_flags(monkeypatch) -> None:
    monkeypatch.setenv("GEMMA4_MLX_MODEL", "mlx-community/gemma-4-12B-it-qat-4bit")
    monkeypatch.setenv("GEMMA4_MLX_PORT", "9090")
    served: dict[str, object] = {}

    def fake_uvicorn_run(app_str, host, port, reload):
        served.update({"app": app_str, "host": host, "port": port, "reload": reload})

    monkeypatch.setattr(cli.uvicorn, "run", fake_uvicorn_run)

    result = runner.invoke(app, ["serve", "--preload"])

    assert result.exit_code == 0
    assert served["port"] == 9090
    assert served["host"] == "127.0.0.1"
    assert "mlx-community/gemma-4-12B-it-qat-4bit" in result.output


def test_cli_doctor_reports_mlx_installed_and_ready() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert '"mlx_installed"' in result.output
    assert '"mlx_ready"' in result.output
    assert '"multimodal_ready"' in result.output


def test_cli_chat_passes_multimodal_images_and_audio(monkeypatch, tmp_path) -> None:
    img = tmp_path / "shot.png"
    clip = tmp_path / "voice.wav"
    img.write_bytes(b"png")
    clip.write_bytes(b"wav")
    captured: list[object] = []

    def fake_create_completion(self, request):
        captured.append(request)
        return {"choices": [{"message": {"content": "Saw image and heard audio."}}]}

    monkeypatch.setattr(cli.ChatService, "create_completion", fake_create_completion)

    result = runner.invoke(
        app,
        ["chat", "Describe both", "--image", str(img), "--audio", str(clip)],
    )

    assert result.exit_code == 0
    assert "Saw image and heard audio." in result.output
    assert captured[0].images == [str(img)]
    assert captured[0].audios == [str(clip)]
