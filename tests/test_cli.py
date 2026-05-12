from typer.testing import CliRunner

from gemma4_mlx_mac.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.output


def test_cli_models_command() -> None:
    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "gemma-4-e2b-it-4bit" in result.output
