from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from .adapters import AdapterRegistry
from .datasets import validate_dataset_dir
from .downloads import (
    ModelDownloader,
    ModelDownloadError,
    ModelDownloadProgress,
    ModelDownloadRequest,
)
from .models import DEFAULT_MODEL_ID, list_model_profiles
from .system import collect_system_info
from .tuning import TuneJobStore, TuneRequest

console = Console()
app = typer.Typer(
    name="gemma4-mlx-mac",
    help="Run, chat with, and fine-tune Gemma 4 locally on Apple Silicon with MLX.",
    no_args_is_help=True,
)
tune_app = typer.Typer(help="Fine-tuning commands.", no_args_is_help=True)
app.add_typer(tune_app, name="tune")


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Check local Mac, Python, and memory readiness."""
    info = collect_system_info()
    if json_output:
        console.print(json.dumps(info.model_dump(), indent=2))
        return

    table = Table(title="gemma4-mlx-mac doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("OS", info.os_version)
    table.add_row("Machine", info.machine)
    table.add_row("Python", info.python_version)
    table.add_row("Memory", f"{info.total_memory_gb:g} GB")
    table.add_row("Apple Silicon", "yes" if info.is_apple_silicon else "no")
    table.add_row("MLX ready", "yes" if info.mlx_ready else "no")
    console.print(table)
    for recommendation in info.recommendations:
        console.print(f"- {recommendation}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8080, help="Port to bind."),
    model: str = typer.Option(DEFAULT_MODEL_ID, help="Model profile to preload later."),
    open_browser: bool = typer.Option(False, "--open", help="Open the app in a browser."),
) -> None:
    """Start the local FastAPI app and web UI."""
    url = f"http://{host}:{port}"
    console.print(f"Starting gemma4-mlx-mac on {url} with model {model}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run("gemma4_mlx_mac.app:app", host=host, port=port, reload=False)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt to send to the local model."),
    model: str = typer.Option(DEFAULT_MODEL_ID, help="Model profile to use."),
) -> None:
    """Send a one-shot chat prompt."""
    console.print(f"Chat command received for {model}.")
    console.print("MLX chat execution will be implemented behind the ChatService interface.")
    console.print(prompt)


@app.command("models")
def models_cmd() -> None:
    """List built-in Gemma 4 model profiles."""
    table = Table(title="Model profiles")
    table.add_column("Default")
    table.add_column("Model", no_wrap=True)
    table.add_column("Memory")
    table.add_column("Notes")
    for profile in list_model_profiles():
        table.add_row(
            "*" if profile.default else "",
            profile.id,
            f"{profile.recommended_memory_gb} GB",
            profile.notes,
        )
    console.print(table)


@app.command("download")
def download_cmd(
    model: str = typer.Option(DEFAULT_MODEL_ID, help="Hugging Face model id to download."),
    revision: str | None = typer.Option(None, help="Optional Hugging Face revision."),
    cache_dir: Path | None = typer.Option(None, help="Optional Hugging Face cache directory."),
    local_dir: Path | None = typer.Option(None, help="Optional target directory."),
    token: str | None = typer.Option(None, help="Optional Hugging Face access token."),
    force: bool = typer.Option(False, "--force", help="Force a fresh download."),
) -> None:
    """Download a model snapshot from Hugging Face."""
    request = ModelDownloadRequest(
        model=model,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        token=token,
        force_download=force,
    )
    try:
        result = _download_with_progress(request)
    except ModelDownloadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"Downloaded {result.model}")
    console.print(f"Path: {result.path}")
    console.print(f"Files: {result.files}")


def _download_with_progress(request: ModelDownloadRequest):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        byte_task = progress.add_task("Bytes", total=None)
        file_task = progress.add_task("Files", total=None, visible=False)

        def on_progress(update: ModelDownloadProgress) -> None:
            if update.kind == "bytes":
                _update_progress_task(progress, byte_task, update)
            else:
                _update_progress_task(progress, file_task, update, visible=False)

        return ModelDownloader().download(request, progress_callback=on_progress)


def _update_progress_task(
    progress: Progress,
    task: TaskID,
    update: ModelDownloadProgress,
    visible: bool = True,
) -> None:
    total = update.total if update.total and update.total > 0 else None
    progress.update(
        task,
        completed=update.completed,
        total=total,
        visible=visible,
    )


@app.command("adapters")
def adapters_cmd(root: Path = typer.Option(Path("adapters"), help="Adapter directory.")) -> None:
    """List local adapter directories."""
    registry = AdapterRegistry(root)
    found = registry.list_adapters()
    if not found:
        console.print("No adapters found.")
        return
    for adapter in found:
        console.print(f"{'* ' if adapter.active else '  '}{adapter.id} -> {adapter.path}")


@tune_app.command("start")
def tune_start(
    data: Path = typer.Option(..., "--data", exists=True, help="Dataset directory or JSONL file."),
    adapter_name: str = typer.Option(..., "--adapter-name", help="Adapter name to create."),
    model: str = typer.Option(DEFAULT_MODEL_ID, help="Base model profile."),
    iters: int = typer.Option(100, min=1, help="Training iterations."),
    batch_size: int = typer.Option(1, min=1, help="Per-step batch size."),
) -> None:
    """Validate a dataset and enqueue a starter tuning job."""
    request = TuneRequest(
        model=model,
        data_path=str(data),
        adapter_name=adapter_name,
        iters=iters,
        batch_size=batch_size,
    )
    store = TuneJobStore()
    job = store.create(request)
    console.print_json(data=job.model_dump(mode="json"))


@tune_app.command("validate")
def tune_validate(data: Path = typer.Argument(..., exists=True)) -> None:
    """Validate a local JSONL tuning dataset."""
    report = validate_dataset_dir(data)
    console.print_json(data=report.model_dump())


def main() -> None:
    app()
