from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
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
from .config import get_default_host, get_default_model, get_default_port
from .datasets import validate_dataset_dir
from .downloads import (
    ModelDownloader,
    ModelDownloadError,
    ModelDownloadProgress,
    ModelDownloadRequest,
)
from .inference import (
    ChatCompletionRequest,
    ChatMessage,
    ChatService,
    InferenceError,
    InferenceNotReady,
)
from .mlx_setup import build_mlx_install_command, is_mlx_available, is_mlx_vlm_available
from .models import list_model_profiles
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
    table.add_row("MLX installed", "yes" if info.mlx_installed else "no")
    table.add_row("MLX-VLM (vision/audio)", "yes" if info.mlx_vlm_installed else "no")
    table.add_row("MLX ready", "yes" if info.mlx_ready else "no")
    table.add_row("Multimodal ready", "yes" if info.multimodal_ready else "no")
    console.print(table)
    for recommendation in info.recommendations:
        console.print(f"- {recommendation}")


def _open_browser_when_ready(host: str, port: int, url: str, timeout: float = 10.0) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=0.2):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.1)


@app.command()
def serve(
    host: str | None = typer.Option(
        None, help="Host to bind (defaults to GEMMA4_MLX_HOST or 127.0.0.1)."
    ),
    port: int | None = typer.Option(
        None, help="Port to bind (defaults to GEMMA4_MLX_PORT or 8080)."
    ),
    model: str | None = typer.Option(
        None, help="Default model profile (defaults to GEMMA4_MLX_MODEL)."
    ),
    preload: bool = typer.Option(
        False, "--preload", help="Preload the selected model into unified memory on startup."
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the app in a browser."),
) -> None:
    """Start the local FastAPI app and web UI."""
    resolved_host = host or get_default_host()
    resolved_port = port if port is not None else get_default_port()
    resolved_model = model or get_default_model()

    os.environ["GEMMA4_MLX_MODEL"] = resolved_model
    if preload:
        os.environ["GEMMA4_MLX_PRELOAD"] = "1"
    else:
        os.environ.pop("GEMMA4_MLX_PRELOAD", None)

    browser_host = "127.0.0.1" if resolved_host in {"0.0.0.0", "::"} else resolved_host
    url = f"http://{browser_host}:{resolved_port}"
    console.print(f"Starting gemma4-mlx-mac on {url} (default model: {resolved_model})")
    if open_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(resolved_host, resolved_port, url),
            daemon=True,
        ).start()
    uvicorn.run("gemma4_mlx_mac.app:app", host=resolved_host, port=resolved_port, reload=False)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Prompt to send to the local model."),
    model: str | None = typer.Option(None, help="Model profile or local path to use."),
    max_tokens: int = typer.Option(512, min=1, help="Maximum tokens to generate."),
    temperature: float = typer.Option(0.7, min=0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, min=0.01, max=1.0, help="Nucleus sampling top_p."),
    thinking: bool = typer.Option(False, "--thinking", help="Enable native Gemma 4 Thinking Mode."),
    adapter: Path | None = typer.Option(None, "--adapter", help="Optional LoRA adapter directory."),
    images: list[Path] | None = typer.Option(
        None,
        "--image",
        "-i",
        exists=True,
        help="Optional image file(s) for multimodal Gemma 4 vision inference.",
    ),
    audios: list[Path] | None = typer.Option(
        None,
        "--audio",
        "-a",
        exists=True,
        help="Optional audio file(s) for Gemma 4 any-to-any inference (E2B/E4B/12B).",
    ),
    videos: list[Path] | None = typer.Option(
        None,
        "--video",
        exists=True,
        help="Optional video file(s) for multimodal Gemma 4 inference.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream generated tokens live to stdout with throughput summary.",
    ),
) -> None:
    """Send a one-shot text or multimodal (image/audio/video) chat prompt."""
    request = ChatCompletionRequest(
        model=model or get_default_model(),
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        show_thinking=thinking,
        adapter_path=str(adapter) if adapter else None,
        images=[str(path) for path in (images or [])],
        audios=[str(path) for path in (audios or [])],
        videos=[str(path) for path in (videos or [])],
        stream=stream,
    )
    service = ChatService()
    try:
        if stream:
            last_chunk = None
            for chunk in service.stream_tokens(request):
                last_chunk = chunk
                if chunk.text:
                    console.print(chunk.text, end="", markup=False, highlight=False)
            console.print()
            if last_chunk is not None:
                console.print(
                    f"[dim]({last_chunk.completion_tokens} tokens, "
                    f"{last_chunk.tokens_per_second:.1f} tok/s)[/dim]"
                )
            return
        response = service.create_completion(request)
    except (InferenceNotReady, InferenceError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(response["choices"][0]["message"]["content"])


@app.command("install-mlx")
def install_mlx() -> None:
    """Install optional MLX text and multimodal (mlx-vlm) dependencies."""
    if is_mlx_available() and is_mlx_vlm_available():
        console.print("MLX text and multimodal dependencies are already installed.")
        return

    command = build_mlx_install_command()
    console.print(f"Running: {command.display}", markup=False)
    result = subprocess.run(command.argv, cwd=command.cwd, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)

    if not is_mlx_available():
        console.print("[red]Install finished, but mlx_lm is still not importable.[/red]")
        raise typer.Exit(1)
    if not is_mlx_vlm_available():
        console.print(
            "[yellow]mlx_lm is installed, but mlx_vlm is not importable; "
            "vision/audio chat will remain unavailable.[/yellow]"
        )
        return
    console.print("MLX text and multimodal dependencies are installed.")


@app.command("models")
def models_cmd() -> None:
    """List built-in Gemma 4 model profiles."""
    table = Table(title="Model profiles")
    table.add_column("Default")
    table.add_column("Model", no_wrap=True)
    table.add_column("Quant")
    table.add_column("Memory")
    table.add_column("Notes")
    for profile in list_model_profiles():
        table.add_row(
            "*" if profile.default else "",
            profile.id,
            profile.quantization,
            f"{profile.recommended_memory_gb} GB",
            profile.notes,
        )
    console.print(table)


@app.command("download")
def download_cmd(
    model: str | None = typer.Option(None, help="Hugging Face model id to download."),
    revision: str | None = typer.Option(None, help="Optional Hugging Face revision."),
    cache_dir: Path | None = typer.Option(None, help="Optional Hugging Face cache directory."),
    local_dir: Path | None = typer.Option(None, help="Optional target directory."),
    token: str | None = typer.Option(
        None,
        envvar="HF_TOKEN",
        help="Hugging Face token (prefer setting HF_TOKEN env var over passing --token).",
    ),
    force: bool = typer.Option(False, "--force", help="Force a fresh download."),
) -> None:
    """Download a model snapshot from Hugging Face."""
    request = ModelDownloadRequest(
        model=model or get_default_model(),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        token=token,
        force_download=force,
    )
    try:
        result = _download_with_progress(request)
    except (ModelDownloadError, ValueError) as exc:
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
        marker = "* " if adapter.active else "  "
        weights = " [weights]" if adapter.has_weights else ""
        console.print(f"{marker}{adapter.id} -> {adapter.path}{weights}")


@tune_app.command("start")
def tune_start(
    data: Path = typer.Option(..., "--data", exists=True, help="Dataset directory or JSONL file."),
    adapter_name: str = typer.Option(..., "--adapter-name", help="Adapter name to create."),
    model: str | None = typer.Option(None, help="Base model profile or local path."),
    iters: int = typer.Option(100, min=1, help="Training iterations."),
    batch_size: int = typer.Option(1, min=1, help="Per-step batch size."),
    learning_rate: float = typer.Option(1e-5, min=1e-9, help="Optimizer learning rate."),
    fine_tune_type: str = typer.Option(
        "lora", help="Fine-tuning method: lora, dora, or full."
    ),
    num_layers: int | None = typer.Option(
        None, min=-1, help="Number of transformer layers to adapt (-1 for all)."
    ),
    grad_checkpoint: bool = typer.Option(
        False,
        "--grad-checkpoint",
        help="Enable gradient checkpointing to reduce unified memory usage.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for the training job to finish and stream progress.",
    ),
) -> None:
    """Validate a dataset and run an MLX-LM LoRA/QLoRA fine-tuning job."""
    request = TuneRequest(
        model=model or get_default_model(),
        data_path=str(data),
        adapter_name=adapter_name,
        iters=iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fine_tune_type=fine_tune_type,  # type: ignore[arg-type]
        num_layers=num_layers,
        grad_checkpoint=grad_checkpoint,
    )
    store = TuneJobStore()
    try:
        job = store.create(request)
        seen_logs = 0
        if wait:
            while True:
                current = store.get(job.id)
                if current is None:
                    break
                for line in current.logs[seen_logs:]:
                    console.print(line, markup=False)
                seen_logs = len(current.logs)
                if current.status in {"succeeded", "failed", "cancelled"}:
                    job = current
                    break
                time.sleep(0.25)
        console.print_json(data=job.model_dump(mode="json"))
        if wait and job.status == "failed":
            raise typer.Exit(1)
    finally:
        store.shutdown()


@tune_app.command("validate")
def tune_validate(data: Path = typer.Argument(..., exists=True)) -> None:
    """Validate a local JSONL tuning dataset."""
    report = validate_dataset_dir(data)
    console.print_json(data=report.model_dump())


def main() -> None:
    app()
