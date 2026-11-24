# Architecture

`gemma4-mlx-mac` is split into four layers:

1. **CLI** (`gemma4_mlx_mac.cli` & `gemma4_mlx_mac.config`) — Typer commands with environment-variable defaults (`GEMMA4_MLX_*`, `HF_TOKEN`).
2. **HTTP API & Security Guard** (`gemma4_mlx_mac.app`, `gemma4_mlx_mac.api`) — FastAPI routes, OpenAI-compatible `/v1/models` and `/v1/chat/completions`, FastAPI `lifespan` executor shutdown, and local-only `Host`/`Origin` middleware.
3. **Local Services** — System readiness (`system.py`), model catalog (`models.py`), Hugging Face downloads with persisted custom paths (`downloads.py`), JSONL dataset validation & synthetic data (`datasets.py`), MLX-LM LoRA subprocess orchestration (`tuning.py`), adapter registry (`adapters.py`), and serialized MLX inference with bounded unified-memory caching (`inference.py`).
4. **React/Vite Workbench** (`frontend/src/`, compiled into `src/gemma4_mlx_mac/web/`) — Multi-turn chat, Markdown + collapsible Thinking Mode rendering, concurrent model downloads, LoRA training monitor, adapter switching, and runtime memory controls.

## Lazy MLX Loading

The base package avoids importing `mlx` or `mlx_lm` at module import time. This keeps `doctor`, dataset validation, API tests, and Linux CI fast and portable. MLX modules are imported lazily inside inference and memory helpers, and `POST /api/inference/install` (or `gemma4-mlx-mac install-mlx`) installs the optional `mlx` extra into the active Python environment.

## Default Model

The default model profile is `mlx-community/gemma-4-E2B-it-qat-4bit` (~2.9 GB static / ~3.5 GB peak unified memory), chosen so almost every Apple Silicon Mac gets a fast first run before moving up to E4B, 12B (256K), 26B-A4B MoE, or 31B.

## Local-Only Defaults

The app binds to `127.0.0.1` by default, blocks non-local `Host` and cross-origin browser requests on mutating endpoints, and never sends prompts, datasets, logs, or telemetry to external services. Outbound network calls happen only when explicitly downloading models from Hugging Face or installing MLX Python packages.
