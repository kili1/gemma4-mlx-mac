# Architecture

`gemma4-mlx-mac` is split into four layers:

1. CLI commands in `gemma4_mlx_mac.cli`.
2. FastAPI routes in `gemma4_mlx_mac.api`.
3. Local services for system detection, model profiles, adapters, datasets, tuning jobs, and inference.
4. A React/Vite frontend that talks to the local API.

The base package avoids importing MLX at module import time. This keeps commands like `doctor`, API tests, and CI useful on machines that do not have Apple Silicon. MLX-backed implementations should be loaded lazily inside service methods.

## Default Model

The default model profile is `mlx-community/gemma-4-e2b-it-4bit`. It is intended to give most Apple Silicon users a quick first success before they choose larger Gemma 4 models.

## Local-Only Defaults

The app should not send prompts, datasets, logs, or metrics to external services unless the user explicitly configures an integration. Model downloads are expected to come from Hugging Face or another user-selected source.
