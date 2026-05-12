# Contributing

Thanks for helping make local Gemma 4 on Mac friendlier.

## Development Setup

```bash
uv venv
uv pip install -e ".[dev]"
pytest
ruff check .
```

For Apple Silicon MLX work:

```bash
uv pip install -e ".[dev,mlx]"
gemma4-mlx-mac doctor
```

## Contribution Focus

Good early issues include:

- MLX-LM inference implementation behind the chat service.
- LoRA/QLoRA job execution and progress parsing.
- Frontend polish for chat streaming, model download state, and adapter switching.
- Memory recommendations for M1/M2/M3/M4 machines.

Please do not commit model weights, private datasets, adapters, or Hugging Face tokens.
