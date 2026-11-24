# Contributing

Thanks for helping make local Gemma 4 on Apple Silicon faster and easier to use.

## Requirements

- Python 3.11+
- Node.js 22+ (for frontend work)
- Apple Silicon Mac with macOS 15+ (only required for running MLX inference/training live; tests and linting run on macOS and Linux without MLX)

## Development Setup

```bash
uv venv
uv pip install -e ".[dev]"
ruff check .
pytest
```

For Apple Silicon MLX testing:

```bash
uv pip install -e ".[dev,mlx]"
gemma4-mlx-mac doctor
```

For frontend changes:

```bash
cd frontend
npm ci
npm test
npm run build
```

> **Important:** The compiled frontend assets in `src/gemma4_mlx_mac/web/` are bundled into the Python wheel. Whenever you change files in `frontend/`, run `npm run build` and commit the updated `src/gemma4_mlx_mac/web/` files—CI enforces that they stay in sync.

## Contribution Focus

High-impact areas on the V2 roadmap include:

- Multimodal Gemma 4 chat (image, audio, video) through `mlx-vlm`.
- Prompt caching and long-context memory management for 128K–256K sessions.
- Adapter evaluation harnesses and side-by-side base vs. LoRA comparisons.
- MTP drafter support for supported Gemma 4 model pairs.
- Hardware memory tuning recommendations across M1/M2/M3/M4 Macs.

Please do not commit model weights, private datasets, adapters, or Hugging Face tokens.
