# gemma4-mlx-mac

[![CI](https://github.com/kili1/gemma4-mlx-mac/actions/workflows/ci.yml/badge.svg)](https://github.com/kili1/gemma4-mlx-mac/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon MLX](https://img.shields.io/badge/Apple_Silicon-MLX-black.svg)](https://github.com/ml-explore/mlx)

Run, chat with, and fine-tune **Gemma 4** locally on Apple Silicon with MLX unified memory.

`gemma4-mlx-mac` is a local-first macOS workbench and OpenAI-compatible server for the Gemma 4 open model family. It bundles a Python CLI, FastAPI backend, React web UI, Hugging Face snapshot downloader with aggregated progress, native Gemma 4 Thinking Mode, and real `mlx_lm lora` fine-tuning with live loss/memory telemetry and one-click LoRA adapter switching.

## Highlights

- **100% Local-First**: Binds to `127.0.0.1` by default with Host/Origin protection, safe-write path checks, zero telemetry, and no bundled model weights.
- **Gemma 4 QAT 4-bit Catalog**: Curated profiles from **E2B** (8 GB Macs) through **12B Unified (256K)**, **26B-A4B MoE**, **31B QAT 4-bit**, and full-precision **bf16** Mac Studio models.
- **Native Thinking Mode**: Passes `enable_thinking=True/False` directly to Gemma 4's Jinja chat template, parses `<|channel>thought...<channel|>` streams into a collapsible Thinking panel, and strips prior thought traces from multi-turn history.
- **Unified Memory Guardrails**: Serializes MLX/Metal generation, evicts idle models automatically, exposes `POST /api/models/unload` (plus an **Unload** button in the UI), and reports native MLX active/cache/peak memory.
- **OpenAI-Compatible API**: Drop-in `GET /v1/models` and `POST /v1/chat/completions` (SSE streaming, `top_p`, `stop`, `seed`, `repetition_penalty`, `stream_options.include_usage`) for tools like Open WebUI, Continue, Cursor, and the OpenAI Python SDK.
- **Real LoRA / QLoRA Fine-Tuning**: Validates JSONL datasets (or generates synthetic train+valid starter sets), runs `python -m mlx_lm lora`, streams live iteration/loss/throughput/peak-memory metrics, supports cancellation, and activates adapters in Chat with one click.

## Requirements

- **Apple Silicon Mac** (M1 / M2 / M3 / M4) running **macOS 15+** recommended for large models.
- **Python 3.11+** (`mlx-lm>=0.31.3` required for Gemma 4 support).
- Optional: [uv](https://docs.astral.sh/uv/) for fast isolated CLI installation.

> Commands like `doctor`, `models`, `download`, dataset validation, and API/unit tests also run on Linux/Intel machines without MLX installed.

## Quick Start

Install from GitHub (or PyPI once published):

```bash
uv tool install "git+https://github.com/kili1/gemma4-mlx-mac"
```

Check local readiness:

```bash
gemma4-mlx-mac doctor
```

Install the Apple Silicon MLX runtime (`mlx` + `mlx-lm[train]`):

```bash
gemma4-mlx-mac install-mlx
```

Start the local web workbench and open it in your browser:

```bash
gemma4-mlx-mac serve --open
```

In the web UI:
1. Open **Models** and download `Gemma 4 E2B Instruct QAT 4-bit` (or any profile that fits your Mac).
2. Switch to **Chat** for multi-turn local inference with Markdown rendering, native Thinking Mode, parameter controls, and live tokens/sec.
3. Use **Fine-tune** to generate a synthetic dataset (or point at your own JSONL folder), train a LoRA/QLoRA adapter, and activate it in **Adapters**.

## Built-in Gemma 4 Profiles

| Profile ID | Size | Quant | Context | Min RAM | Modality | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `mlx-community/gemma-4-E2B-it-qat-4bit` *(default)* | E2B | QAT 4-bit | 128K | 8 GB | any-to-any | Best first-run profile (~2.9 GB static / ~3.5 GB peak) |
| `mlx-community/gemma-4-E4B-it-qat-4bit` | E4B | QAT 4-bit | 128K | 16 GB | any-to-any | Stronger reasoning & multimodal (~4.5 GB static) |
| `mlx-community/gemma-4-12B-it-qat-4bit` | 12B | QAT 4-bit | 256K | 16 GB | any-to-any | MacBook Pro sweet spot (~6.7 GB static) |
| `mlx-community/gemma-4-26B-A4B-it-qat-4bit` | 26B-A4B | QAT 4-bit | 256K | 24 GB | image-text | Fast 4B-active MoE (~14.4 GB static) |
| `mlx-community/gemma-4-31B-it-qat-4bit` | 31B | QAT 4-bit | 256K | 32 GB | image-text | Dense flagship QAT 4-bit (~17.5 GB static) |
| `mlx-community/gemma-4-26b-a4b-it-bf16` | 26B-A4B | bf16 | 256K | 64 GB | image-text | Full-precision MoE workstation (~57.7 GB static) |
| `mlx-community/gemma-4-31b-it-bf16` | 31B | bf16 | 256K | 96 GB | image-text | Full-precision Mac Studio flagship (~69.9 GB static) |

> You can also point Chat directly at any local MLX model snapshot directory via the **Path** toggle or `--model /path/to/snapshot`.

## CLI Usage

```bash
# Check hardware, Python, and MLX installation
gemma4-mlx-mac doctor
gemma4-mlx-mac doctor --json

# List built-in Gemma 4 profiles
gemma4-mlx-mac models

# Download a model snapshot (set HF_TOKEN in env for gated repos)
gemma4-mlx-mac download
gemma4-mlx-mac download --model mlx-community/gemma-4-12B-it-qat-4bit --local-dir ~/Models/gemma4-12b

# Start the local web UI + OpenAI API server (optionally preloading the model)
gemma4-mlx-mac serve --open --model mlx-community/gemma-4-E2B-it-qat-4bit --preload

# One-shot CLI chat with optional Thinking Mode and LoRA adapter
gemma4-mlx-mac chat "Compare MoE vs dense transformers." --thinking
gemma4-mlx-mac chat "Summarize our style guide." --adapter ./adapters/demo-adapter

# Validate a JSONL dataset and run LoRA/QLoRA fine-tuning
gemma4-mlx-mac tune validate ./examples/data
gemma4-mlx-mac tune start --data ./examples/data --adapter-name demo-adapter --iters 100 --grad-checkpoint

# List local adapters
gemma4-mlx-mac adapters
```

## OpenAI-Compatible API

Connect any OpenAI-compatible client (Open WebUI, Continue, Cursor, Python `openai` SDK) to `http://127.0.0.1:8080/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-only")

response = client.chat.completions.create(
    model="mlx-community/gemma-4-E2B-it-qat-4bit",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    temperature=0.7,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

Or stream with `curl`:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-E2B-it-qat-4bit",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64,
    "stream": true
  }'
```

### API Endpoints

- `GET /v1/models` — OpenAI-compatible model listing
- `POST /v1/chat/completions` — Chat completions & SSE streaming (supports `show_thinking`, `adapter_path`, `top_p`, `stop`, `seed`, `repetition_penalty`, `stream_options`)
- `GET /api/system` — Hardware, OS, Python, and MLX readiness
- `GET /api/models`, `POST /api/models/download`, `GET /api/models/download/{job_id}`, `GET /api/models/downloads`, `POST /api/models/unload`
- `GET /api/inference/status`, `POST /api/inference/install`, `GET /api/inference/install/{job_id}`
- `GET /api/tunes`, `POST /api/tunes`, `GET /api/tunes/{id}`, `POST /api/tunes/{id}/cancel`
- `POST /api/datasets/synthetic`
- `GET /api/adapters`, `POST /api/adapters/{id}/activate`, `POST /api/adapters/deactivate`

## Configuration (Environment Variables)

See [.env.example](.env.example):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMMA4_MLX_MODEL` | `mlx-community/gemma-4-E2B-it-qat-4bit` | Default model for CLI & server |
| `GEMMA4_MLX_HOST` | `127.0.0.1` | Bind host for `serve` |
| `GEMMA4_MLX_PORT` | `8080` | Bind port for `serve` |
| `GEMMA4_MLX_PRELOAD` | unset | Set to `1` (or pass `--preload`) to load the default model at startup |
| `GEMMA4_MLX_STATE_FILE` | `~/.cache/gemma4-mlx-mac/state.json` | Persists custom download folders across restarts |
| `GEMMA4_MLX_ALLOWED_ROOTS` | cwd, `~`, `/tmp` | Path-separated directories allowed for API dataset/download writes |
| `GEMMA4_MLX_ALLOW_REMOTE` | unset | Set to `1` only if you intentionally expose the server beyond localhost |
| `HF_TOKEN` | unset | Hugging Face access token for gated repos (preferred over `--token`) |

## Fine-Tuning

V1 runs text LoRA/QLoRA/DoRA via `python -m mlx_lm lora`. Supported JSONL formats (use one consistent format per file):

```jsonl
{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
{"prompt":"What is the capital of France?","completion":"Paris."}
{"text":"A plain text training example."}
```

See [docs/finetuning.md](docs/finetuning.md) and [docs/architecture.md](docs/architecture.md) for details.

## Troubleshooting

- **Gated or 401 Hugging Face errors**: Authenticate once with `huggingface-cli login` or export `HF_TOKEN`, and accept the model's license terms on Hugging Face.
- **Out-of-memory during fine-tuning**: Lower `--batch-size` to `1`, enable `--grad-checkpoint`, reduce `--num-layers` (e.g. `8`), or start with `E2B` / `12B` QAT 4-bit.
- **Model memory stays high after switching tasks**: Click **Unload** in the Model memory card (or `POST /api/models/unload`) to release MLX Metal allocations immediately.
- **`mlx_lm` fails to load Gemma 4**: Upgrade to `mlx-lm>=0.31.3` on Python 3.11+ via `gemma4-mlx-mac install-mlx`.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
ruff check .
pytest
```

Frontend development and production bundle rebuild:

```bash
cd frontend
npm ci
npm test
npm run build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [CHANGELOG.md](CHANGELOG.md) for release notes.

## Disclaimer & License

- **Code License**: [Apache-2.0](LICENSE).
- **Model Weights**: This repository does **not** redistribute Gemma model weights. Downloaded models remain subject to [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms) and each Hugging Face repository's license.
- **Affiliation**: `gemma4-mlx-mac` is an independent open-source project and is not officially affiliated with or endorsed by Google or Apple.

## References

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Run Gemma with MLX](https://ai.google.dev/gemma/docs/integrations/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm) & [MLX-LM LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
