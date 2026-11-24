# gemma4-mlx-mac

[![CI](https://github.com/kili1/gemma4-mlx-mac/actions/workflows/ci.yml/badge.svg)](https://github.com/kili1/gemma4-mlx-mac/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon MLX](https://img.shields.io/badge/Apple_Silicon-MLX-black.svg)](https://github.com/ml-explore/mlx)

Run, chat with (text, vision & audio), and fine-tune **Gemma 4** locally on Apple Silicon with MLX unified memory.

`gemma4-mlx-mac` turns the official Gemma 4 MLX stack (`mlx`, `mlx-lm`, and `mlx-vlm`) into a complete local-first Mac workbench: a React web UI, Typer CLI, and hardened OpenAI-compatible server with native Gemma 4 Thinking Mode, multimodal image/audio/video chat, unified-memory guardrails, and managed LoRA/QLoRA fine-tuning.

## Why `gemma4-mlx-mac` vs. Raw `mlx-lm` / `mlx-vlm` Commands?

| Capability | Raw `mlx-lm` / `mlx-vlm` CLI | `gemma4-mlx-mac` |
| --- | --- | --- |
| **Text + Vision + Audio** | Separate `mlx_lm` vs. `mlx_vlm` commands | **Unified** CLI, API, and Web UI — auto-routes text to `mlx-lm` and images/audio/video to `mlx-vlm` |
| **Web Workbench** | None | Multi-turn chat, Markdown + code copy, drag/drop & paste images/audio, Stop button, live tok/s |
| **Gemma 4 Thinking Mode** | Raw `<\|channel>thought...<channel\|>` tokens | Native `enable_thinking` toggle, collapsible Thinking UI, automatic multi-turn history stripping |
| **Model & Unified Memory** | Manual HF pulls; single fixed `--model` | 7 RAM-sized QAT 4-bit & bf16 profiles, concurrent HF downloader, verified cache, LRU eviction, MLX memory card + **Unload** |
| **LoRA / QLoRA Fine-Tuning** | Manual JSONL prep & CLI flags | JSONL validator, synthetic `train`/`valid` generator, live loss/tok-s/peak-mem monitor + cancel, one-click adapter activation |
| **Local Security** | Unprotected dev server | Local-only `Host`/`Origin` CSRF guard, SSRF-blocking local-only media handling, safe-write path checks |

## Requirements

- **Apple Silicon Mac** (M1 / M2 / M3 / M4) running **macOS 15+** recommended for large models.
- **Python 3.11+** (`mlx-lm>=0.31.3` and `mlx-vlm>=0.8.0` installed via the `[mlx]` extra).
- Optional: [uv](https://docs.astral.sh/uv/) for fast isolated CLI installation.

> Commands like `doctor`, `models`, `download`, dataset validation, and tests also run on Linux/Intel machines without MLX installed.

## Quick Start

Install from GitHub (or PyPI once published):

```bash
uv tool install "git+https://github.com/kili1/gemma4-mlx-mac"
```

Check local readiness:

```bash
gemma4-mlx-mac doctor
```

Install the Apple Silicon MLX runtime (`mlx` + `mlx-lm[train]` + `mlx-vlm`):

```bash
gemma4-mlx-mac install-mlx
```

Start the local web workbench and open it in your browser:

```bash
gemma4-mlx-mac serve --open
```

In the web UI:
1. Open **Models** and download `Gemma 4 E2B Instruct QAT 4-bit` (or any profile that fits your Mac).
2. Switch to **Chat** — type a prompt, paste a screenshot (`Cmd+V`), or drag-and-drop images/audio clips, and toggle **Native Thinking** on demand.
3. Use **Fine-tune** to generate a synthetic dataset (or point at your own JSONL folder), train a LoRA/QLoRA adapter, and activate it in **Adapters**.

## Built-in Gemma 4 Profiles

| Profile ID | Size | Quant | Context | Min RAM | Modality | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `mlx-community/gemma-4-E2B-it-qat-4bit` *(default)* | E2B | QAT 4-bit | 128K | 8 GB | any-to-any | Best first-run profile (~2.9 GB static / ~3.5 GB peak) |
| `mlx-community/gemma-4-E4B-it-qat-4bit` | E4B | QAT 4-bit | 128K | 16 GB | any-to-any | Stronger reasoning, vision & audio (~4.5 GB static) |
| `mlx-community/gemma-4-12B-it-qat-4bit` | 12B | QAT 4-bit | 256K | 16 GB | any-to-any | MacBook Pro sweet spot (~6.7 GB static) |
| `mlx-community/gemma-4-26B-A4B-it-qat-4bit` | 26B-A4B | QAT 4-bit | 256K | 24 GB | image-text | Fast 4B-active MoE (~14.4 GB static) |
| `mlx-community/gemma-4-31B-it-qat-4bit` | 31B | QAT 4-bit | 256K | 32 GB | image-text | Dense flagship QAT 4-bit (~17.5 GB static) |
| `mlx-community/gemma-4-26b-a4b-it-bf16` | 26B-A4B | bf16 | 256K | 64 GB | image-text | Full-precision MoE workstation (~57.7 GB static) |
| `mlx-community/gemma-4-31b-it-bf16` | 31B | bf16 | 256K | 96 GB | image-text | Full-precision Mac Studio flagship (~69.9 GB static) |

> You can also point Chat directly at any local MLX model snapshot directory via the **Path** toggle or `--model /path/to/snapshot`.

## CLI Usage

```bash
# Check hardware, Python, mlx-lm, and mlx-vlm readiness
gemma4-mlx-mac doctor
gemma4-mlx-mac doctor --json

# List built-in Gemma 4 profiles
gemma4-mlx-mac models

# Download a model snapshot (set HF_TOKEN in env for gated repos)
gemma4-mlx-mac download
gemma4-mlx-mac download --model mlx-community/gemma-4-12B-it-qat-4bit --local-dir ~/Models/gemma4-12b

# Start the local web UI + OpenAI API server (optionally preloading the model)
gemma4-mlx-mac serve --open --model mlx-community/gemma-4-E2B-it-qat-4bit --preload

# One-shot text, vision, and audio CLI chat (with optional --stream, --thinking, --adapter)
gemma4-mlx-mac chat "Compare MoE vs dense transformers." --thinking --stream
gemma4-mlx-mac chat "Explain this architecture diagram." --image ./diagram.png --thinking
gemma4-mlx-mac chat "Transcribe and summarize this clip." --audio ./meeting.wav
gemma4-mlx-mac chat "Summarize our style guide." --adapter ./adapters/demo-adapter

# Validate a JSONL dataset and run LoRA/QLoRA fine-tuning
gemma4-mlx-mac tune validate ./examples/data
gemma4-mlx-mac tune start --data ./examples/data --adapter-name demo-adapter --iters 100 --grad-checkpoint

# List local adapters
gemma4-mlx-mac adapters
```

## OpenAI-Compatible API (Text + Vision + Audio)

Connect any OpenAI-compatible client (Open WebUI, Continue, Cursor, Python `openai` SDK) to `http://127.0.0.1:8080/v1`:

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-only")

# Text chat
response = client.chat.completions.create(
    model="mlx-community/gemma-4-E2B-it-qat-4bit",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)

# Vision chat (OpenAI image_url with base64 data URI or local path)
img_b64 = base64.b64encode(Path("diagram.png").read_bytes()).decode()
vision = client.chat.completions.create(
    model="mlx-community/gemma-4-E2B-it-qat-4bit",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Describe what you see."},
            ],
        }
    ],
)
print(vision.choices[0].message.content)
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
- `POST /v1/chat/completions` — Text & multimodal (`image_url`, `input_audio`, `audio_url`, `video_url`, plus top-level `images`/`audios`/`videos`) chat completions & SSE streaming (`show_thinking`, `adapter_path`, `top_p`, `stop`, `seed`, `repetition_penalty`, `stream_options`)
- `GET /api/system` — Hardware, OS, Python, `mlx-lm`, and `mlx-vlm` readiness
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
| `GEMMA4_MLX_ALLOW_REMOTE_MEDIA` | unset | Set to `1` to allow `http(s)://` image/audio URLs (blocked by default for local-only privacy) |
| `GEMMA4_MLX_MAX_MEDIA_MB` | `25` | Maximum decoded size per attached image/audio file |
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
- **Vision/audio says `mlx-vlm` is missing**: Run `gemma4-mlx-mac install-mlx` (or click **Install MLX** in the Chat sidebar) to install `mlx-vlm>=0.8.0`.
- **Audio rejected on 26B/31B models**: Audio input is supported on Gemma 4 `any-to-any` models (`E2B`, `E4B`, and `12B`); `26B-A4B` and `31B` are `image-text`.
- **Out-of-memory during fine-tuning**: Lower `--batch-size` to `1`, enable `--grad-checkpoint`, reduce `--num-layers` (e.g. `8`), or start with `E2B` / `12B` QAT 4-bit.
- **Model memory stays high after switching tasks**: Click **Unload** in the Model memory card (or `POST /api/models/unload`) to release MLX Metal allocations immediately.

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
- [Run Gemma with MLX (official docs)](https://gemma.ai/deployment/mlx/)
- [MLX-LM](https://github.com/ml-explore/mlx-lm) & [MLX-LM LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
