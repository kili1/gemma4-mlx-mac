# gemma4-mlx-mac

Run, chat with, and fine-tune Gemma 4 locally on Apple Silicon with MLX unified memory.

`gemma4-mlx-mac` is a local-first Mac app for the Gemma 4 open model family. It includes a Python CLI, FastAPI backend, React frontend, Hugging Face model downloads with progress, MLX dependency setup, OpenAI-compatible chat, and starter LoRA/QLoRA fine-tuning workflows.

> Status: pre-alpha, but usable. Local MLX text inference is wired through `mlx_lm.stream_generate`; training execution is still a starter job interface.

## Features

- Local-only operation: no hosted API, no telemetry, and no bundled model weights.
- Default model profile: `mlx-community/gemma-4-e2b-it-4bit`.
- One-click MLX install from the Chat tab when `mlx` / `mlx_lm` are missing.
- Hugging Face model download from the Models tab or CLI, with live progress.
- Optional download folder selection before starting a model download.
- Chat model selection from downloaded models, plus direct local snapshot path entry.
- Streaming chat output with live token count and tokens/sec.
- OpenAI-compatible `/v1/chat/completions` route, including SSE streaming when `stream: true`.
- Starter Fine-tune, Adapters, Models, Chat, and System views in the frontend.

## Quick Start

Install from the repository while the package is not yet published:

```bash
uv tool install "git+https://github.com/kili1/gemma4-mlx-mac"
```

After PyPI publication, install with either:

```bash
uv tool install gemma4-mlx-mac
pipx install gemma4-mlx-mac
```

Run the system check:

```bash
gemma4-mlx-mac doctor
```

Start the local app:

```bash
gemma4-mlx-mac serve --open
```

Open the app, install MLX if prompted, download a model, then chat locally.

## MLX Dependencies

The base package keeps CI and basic commands lightweight. For real local inference or future fine-tuning execution, install the MLX extra:

```bash
gemma4-mlx-mac install-mlx
```

You can also install the extra directly:

```bash
uv tool install "gemma4-mlx-mac[mlx]"
```

The project uses:

- `mlx-lm` for text generation, streaming, adapters, LoRA, and QLoRA.
- `mlx-vlm` for future Gemma 4 image, audio, and video support.

## Models

List built-in model profiles:

```bash
gemma4-mlx-mac models
```

Download the default Gemma 4 MLX model snapshot:

```bash
gemma4-mlx-mac download
```

Download a specific model:

```bash
gemma4-mlx-mac download --model mlx-community/gemma-4-e2b-it-4bit
```

The CLI shows byte/file progress. The web UI also shows download progress and lets you set a target folder such as `~/Models/gemma4` before downloading.

For gated models, authenticate first:

```bash
huggingface-cli login
```

## Chat

The Chat tab supports two model sources:

- **Downloaded**: only shows model profiles available locally.
- **Path**: points directly at an existing local MLX model snapshot directory.

Responses stream token-by-token. The assistant message displays generated token count and tokens/sec as the model runs.

CLI one-shot chat is also available:

```bash
gemma4-mlx-mac chat "Say hello in one sentence."
```

## API

The backend exposes:

- `GET /api/system`
- `GET /api/models`
- `POST /api/models/download`
- `GET /api/models/download/{job_id}`
- `GET /api/models/downloads`
- `GET /api/inference/status`
- `POST /api/inference/install`
- `GET /api/inference/install/{job_id}`
- `POST /v1/chat/completions`
- `POST /api/tunes`
- `GET /api/tunes/{id}`
- `POST /api/datasets/synthetic`
- `GET /api/adapters`
- `POST /api/adapters/{id}/activate`

Non-streaming chat:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e2b-it-4bit",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64
  }'
```

Streaming chat with token metrics:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e2b-it-4bit",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64,
    "stream": true
  }'
```

Streaming responses are Server-Sent Events. Each chunk includes an OpenAI-style `delta.content` plus `metrics.completion_tokens`, `metrics.elapsed_seconds`, and `metrics.tokens_per_second`.

## Fine-Tuning

V1 focuses on text LoRA/QLoRA. The current UI and API validate datasets and create starter jobs; MLX training execution will run behind the same job interface.

The Fine-tune tab can also create a local synthetic starter dataset. Choose a topic, example count, format, and output folder; the app writes a valid `train.jsonl` and fills the dataset path for the fine-tune job form.

Supported JSONL shapes:

```jsonl
{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
{"prompt":"What is the capital of France?","completion":"Paris."}
{"text":"A plain text training example."}
```

See [docs/finetuning.md](docs/finetuning.md) for the expected workflow and memory tradeoffs.

## Development

Install dependencies:

```bash
uv pip install -e ".[dev]"
```

Run checks:

```bash
uv run --extra dev ruff check .
uv run --extra dev pytest
```

Build the frontend into the Python package:

```bash
cd frontend
npm install
npm run build
```

## References

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Run Gemma with MLX](https://ai.google.dev/gemma/docs/integrations/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-LM LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)

## License

Apache-2.0. This repository does not redistribute Gemma model weights.
