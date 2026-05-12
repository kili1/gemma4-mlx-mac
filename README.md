# gemma4-mlx-mac

Run, chat with, and fine-tune Gemma 4 locally on Apple Silicon with MLX unified memory.

`gemma4-mlx-mac` is an open source starter for a local-first Mac app around the Gemma 4 open model family. It provides a Python CLI, FastAPI backend, OpenAI-compatible chat endpoint, frontend scaffold, and a text LoRA/QLoRA fine-tuning path built around MLX.

> Status: pre-alpha scaffold. The public API shape, project layout, docs, and tests are in place. Full MLX inference and training execution will land behind the interfaces in this starter.

## Goals

- One easy install for Mac users: `uv tool install gemma4-mlx-mac` or `pipx install gemma4-mlx-mac`.
- Local-only by default. No hosted API, no telemetry, and no bundled model weights.
- Sensible first-run default: `mlx-community/gemma-4-e2b-it-4bit`.
- Clear upgrade path for larger Gemma 4 models, adapters, prompt caching, and multimodal support.
- Friendly fine-tuning for chat, completion, and plain text JSONL datasets.

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

The Chat tab only lets you select downloaded model profiles by default. You can also switch to Path and point chat at an existing local MLX model snapshot directory.

If the MLX optional dependencies are not installed, the Chat tab shows an Install MLX action that installs the `mlx` extra into the current Python environment and streams the local install log.

List the built-in model profiles:

```bash
gemma4-mlx-mac models
```

Download the default Gemma 4 MLX model snapshot from Hugging Face:

```bash
gemma4-mlx-mac download
```

The CLI shows live byte/file progress. The web UI starts downloads in the background and shows progress in the Models tab.

For gated models, authenticate first:

```bash
huggingface-cli login
gemma4-mlx-mac download --model mlx-community/gemma-4-e2b-it-4bit
```

## MLX Dependencies

The base package intentionally keeps CI and docs lightweight. On Apple Silicon, install the MLX extras before running real local inference or fine-tuning:

```bash
uv tool install "gemma4-mlx-mac[mlx]"
```

From a source checkout or existing install, you can also run:

```bash
gemma4-mlx-mac install-mlx
```

The project is designed around:

- `mlx-lm` for text generation, serving integration, adapters, LoRA, and QLoRA.
- `mlx-vlm` for future Gemma 4 image, audio, and video support.

## API Shape

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
- `POST /api/adapters/{id}/activate`

The OpenAI-compatible chat route uses `mlx_lm.stream_generate` after the selected model is downloaded and the MLX extra is installed.

## Fine-Tuning

V1 focuses on text LoRA/QLoRA. Supported dataset shapes follow the MLX-LM conventions:

```jsonl
{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
{"prompt":"What is the capital of France?","completion":"Paris."}
{"text":"A plain text training example."}
```

See [docs/finetuning.md](docs/finetuning.md) for the expected workflow and memory tradeoffs.

## References

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Run Gemma with MLX](https://ai.google.dev/gemma/docs/integrations/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-LM LoRA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)

## License

Apache-2.0. This repository does not redistribute Gemma model weights.
