# Changelog

All notable changes to `gemma4-mlx-mac` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified multimodal Gemma 4 inference via `mlx-vlm` (image, audio, and video) across the OpenAI `/v1/chat/completions` API (`image_url`, `input_audio`, `audio_url`, `video_url`), CLI (`--image`, `--audio`, `--video`, `--stream`), and React Chat UI (file picker, drag-and-drop, clipboard screenshot paste, and inline thumbnails/audio players).
- Gemma 4 QAT 4-bit model catalog (`E2B`, `E4B`, `12B` 256K, `26B-A4B` MoE, `31B`) plus canonical `bf16` workstation profiles.
- Native Gemma 4 Thinking Mode (`enable_thinking=True/False`) with streaming `<|channel>thought...<channel|>` parsing, collapsible UI trace, and multi-turn history stripping.
- Real `python -m mlx_lm lora` fine-tuning subprocess runner with live iteration, train/val loss, tokens/sec, peak memory parsing, and cancellation (`POST /api/tunes/{id}/cancel`, `GET /api/tunes`).
- Active LoRA adapter loading into `mlx_lm.load` and strict=False fallback, plus UI Activate/Deactivate controls.
- OpenAI-compatible `GET /v1/models`, `top_p`, `stop`, `seed`, `repetition_penalty`, `stream_options.include_usage`, content-part arrays, and SSE `role: "assistant"`.
- Unified memory controls: bounded LRU model cache, `POST /api/models/unload`, MLX active/cache/peak memory metrics, and generation serialization lock.
- Environment variable configuration (`GEMMA4_MLX_MODEL`, `GEMMA4_MLX_HOST`, `GEMMA4_MLX_PORT`, `GEMMA4_MLX_PRELOAD`, `HF_TOKEN`) and persisted custom download paths.
- Local-only `Host`/`Origin` security guard middleware and safe-write directory checks.
- Modular React workbench with multi-turn chat history, Markdown + code-block copy, Stop button, concurrent model downloads with subfolder isolation, and Node 22 frontend unit tests.
- Automated CI matrix (macOS 15 + Ubuntu, Python 3.11–3.13, frontend audit/test/bundle-drift check, wheel smoke test), Dependabot, and PyPI Trusted Publishing release workflow.

### Fixed
- Aggregated parallel Hugging Face download progress bars instead of per-file overwrites.
- Verified `.safetensors`/shard presence in Hugging Face cache before marking models downloaded.
- Fixed `doctor` `mlx_ready` check to require both Apple Silicon and installed MLX packages.
- Fixed `serve --open` race condition by waiting for the local socket before opening the browser.
- Updated minimum requirements to `Python >=3.11` and `mlx-lm[train]>=0.31.3` with explicit `tqdm` dependency.
- Resolved all frontend `npm audit` vulnerabilities (0 vulnerabilities).

## [0.1.0] - 2026-05-12

### Added
- Initial local FastAPI backend, Typer CLI, React workbench, Hugging Face model downloads, MLX setup installer, streaming chat, and starter dataset/fine-tuning workflows.
