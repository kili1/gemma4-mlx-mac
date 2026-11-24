# Roadmap

## V1 (Shipped)

- [x] MLX-LM text generation behind OpenAI-compatible `/v1/models` and `/v1/chat/completions`.
- [x] Multi-turn streaming chat in the React workbench with Markdown rendering, Stop control, and native Gemma 4 Thinking Mode.
- [x] Gemma 4 QAT 4-bit and bf16 catalog, Hugging Face downloads with aggregated progress, and verified cache status.
- [x] Real `mlx_lm lora` text LoRA/QLoRA/DoRA execution with live metrics, cancellation, and adapter registry.
- [x] One-click adapter activation/deactivation and unified-memory eviction/unload controls.

## V2 (Planned)

- [ ] Gemma 4 multimodal chat (image, audio, video) through `mlx-vlm`.
- [ ] Image/audio dataset preparation for multimodal fine-tuning.
- [ ] Prompt caching and KV-cache quantization controls for 128K–256K workflows.
- [ ] MTP drafter support for supported Gemma 4 model pairs.
- [ ] Adapter evaluation and side-by-side base vs. tuned response comparison.
- [ ] Signed macOS app bundle / Homebrew formula once the Python install path stabilizes.
