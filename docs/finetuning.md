# Fine-Tuning

V1 focuses on text LoRA/QLoRA using MLX-LM.

## Dataset Formats

`train.jsonl` is required for training. `valid.jsonl` and `test.jsonl` are optional.

Supported JSONL records:

```jsonl
{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
{"prompt":"What is the capital of France?","completion":"Paris."}
{"text":"A plain text training example."}
```

Each example must be one JSON object on one line.

## CLI Fine-Tuning Command

```bash
gemma4-mlx-mac tune start \
  --model mlx-community/gemma-4-E2B-it-qat-4bit \
  --data ./examples/data \
  --adapter-name demo-adapter \
  --iters 100 \
  --wait
```

`TuneJobStore` validates the dataset, resolves any local/cached model snapshot, and runs `python -m mlx_lm lora` as a managed subprocess. The backend parses iteration progress, train/val loss, throughput, and peak memory live (`GET /api/tunes/{id}`), supports cancellation (`POST /api/tunes/{id}/cancel`), and writes adapter weights into `adapters/<adapter_name>` so activating the adapter routes Chat completions through `adapter_path`.

## Memory Notes

For larger models, users may need to reduce batch size, reduce the number of trainable layers, enable gradient checkpointing, or increase the macOS wired memory limit. The app should surface these recommendations through `doctor` and the System view rather than hiding them in logs.
