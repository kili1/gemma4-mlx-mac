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

## Starter Command Shape

```bash
gemma4-mlx-mac tune start \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --data ./examples/data \
  --adapter-name demo-adapter \
  --iters 100
```

The starter validates dataset shape and records a job. The MLX execution layer will map this into `mlx_lm.lora` with LoRA/QLoRA defaults.

## Memory Notes

For larger models, users may need to reduce batch size, reduce the number of trainable layers, enable gradient checkpointing, or increase the macOS wired memory limit. The app should surface these recommendations through `doctor` and the System view rather than hiding them in logs.
