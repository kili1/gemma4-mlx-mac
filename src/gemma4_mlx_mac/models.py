from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_MODEL_ID = "mlx-community/gemma-4-E2B-it-qat-4bit"


class ModelProfile(BaseModel):
    id: str
    label: str
    family: str = "gemma-4"
    size: str
    quantization: str
    recommended_memory_gb: int
    context_length: int = 128_000
    qat: bool = False
    modality: Literal["text", "image-text", "any-to-any"]
    default: bool = False
    downloaded: bool = False
    local_path: str | None = None
    notes: str = Field(default="")


MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        id=DEFAULT_MODEL_ID,
        label="Gemma 4 E2B Instruct QAT 4-bit",
        size="E2B",
        quantization="QAT 4-bit",
        recommended_memory_gb=8,
        context_length=128_000,
        qat=True,
        modality="any-to-any",
        default=True,
        notes="Best first-run Mac profile (~2.9 GB static / ~3.5 GB peak RAM). QAT 4-bit.",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-E4B-it-qat-4bit",
        label="Gemma 4 E4B Instruct QAT 4-bit",
        size="E4B",
        quantization="QAT 4-bit",
        recommended_memory_gb=16,
        context_length=128_000,
        qat=True,
        modality="any-to-any",
        notes="Stronger multimodal & reasoning quality (~4.5 GB static Q4_0 weights).",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-12B-it-qat-4bit",
        label="Gemma 4 12B Unified QAT 4-bit",
        size="12B",
        quantization="QAT 4-bit",
        recommended_memory_gb=16,
        context_length=256_000,
        qat=True,
        modality="any-to-any",
        notes="MacBook Pro sweet spot: 256K context, text+vision+audio (~6.7 GB static).",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        label="Gemma 4 26B A4B MoE QAT 4-bit",
        size="26B-A4B",
        quantization="QAT 4-bit",
        recommended_memory_gb=24,
        context_length=256_000,
        qat=True,
        modality="image-text",
        notes="Fast 4B-active MoE reasoning on 24–36 GB Macs (~14.4 GB static Q4_0).",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-31B-it-qat-4bit",
        label="Gemma 4 31B Instruct QAT 4-bit",
        size="31B",
        quantization="QAT 4-bit",
        recommended_memory_gb=32,
        context_length=256_000,
        qat=True,
        modality="image-text",
        notes="Dense flagship QAT 4-bit for 32+ GB Macs (~17.5 GB static Q4_0).",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-26B-A4B-it-bf16",
        label="Gemma 4 26B A4B Instruct bf16",
        size="26B-A4B",
        quantization="bf16",
        recommended_memory_gb=64,
        context_length=256_000,
        qat=False,
        modality="image-text",
        notes="Full-precision MoE workstation profile (~57.7 GB static).",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-31B-it-bf16",
        label="Gemma 4 31B Instruct bf16",
        size="31B",
        quantization="bf16",
        recommended_memory_gb=96,
        context_length=256_000,
        qat=False,
        modality="image-text",
        notes="Full-precision flagship showcase for high-memory Mac Studio (~69.9 GB static).",
    ),
)


def list_model_profiles() -> list[ModelProfile]:
    return list(MODEL_PROFILES)


def get_model_profile(model_id: str) -> ModelProfile | None:
    return next((profile for profile in MODEL_PROFILES if profile.id == model_id), None)
