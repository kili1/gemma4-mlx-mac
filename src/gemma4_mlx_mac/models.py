from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"


class ModelProfile(BaseModel):
    id: str
    label: str
    family: str = "gemma-4"
    size: str
    quantization: str
    recommended_memory_gb: int
    modality: Literal["text", "image-text", "any-to-any"]
    default: bool = False
    notes: str = Field(default="")


MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        id=DEFAULT_MODEL_ID,
        label="Gemma 4 E2B Instruct 4-bit",
        size="E2B",
        quantization="4-bit",
        recommended_memory_gb=16,
        modality="any-to-any",
        default=True,
        notes="Best first-run profile for common Apple Silicon Macs.",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-e4b-it-4bit",
        label="Gemma 4 E4B Instruct 4-bit",
        size="E4B",
        quantization="4-bit",
        recommended_memory_gb=24,
        modality="any-to-any",
        notes="Better quality with a higher memory and download cost.",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-26B-A4B-it-bf16",
        label="Gemma 4 26B A4B Instruct bf16",
        size="26B-A4B",
        quantization="bf16",
        recommended_memory_gb=64,
        modality="image-text",
        notes="Large workstation profile. Consider MTP drafter support later.",
    ),
    ModelProfile(
        id="mlx-community/gemma-4-31B-it-bf16",
        label="Gemma 4 31B Instruct bf16",
        size="31B",
        quantization="bf16",
        recommended_memory_gb=96,
        modality="image-text",
        notes="Showcase profile for high-memory Macs.",
    ),
)


def list_model_profiles() -> list[ModelProfile]:
    return list(MODEL_PROFILES)


def get_model_profile(model_id: str) -> ModelProfile | None:
    return next((profile for profile in MODEL_PROFILES if profile.id == model_id), None)
