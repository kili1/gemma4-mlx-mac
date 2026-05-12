from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .models import DEFAULT_MODEL_ID


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    stream: bool = False


class InferenceNotReady(RuntimeError):
    pass


class ChatService:
    def create_completion(self, request: ChatCompletionRequest) -> dict:
        raise InferenceNotReady(
            "MLX inference is not wired yet. Install the MLX extra and implement the "
            "ChatService backend with mlx_lm.stream_generate."
        )
