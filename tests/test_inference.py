from __future__ import annotations

from types import SimpleNamespace

from gemma4_mlx_mac import inference
from gemma4_mlx_mac.inference import ChatCompletionRequest, ChatMessage, ChatService
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert messages == [{"role": "user", "content": "hello"}]
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"


def test_chat_service_uses_mlx_stream_generate_shape() -> None:
    def fake_loader(model_path: str):
        assert model_path == "/tmp/gemma"
        return object(), FakeTokenizer()

    def fake_stream_generate(model, tokenizer, prompt, **kwargs):
        assert isinstance(model, object)
        assert isinstance(tokenizer, FakeTokenizer)
        assert prompt == "rendered prompt"
        assert kwargs["max_tokens"] == 3
        assert kwargs["sampler"] == ("sampler", 0.2)
        yield SimpleNamespace(
            text="hello",
            prompt_tokens=4,
            generation_tokens=1,
            finish_reason=None,
        )
        yield SimpleNamespace(
            text=" back",
            prompt_tokens=4,
            generation_tokens=2,
            finish_reason="stop",
        )

    service = ChatService(
        model_loader=fake_loader,
        stream_generator=fake_stream_generate,
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda model_id: "/tmp/gemma" if model_id == DEFAULT_MODEL_ID else None,
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=3,
            temperature=0.2,
        )
    )

    assert response["model"] == DEFAULT_MODEL_ID
    assert response["choices"][0]["message"]["content"] == "hello back"
    assert response["usage"]["prompt_tokens"] == 4
    assert response["usage"]["completion_tokens"] == 2


def test_default_loader_falls_back_for_unused_gemma4_weights(monkeypatch) -> None:
    def strict_loader(model_path: str):
        assert model_path == "/tmp/gemma"
        raise ValueError("Received 1 parameters not in model")

    monkeypatch.setattr(inference, "_load_mlx_lm", lambda: {"load": strict_loader})
    monkeypatch.setattr(
        inference,
        "_load_model_without_strict_weight_matching",
        lambda model_path: ("model", "tokenizer"),
    )

    assert inference._default_model_loader("/tmp/gemma") == ("model", "tokenizer")
