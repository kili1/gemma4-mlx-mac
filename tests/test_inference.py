from __future__ import annotations

from types import SimpleNamespace

from gemma4_mlx_mac import inference
from gemma4_mlx_mac.inference import (
    ChatCompletionRequest,
    ChatMessage,
    ChatService,
    InferenceNotReady,
)
from gemma4_mlx_mac.models import DEFAULT_MODEL_ID


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages = []
        self.enable_thinking: bool | None = None

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        self.messages = messages
        self.enable_thinking = enable_thinking
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"


class LegacyTokenizer:
    def __init__(self) -> None:
        self.messages = []

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "legacy rendered prompt"


def test_chat_service_uses_mlx_stream_generate_shape() -> None:
    tokenizer = FakeTokenizer()

    def fake_loader(model_path: str):
        assert model_path == "/tmp/gemma"
        return object(), tokenizer

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
    assert tokenizer.enable_thinking is False
    assert tokenizer.messages == [{"role": "user", "content": "hello"}]


def test_chat_service_enables_native_gemma4_thinking_and_formats_channels() -> None:
    tokenizer = FakeTokenizer()

    def fake_stream_generate(*_args, **_kwargs):
        yield SimpleNamespace(text="<|chan", prompt_tokens=3, generation_tokens=1)
        yield SimpleNamespace(text="nel>thought\nCheck math.", prompt_tokens=3, generation_tokens=2)
        yield SimpleNamespace(text="<chan", prompt_tokens=3, generation_tokens=3)
        yield SimpleNamespace(text="nel|>42", prompt_tokens=3, generation_tokens=4)

    service = ChatService(
        model_loader=lambda _model_path: (object(), tokenizer),
        stream_generator=fake_stream_generate,
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="What is 6*7?")],
            show_thinking=True,
        )
    )

    assert tokenizer.enable_thinking is True
    assert response["choices"][0]["message"]["content"] == "Thinking:\nCheck math.\n\nAnswer:\n42"


def test_chat_service_strips_gemma4_thought_channels_when_thinking_off() -> None:
    tokenizer = FakeTokenizer()

    def fake_stream_generate(*_args, **_kwargs):
        yield SimpleNamespace(text="<|channel>thought\n", prompt_tokens=2, generation_tokens=1)
        yield SimpleNamespace(text="private step<channel|>Direct answer", generation_tokens=2)

    service = ChatService(
        model_loader=lambda _model_path: (object(), tokenizer),
        stream_generator=fake_stream_generate,
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            show_thinking=False,
        )
    )

    assert tokenizer.enable_thinking is False
    assert response["choices"][0]["message"]["content"] == "Direct answer"


def test_chat_service_strips_prior_thought_channels_from_multiturn_history() -> None:
    tokenizer = FakeTokenizer()
    service = ChatService(
        model_loader=lambda _model_path: (object(), tokenizer),
        stream_generator=lambda *_args, **_kwargs: [SimpleNamespace(text="Sure!")],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[
                ChatMessage(role="user", content="First"),
                ChatMessage(
                    role="assistant",
                    content="<|channel>thought\nsecret<channel|>Clean reply",
                ),
                ChatMessage(role="user", content="Follow up"),
            ],
        )
    )

    assert tokenizer.messages[1] == {"role": "assistant", "content": "Clean reply"}


def test_chat_service_supports_legacy_tokenizer_without_enable_thinking() -> None:
    tokenizer = LegacyTokenizer()
    service = ChatService(
        model_loader=lambda _model_path: (object(), tokenizer),
        stream_generator=lambda *_args, **_kwargs: [SimpleNamespace(text="plain answer")],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            show_thinking=True,
        )
    )

    assert response["choices"][0]["message"]["content"] == "plain answer"


def test_chat_service_strips_private_thinking_tags() -> None:
    def fake_stream_generate(*_args, **_kwargs):
        yield SimpleNamespace(text="<thi", prompt_tokens=2, generation_tokens=1)
        yield SimpleNamespace(text="nk>private reasoning", prompt_tokens=2, generation_tokens=2)
        yield SimpleNamespace(text="</think>visible answer", prompt_tokens=2, generation_tokens=3)

    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        stream_generator=fake_stream_generate,
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    chunks = list(
        service.stream_tokens(
            ChatCompletionRequest(
                model=DEFAULT_MODEL_ID,
                messages=[ChatMessage(role="user", content="hello")],
            )
        )
    )

    assert "".join(chunk.text for chunk in chunks) == "visible answer"


def test_chat_service_loads_active_and_explicit_adapters(tmp_path) -> None:
    adapter_dir = tmp_path / "my-lora"
    adapter_dir.mkdir()
    loaded_calls: list[tuple[str, str | None]] = []

    def fake_loader(model_path: str, adapter_path: str | None = None):
        loaded_calls.append((model_path, adapter_path))
        return object(), FakeTokenizer()

    active_adapter: list[str | None] = [None]
    service = ChatService(
        model_loader=fake_loader,
        stream_generator=lambda *_args, **_kwargs: [SimpleNamespace(text="ok")],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
        active_adapter_resolver=lambda: active_adapter[0],
    )

    # First call without adapter
    service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hi")],
        )
    )
    # Activate adapter in registry — triggers reload with adapter_path
    active_adapter[0] = str(adapter_dir)
    service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hi")],
        )
    )

    assert loaded_calls == [("/tmp/gemma", None), ("/tmp/gemma", str(adapter_dir))]
    memory = service.memory_status()
    assert any(item.adapter_path == str(adapter_dir) for item in memory.loaded_models)


def test_chat_service_streams_token_metrics() -> None:
    def fake_stream_generate(*_args, **_kwargs):
        yield SimpleNamespace(text="hi", prompt_tokens=2, generation_tokens=1)
        yield SimpleNamespace(text=" there", prompt_tokens=2, generation_tokens=2)

    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        stream_generator=fake_stream_generate,
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    chunks = list(
        service.stream_tokens(
            ChatCompletionRequest(
                model=DEFAULT_MODEL_ID,
                messages=[ChatMessage(role="user", content="hello")],
                max_tokens=2,
            )
        )
    )

    assert [chunk.text for chunk in chunks] == ["hi", " there"]
    assert chunks[-1].completion_tokens == 2
    assert chunks[-1].tokens_per_second > 0


def test_chat_service_reports_loaded_model_memory(monkeypatch) -> None:
    memory_readings = [1000, 7000]
    monkeypatch.setattr(inference, "_process_resident_memory_bytes", lambda: memory_readings[0])
    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        stream_generator=lambda *_args, **_kwargs: [SimpleNamespace(text="ok")],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    unloaded = service.memory_status()
    assert unloaded.loaded is False
    assert unloaded.model_memory_bytes == 0

    service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
        )
    )
    memory_readings[0] = 7000

    loaded = service.memory_status()
    assert loaded.loaded is True
    assert loaded.loaded_count == 1
    assert loaded.loaded_models[0].path == "/tmp/gemma"
    assert loaded.model_memory_bytes == 6000


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


def test_chat_service_accepts_existing_local_model_path(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def fake_loader(model_path: str):
        assert model_path == str(model_dir)
        return object(), FakeTokenizer()

    service = ChatService(
        model_loader=fake_loader,
        stream_generator=lambda *_args, **_kwargs: [SimpleNamespace(text="ok")],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: None,
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=str(model_dir),
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=1,
        )
    )

    assert response["model"] == str(model_dir)
    assert response["choices"][0]["message"]["content"] == "ok"


def test_chat_service_reports_missing_local_model_path() -> None:
    service = ChatService(model_path_resolver=lambda _model_id: None)

    try:
        service.create_completion(
            ChatCompletionRequest(
                model="/missing/model/path",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )
    except InferenceNotReady as exc:
        assert "Model path does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing local model path to fail before MLX loading.")


def test_chat_service_handles_char_by_char_ghost_channel_and_false_prefixes() -> None:
    # Ghost thought channel (<|channel>thought + separate "\n" + <channel|>) must not emit Thinking:
    raw_ghost = (
        "<|channel>thought\n<channel|>"
        "Clean 12B output with <|chan_literal and <thi_literal."
    )

    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        stream_generator=lambda *_a, **_kw: [SimpleNamespace(text=ch) for ch in raw_ghost],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            show_thinking=True,
        )
    )

    assert response["choices"][0]["message"]["content"] == (
        "Clean 12B output with <|chan_literal and <thi_literal."
    )


def test_chat_service_handles_char_by_char_thinking_stream() -> None:
    raw = "<|channel>thought\nLine 1\nLine 2<channel|>Final 🚀"

    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        stream_generator=lambda *_a, **_kw: [SimpleNamespace(text=ch) for ch in raw],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    response = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            show_thinking=True,
        )
    )

    assert response["choices"][0]["message"]["content"] == (
        "Thinking:\nLine 1\nLine 2\n\nAnswer:\nFinal 🚀"
    )


def test_chat_service_handles_unterminated_thought_hitting_max_tokens() -> None:
    raw = "<|channel>thought\nStill reasoning when limit hit"

    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        stream_generator=lambda *_a, **_kw: [
            SimpleNamespace(text=raw[:17]),
            SimpleNamespace(text=raw[17:]),
        ],
        sampler_factory=lambda temp: ("sampler", temp),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    shown = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            show_thinking=True,
        )
    )
    hidden = service.create_completion(
        ChatCompletionRequest(
            model=DEFAULT_MODEL_ID,
            messages=[ChatMessage(role="user", content="hello")],
            show_thinking=False,
        )
    )

    assert shown["choices"][0]["message"]["content"] == "Thinking:\nStill reasoning when limit hit"
    assert hidden["choices"][0]["message"]["content"] == ""


def test_chat_service_rejects_missing_explicit_adapter_path() -> None:
    service = ChatService(
        model_loader=lambda _model_path: (object(), FakeTokenizer()),
        model_path_resolver=lambda _model_id: "/tmp/gemma",
    )

    try:
        service.create_completion(
            ChatCompletionRequest(
                model=DEFAULT_MODEL_ID,
                messages=[ChatMessage(role="user", content="hello")],
                adapter_path="/does/not/exist/adapter",
            )
        )
    except InferenceNotReady as exc:
        assert "Adapter path does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing explicit adapter_path to raise InferenceNotReady.")
