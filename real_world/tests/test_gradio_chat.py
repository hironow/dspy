from typing import Any

import pytest

from real_world.gradio_chat import (
    ChatBackend,
    _build_messages,
    _format_history,
    _require_env,
    _sanitize_cell,
)


class _FakeLM:
    def __init__(self, response: str = "OK"):
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def __call__(self, messages, **kwargs):  # noqa: ANN001 - signature matches dspy.LM
        self.calls.append((messages, kwargs))
        return [{"text": self.response}]


def test_require_env_missing(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SOME_KEY"):
        _require_env("SOME_KEY")


def test_build_messages_with_system_prompt():
    history = [("hello", "hi"), ("how are you?", None)]
    result = _build_messages(history, "next message", "system prompt")
    assert result[0] == {"role": "system", "content": "system prompt"}
    assert result[-1] == {"role": "user", "content": "next message"}
    # ensure assistant turn missing response is skipped
    assert {"role": "assistant", "content": "how are you?"} not in result


def test_format_history_compacts_conversation():
    history = [("hi", "hello"), ("what's up?", None)]
    text = _format_history(history)
    assert "User: hi" in text
    assert "Assistant: hello" in text
    assert text.count("Assistant:") == 1


def test_sanitize_cell_replaces_control_chars():
    raw = "line1\nline2\tcol\r"
    sanitized = _sanitize_cell(raw)
    assert "\n" not in sanitized
    assert "\t" not in sanitized
    assert "\r" not in sanitized
    assert "\\n" in sanitized


def test_chatbackend_dummy_returns_echo():
    backend = ChatBackend()
    history: list[tuple[str, str | None]] = [("earlier question", "answer")]
    messages = _build_messages(history, "Please echo me", None)
    reply = backend.generate(
        "dummy (offline echo)",
        messages,
        temperature=0.1,
        max_tokens=64,
    )
    assert "Please echo me" in reply


def test_chatbackend_custom_backend():
    backend = ChatBackend()
    fake = _FakeLM("generated text")
    backend._builders["custom-backend"] = lambda: fake  # type: ignore[attr-defined]
    backend._cache["custom-backend"] = fake  # type: ignore[attr-defined]

    messages = [{"role": "user", "content": "hello"}]
    reply = backend.generate("custom-backend", messages, temperature=0.5, max_tokens=32)

    assert reply == "generated text"
    assert fake.calls, "Custom LM should have been invoked"
    _messages, kwargs = fake.calls[0]
    assert kwargs["temperature"] == pytest.approx(0.5)
    assert kwargs["max_tokens"] == 32
