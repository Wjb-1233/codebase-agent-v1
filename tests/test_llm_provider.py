"""Tests for LLM providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codebase_agent.exceptions import ConfigError, LLMError
from codebase_agent.rag.llm import FakeLLMProvider, OpenAILLMProvider


def test_fake_llm_returns_fixed_response():
    provider = FakeLLMProvider("固定回答")
    result = provider.generate("任意 prompt")
    assert result == "固定回答"


def test_fake_llm_default_response():
    provider = FakeLLMProvider()
    result = provider.generate("test")
    assert len(result) > 0


def test_fake_llm_records_last_prompt():
    provider = FakeLLMProvider()
    provider.generate("hello world")
    assert provider.last_prompt == "hello world"


def test_fake_llm_different_calls_update_last_prompt():
    provider = FakeLLMProvider()
    provider.generate("first")
    assert provider.last_prompt == "first"
    provider.generate("second")
    assert provider.last_prompt == "second"


def test_openai_provider_uses_injected_client():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  grounded answer  "))]
    )
    provider = OpenAILLMProvider(client=client, model="test-model")

    result = provider.generate("answer from context")

    assert result == "grounded answer"
    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "answer from context"}],
        temperature=0.2,
        max_tokens=1024,
    )


def test_openai_provider_rejects_empty_prompt():
    provider = OpenAILLMProvider(client=MagicMock())

    with pytest.raises(ValueError, match="prompt"):
        provider.generate("   ")


def test_openai_provider_rejects_empty_response():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )
    provider = OpenAILLMProvider(client=client)

    with pytest.raises(LLMError, match="空响应"):
        provider.generate("question")


def test_openai_provider_rejects_invalid_response_shape():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    provider = OpenAILLMProvider(client=client)

    with pytest.raises(LLMError, match="结构无效"):
        provider.generate("question")


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr("codebase_agent.rag.llm.load_dotenv", lambda **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAILLMProvider()


def test_openai_provider_passes_base_url(monkeypatch):
    created_clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)
            self.chat = MagicMock()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr("codebase_agent.rag.llm.load_dotenv", lambda **kwargs: None)

    OpenAILLMProvider(client_factory=FakeClient)

    assert created_clients == [
        {"api_key": "test-key", "base_url": "https://example.test/v1"}
    ]
