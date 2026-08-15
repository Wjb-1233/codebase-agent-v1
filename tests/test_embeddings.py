from types import SimpleNamespace

import pytest

from codebase_agent.exceptions import ConfigError, EmbeddingError
from codebase_agent.rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class FakeEmbeddingsResource:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def make_response(items):
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=embedding) for index, embedding in items]
    )


def make_client(response=None, error=None):
    return SimpleNamespace(embeddings=FakeEmbeddingsResource(response=response, error=error))


def test_embed_texts_calls_openai_embeddings_resource_and_preserves_order():
    response = make_response(
        [
            (1, [0.0, 1.0, 0.0]),
            (0, [1.0, 0.0, 0.0]),
        ]
    )
    client = make_client(response=response)
    provider = OpenAIEmbeddingProvider(client=client, model="test-embedding-model")

    vectors = provider.embed_texts(["first chunk", "second chunk"])

    assert client.embeddings.calls == [
        {"model": "test-embedding-model", "input": ["first chunk", "second chunk"]}
    ]
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_embed_texts_returns_empty_list_without_api_call():
    client = make_client(response=make_response([]))
    provider = OpenAIEmbeddingProvider(client=client)

    assert provider.embed_texts([]) == []
    assert client.embeddings.calls == []


def test_provider_uses_default_model_and_client_factory(monkeypatch):
    created_clients = []

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.embeddings = FakeEmbeddingsResource(response=make_response([(0, [0.5])]))
            created_clients.append(self)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr("codebase_agent.rag.embeddings.load_dotenv", lambda **kwargs: None)

    provider = OpenAIEmbeddingProvider(client_factory=FakeClient)

    assert provider.model == DEFAULT_EMBEDDING_MODEL
    assert created_clients[0].api_key == "test-key"
    assert provider.embed_texts(["hello"]) == [[0.5]]


def test_provider_reads_embedding_model_from_environment(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.embeddings = FakeEmbeddingsResource(response=make_response([(0, [0.7])]))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "custom-embedding-model")
    monkeypatch.setattr("codebase_agent.rag.embeddings.load_dotenv", lambda **kwargs: None)

    provider = OpenAIEmbeddingProvider(client_factory=FakeClient)

    assert provider.model == "custom-embedding-model"


def test_provider_prefers_embedding_specific_config(monkeypatch):
    created_clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            created_clients.append(kwargs)
            self.embeddings = FakeEmbeddingsResource(response=make_response([(0, [0.5])]))

    monkeypatch.setenv("OPENAI_API_KEY", "chat-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.example.test")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setattr("codebase_agent.rag.embeddings.load_dotenv", lambda **kwargs: None)

    provider = OpenAIEmbeddingProvider(client_factory=FakeClient)

    assert provider.model == "embedding-model"
    assert created_clients == [
        {"api_key": "embedding-key", "base_url": "https://embedding.example.test/v1"}
    ]


def test_missing_openai_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr("codebase_agent.rag.embeddings.load_dotenv", lambda **kwargs: None)

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()


def test_invalid_text_inputs_raise_value_error():
    provider = OpenAIEmbeddingProvider(client=make_client())

    with pytest.raises(ValueError, match="字符串序列"):
        provider.embed_texts("single string")

    with pytest.raises(ValueError, match=r"texts\[0\] 不能为空"):
        provider.embed_texts(["   "])

    with pytest.raises(ValueError, match=r"texts\[1\] 必须是字符串"):
        provider.embed_texts(["ok", 123])


def test_openai_embedding_errors_are_wrapped():
    client = make_client(error=RuntimeError("upstream failed"))
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingError, match="OpenAI Embeddings 请求失败") as exc_info:
        provider.embed_texts(["chunk"])

    assert exc_info.value.code == 502


def test_local_embedding_provider_uses_sentence_transformer_factory():
    created_models = []

    class FakeLocalModel:
        def encode(self, texts, normalize_embeddings):
            assert texts == ["chunk one", "chunk two"]
            assert normalize_embeddings is True
            return [[1, 0], [0.5, 0.5]]

    def fake_factory(model_name):
        created_models.append(model_name)
        return FakeLocalModel()

    provider = SentenceTransformerEmbeddingProvider(model_factory=fake_factory)

    assert provider.model_name == DEFAULT_LOCAL_EMBEDDING_MODEL
    assert created_models == [DEFAULT_LOCAL_EMBEDDING_MODEL]
    assert provider.embed_texts(["chunk one", "chunk two"]) == [[1.0, 0.0], [0.5, 0.5]]


def test_local_embedding_provider_reads_model_from_environment(monkeypatch):
    class FakeLocalModel:
        def encode(self, texts, normalize_embeddings):
            return [[0.1]]

    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "custom-local-model")
    monkeypatch.setattr("codebase_agent.rag.embeddings.load_dotenv", lambda **kwargs: None)

    provider = SentenceTransformerEmbeddingProvider(
        model_factory=lambda model_name: FakeLocalModel()
    )

    assert provider.model_name == "custom-local-model"
