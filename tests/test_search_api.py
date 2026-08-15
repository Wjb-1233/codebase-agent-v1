"""POST /search 接口测试。"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import codebase_agent.backend.main as main_module
from codebase_agent.backend.main import app, get_embedding_provider, get_reranker
from codebase_agent.rag.reranker import IdentityReranker
from codebase_agent.exceptions import EmbeddingError


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    main_module._reranker = None
    yield
    app.dependency_overrides.clear()
    main_module._reranker = None


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """根据代码和搜索关键词返回确定性向量。"""
    vectors: list[list[float]] = []
    for text in texts:
        lower_text = text.lower()
        if "connect" in lower_text or "database" in lower_text or "db" in lower_text:
            vectors.append([1.0, 0.0, 0.0])
        elif "render" in lower_text or "html" in lower_text or "ui" in lower_text:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


def _make_fake_provider():
    provider = MagicMock()
    provider.embed_texts = _fake_embed
    return provider


def _two_files():
    return [
        {
            "file_path": "db.py",
            "content": "def connect_db():\n    return 'connected'\n",
        },
        {
            "file_path": "ui.py",
            "content": "def render_html():\n    return '<div/>'\n",
        },
    ]


def test_search_returns_relevant_chunk_first():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={"query": "connect to database", "top_k": 3, "files": _two_files()},
    )

    assert resp.status_code == 200
    results = resp.json()["results"]

    assert len(results) >= 1
    assert results[0]["file_path"] == "db.py"
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]


def test_search_response_contains_all_metadata_fields():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={
            "query": "render",
            "top_k": 1,
            "files": [{"file_path": "ui.py", "content": "def render_html():\n    pass\n"}],
        },
    )

    assert resp.status_code == 200
    item = resp.json()["results"][0]

    assert isinstance(item["text"], str) and len(item["text"]) > 0
    assert isinstance(item["score"], float)
    assert pytest.approx(item["score"], abs=1e-6) == item["score"]
    assert 0.0 <= item["score"] <= 1.01
    assert isinstance(item["chunk_id"], str) and len(item["chunk_id"]) > 0
    assert isinstance(item["file_path"], str)
    assert isinstance(item["start_line"], int)
    assert isinstance(item["end_line"], int)
    assert "symbol_name" in item
    assert item["parent_id"] == "ui.py:render_html"
    assert item["parent_start_line"] == 1
    assert item["parent_end_line"] >= 1
    assert item["score_source"] in {"vector", "keyword", "both"}
    assert item["reranker"] == "identity"


def test_search_top_k_respected():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={"query": "test", "top_k": 1, "files": _two_files()},
    )

    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 1


def test_search_empty_query_returns_422():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={"query": "", "files": _two_files()},
    )

    assert resp.status_code == 422


def test_search_top_k_zero_rejected():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={"query": "test", "top_k": 0, "files": _two_files()},
    )

    assert resp.status_code == 422


def test_search_empty_files_returns_400():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={"query": "test", "files": []},
    )

    assert resp.status_code == 400


def test_search_no_valid_chunks_returns_empty_results():
    provider = _make_fake_provider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={"query": "test", "files": [{"file_path": "empty.py", "content": ""}]},
    )

    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_embedding_provider_error_returns_502():
    provider = MagicMock()
    provider.embed_texts.side_effect = EmbeddingError("simulated failure")
    app.dependency_overrides[get_embedding_provider] = lambda: provider

    resp = client.post(
        "/search",
        json={
            "query": "test",
            "files": [{"file_path": "a.py", "content": "def foo():\n    pass\n"}],
        },
    )

    assert resp.status_code == 502

def test_default_reranker_is_identity(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    main_module._reranker = None

    assert isinstance(get_reranker(), IdentityReranker)


def test_build_vector_store_for_files_uses_qdrant(monkeypatch):
    class FakeQdrantClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.collections = {}

        def collection_exists(self, collection_name: str) -> bool:
            return self.collections.get(collection_name, False)

        def create_collection(self, collection_name: str, vectors_config: object) -> None:
            self.collections[collection_name] = True

        def upsert(self, collection_name: str, points: list) -> None:
            self.points = points

        def query_points(self, collection_name: str, query: list[float], limit: int) -> list:
            return []

    monkeypatch.setenv("VECTOR_STORE_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_code_chunks")
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "3")
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))

    store, backend = main_module.build_vector_store_for_files(
        [main_module.SearchFileInput(file_path="a.py", content="def a():\n    pass\n")]
    )

    assert backend == "qdrant"
    assert store is not None
    assert store._collection_name.startswith("test_code_chunks_")
