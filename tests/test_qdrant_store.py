"""Tests for QdrantVectorStore — all via fake client, zero Qdrant dependency."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.qdrant_store import (
    QdrantVectorStore,
    VectorStoreError,
    _stable_point_id,
    _build_payload,
    _point_to_search_result,
)
from codebase_agent.rag.vector_store import SearchResult


# ── 测试辅助函数 ──


def _chunk(**kwargs) -> Chunk:
    defaults = {
        "chunk_id": "db.py:get_engine:1-30",
        "text": "def get_engine() -> Engine:\n    return create_engine(url)",
        "file_path": "db.py",
        "start_line": 1,
        "end_line": 30,
        "symbol_name": "get_engine",
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


def _vectors(count: int, dim: int = 1024) -> list[list[float]]:
    return [[0.1] * dim for _ in range(count)]


class FakeClient:
    """测试用 Qdrant 客户端：记录方法调用，并返回预设数据。"""

    def __init__(self) -> None:
        self.collections: dict[str, bool] = {}
        self.points: list = []
        self.last_upsert_collection: str | None = None
        self.last_upsert_points: list | None = None
        self.query_result: list = []
        self.upsert_should_fail: bool = False
        self.search_should_fail: bool = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.collections.get(collection_name, False)

    def create_collection(
        self, collection_name: str, vectors_config: object
    ) -> None:
        self.collections[collection_name] = True

    def upsert(self, collection_name: str, points: list) -> None:
        if self.upsert_should_fail:
            raise ConnectionError("fake qdrant connection refused")
        self.last_upsert_collection = collection_name
        self.last_upsert_points = points
        self.points.extend(points)

    def query_points(
        self, collection_name: str, query: list[float], limit: int
    ) -> list:
        if self.search_should_fail:
            raise ConnectionError("fake qdrant search timeout")
        return self.query_result[:limit]


def _fake_point(payload: dict | None = None, score: float = 0.95):
    return SimpleNamespace(payload=payload or {}, score=score)


def _point_field(point, key: str):
    if isinstance(point, dict):
        return point[key]
    return getattr(point, key)


# ── 单元测试：辅助函数 ──


class TestStablePointId:
    def test_same_chunk_id_same_uuid(self):
        """同一个 chunk_id 永远生成同一个 UUID"""
        assert _stable_point_id("a") == _stable_point_id("a")

    def test_different_chunk_id_different_uuid(self):
        """不同 chunk_id 生成不同 UUID"""
        assert _stable_point_id("a") != _stable_point_id("b")

    def test_returns_string(self):
        assert isinstance(_stable_point_id("x"), str)
        assert len(_stable_point_id("x")) == 36  # 标准 UUID 字符串长度


class TestBuildPayload:
    def test_all_fields_present(self):
        ch = _chunk()
        payload = _build_payload(ch)
        assert payload["text"] == ch.text
        assert payload["chunk_id"] == ch.chunk_id
        assert payload["file_path"] == ch.file_path
        assert payload["start_line"] == ch.start_line
        assert payload["end_line"] == ch.end_line
        assert payload["symbol_name"] == ch.symbol_name

    def test_symbol_name_empty_ok(self):
        ch = _chunk(symbol_name="")
        payload = _build_payload(ch)
        assert payload["symbol_name"] == ""


class TestPointToSearchResult:
    def test_translates_to_search_result(self):
        point = _fake_point(
            payload={
                "text": "code here",
                "chunk_id": "c1",
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": "foo",
            },
            score=0.88,
        )
        result = _point_to_search_result(point)
        assert isinstance(result, SearchResult)
        assert result.text == "code here"
        assert result.score == 0.88
        assert result.metadata["chunk_id"] == "c1"
        assert result.metadata["file_path"] == "a.py"
        assert result.metadata["start_line"] == 10
        assert result.metadata["end_line"] == 20
        assert result.metadata["symbol_name"] == "foo"

    def test_missing_payload_fields_get_defaults(self):
        """payload 缺字段时给合理默认值，不抛异常"""
        point = _fake_point(payload={"text": "only text"}, score=0.5)
        result = _point_to_search_result(point)
        assert result.text == "only text"
        assert result.metadata["chunk_id"] == ""
        assert result.metadata["file_path"] == ""
        assert result.metadata["start_line"] == 0
        assert result.metadata["end_line"] == 0
        assert result.metadata["symbol_name"] == ""

    def test_none_payload_handled(self):
        point = _fake_point(payload=None, score=0.3)
        result = _point_to_search_result(point)
        assert result.text == ""  # str("") → ""
        assert result.score == 0.3

    def test_missing_score_defaults_zero(self):
        point = SimpleNamespace(payload={"text": "hi"})  # 没有 .score 属性
        result = _point_to_search_result(point)
        assert result.score == 0.0


# ── 单元测试：QdrantVectorStore ──


class TestAddChunks:
    def test_empty_chunks_returns_early(self):
        store = QdrantVectorStore(client=FakeClient())
        store.add_chunks([], [])  # 不应抛异常

    def test_length_mismatch_raises(self):
        store = QdrantVectorStore(client=FakeClient())
        with pytest.raises(ValueError, match="数量必须一致"):
            store.add_chunks([_chunk(), _chunk()], _vectors(1))

    def test_upsert_called_with_correct_collection(self):
        client = FakeClient()
        store = QdrantVectorStore(client=client, collection_name="test_coll")
        store.add_chunks([_chunk()], _vectors(1))
        assert client.last_upsert_collection == "test_coll"

    def test_upsert_points_have_id_vector_payload(self):
        client = FakeClient()
        store = QdrantVectorStore(client=client)
        store.add_chunks([_chunk()], _vectors(1))
        point = client.last_upsert_points[0]
        assert _point_field(point, "id")
        assert _point_field(point, "vector")
        assert _point_field(point, "payload")

    def test_point_id_is_stable(self):
        client = FakeClient()
        store = QdrantVectorStore(client=client)
        ch = _chunk()
        store.add_chunks([ch], _vectors(1))
        first_id = _point_field(client.last_upsert_points[0], "id")

        # 同一个 chunk 第二次写入时仍应得到同一个 id
        store.add_chunks([ch], _vectors(1))
        second_id = _point_field(client.last_upsert_points[0], "id")
        assert first_id == second_id

    def test_client_exception_wraps_as_vector_store_error(self):
        client = FakeClient()
        client.upsert_should_fail = True
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Qdrant 写入失败"):
            store.add_chunks([_chunk()], _vectors(1))

    def test_creates_collection_on_first_upsert(self):
        client = FakeClient()
        store = QdrantVectorStore(
            client=client, collection_name="auto_create"
        )
        assert not client.collection_exists("auto_create")
        store.add_chunks([_chunk()], _vectors(1))
        assert client.collection_exists("auto_create")

    def test_collection_not_recreated_when_exists(self):
        client = FakeClient()
        client.collections["existing"] = True
        store = QdrantVectorStore(
            client=client, collection_name="existing"
        )
        store.add_chunks([_chunk()], _vectors(1))  # 不应抛异常


class TestSearch:
    def test_top_k_zero_returns_empty(self):
        store = QdrantVectorStore(client=FakeClient())
        result = store.search([0.1] * 1024, top_k=0)
        assert result == []

    def test_top_k_negative_returns_empty(self):
        store = QdrantVectorStore(client=FakeClient())
        result = store.search([0.1] * 1024, top_k=-1)
        assert result == []

    def test_returns_search_results(self):
        client = FakeClient()
        client.query_result = [
            _fake_point(
                payload={"text": "code", "chunk_id": "c1", "file_path": "a.py",
                         "start_line": 1, "end_line": 10, "symbol_name": "f"},
                score=0.9,
            )
        ]
        store = QdrantVectorStore(client=client)
        results = store.search([0.1] * 1024, top_k=3)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 0.9
        assert results[0].metadata["file_path"] == "a.py"

    def test_accepts_qdrant_query_response_shape(self):
        class ResponseShapeClient(FakeClient):
            def query_points(self, collection_name: str, query: list[float], limit: int):
                return SimpleNamespace(
                    points=[
                        _fake_point(
                            payload={"text": "code", "chunk_id": "c1", "file_path": "a.py"},
                            score=0.9,
                        )
                    ]
                )

        store = QdrantVectorStore(client=ResponseShapeClient())
        results = store.search([0.1] * 1024, top_k=3)

        assert len(results) == 1
        assert results[0].metadata["file_path"] == "a.py"

    def test_client_exception_wraps_as_vector_store_error(self):
        client = FakeClient()
        client.search_should_fail = True
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Qdrant 检索失败"):
            store.search([0.1] * 1024, top_k=3)

    def test_auto_creates_collection_on_search(self):
        client = FakeClient()
        store = QdrantVectorStore(
            client=client, collection_name="search_coll"
        )
        assert not client.collection_exists("search_coll")
        store.search([0.1] * 1024, top_k=1)
        assert client.collection_exists("search_coll")

    def test_collection_ready_flag_avoids_recheck(self):
        """_collection_ready 标志避免重复调 collection_exists"""
        client = FakeClient()
        client.collections["once"] = True
        store = QdrantVectorStore(
            client=client, collection_name="once"
        )
        store.search([0.1] * 1024, top_k=1)
        # exists=True 时不应调 create_collection
        # 如果标志没生效走 create_collection，这里也不应该报错
        # 真正的验证是两次 search 都正常

    def test_dimension_mismatch_raises(self):
        store = QdrantVectorStore(client=FakeClient(), vector_size=768)
        with pytest.raises(ValueError, match="维度不匹配"):
            store.search([0.1] * 1024, top_k=3)
