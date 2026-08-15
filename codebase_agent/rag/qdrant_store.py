"""Qdrant 适配层：复用 SearchResult 边界，提供更接近生产的向量存储。"""

from __future__ import annotations

import uuid
from typing import Protocol

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.vector_store import SearchResult

# ── 客户端协议：方便业务代码和测试注入真实或测试用 Qdrant 客户端 ──


class QdrantClientProtocol(Protocol):
    """项目需要的最小 Qdrant client 接口。"""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(
        self,
        collection_name: str,
        vectors_config: object,
    ) -> None: ...

    def upsert(self, collection_name: str, points: list[object]) -> None: ...

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
    ) -> object: ...


# ── 项目异常 ──


class VectorStoreError(RuntimeError):
    """向量库适配层遇到不可恢复错误时抛出。"""


# ── 辅助函数 ──

_QDRANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _stable_point_id(chunk_id: str) -> str:
    """生成稳定 UUID，保证同一个 chunk_id 对应同一个 point id。"""
    return str(uuid.uuid5(_QDRANT_NAMESPACE, chunk_id))


def _build_payload(chunk: Chunk) -> dict[str, object]:
    return {
        "text": chunk.text,
        "chunk_id": chunk.chunk_id,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol_name": chunk.symbol_name,
    }


def _build_point(chunk: Chunk, vector: list[float]) -> object:
    payload = _build_payload(chunk)
    point_id = _stable_point_id(chunk.chunk_id)

    try:
        from qdrant_client.http import models as qmodels

        return qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
    except ImportError:
        return {
            "id": point_id,
            "vector": vector,
            "payload": payload,
        }


def _point_to_search_result(point) -> SearchResult:
    payload: dict[str, object] = point.payload or {}
    _get = lambda key, default="": payload.get(key, default)
    score = float(getattr(point, "score", 0.0))

    return SearchResult(
        text=str(_get("text")),
        score=score,
        metadata={
            "chunk_id": _get("chunk_id"),
            "file_path": _get("file_path"),
            "start_line": payload.get("start_line", 0),
            "end_line": payload.get("end_line", 0),
            "symbol_name": _get("symbol_name"),
        },
        score_source="vector",
        vector_score=score,
    )


# ── Qdrant 适配器 ──


class QdrantVectorStore:
    """把 chunk 向量保存到 Qdrant，接口形状和 InMemoryVectorStore 一致。"""

    def __init__(
        self,
        client: QdrantClientProtocol,
        *,
        collection_name: str = "code_chunks",
        vector_size: int = 1024,
        distance: str = "Cosine",
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
        self._collection_ready = False

    # ── 对外接口：接口形状和 InMemoryVectorStore 一致 ──

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")

        if not chunks:
            return

        self._ensure_collection()

        points: list[object] = []
        for chunk, vector in zip(chunks, vectors):
            if not chunk.text.strip():
                raise ValueError("chunk 文本不能为空")
            if len(vector) != self._vector_size:
                raise ValueError(
                    f"vector 维度不匹配: 期望 {self._vector_size}, 实际 {len(vector)}"
                )

            points.append(_build_point(chunk, vector))

        try:
            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant 写入失败: {exc}") from exc

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        if len(query_vector) != self._vector_size:
            raise ValueError(
                f"query vector 维度不匹配: 期望 {self._vector_size}, 实际 {len(query_vector)}"
            )

        self._ensure_collection()

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant 检索失败: {exc}") from exc

        points = getattr(response, "points", response)
        return [_point_to_search_result(p) for p in points]

    # ── 内部实现 ──

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return

        try:
            if not self._client.collection_exists(self._collection_name):
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=self._make_vectors_config(),
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant collection 初始化失败 '{self._collection_name}': {exc}"
            ) from exc

        self._collection_ready = True

    def _make_vectors_config(self) -> object:
        """构造 vectors_config；如果已安装 qdrant_client，就使用官方类型。"""
        try:
            from qdrant_client.http import models as qmodels

            distance = getattr(qmodels.Distance, self._distance.upper(), self._distance)
            return qmodels.VectorParams(
                size=self._vector_size,
                distance=distance,
            )
        except ImportError:
            return {"size": self._vector_size, "distance": self._distance}
