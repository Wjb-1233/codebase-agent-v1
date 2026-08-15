"""代码 chunk 的内存向量检索实现。"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Protocol

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.embeddings import EmbeddingProvider

if TYPE_CHECKING:
    from codebase_agent.rag.keyword_search import KeywordIndex
    from codebase_agent.rag.reranker import Reranker


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    metadata: dict[str, object]
    score_source: str = ""          # "vector" | "keyword" | "both"
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    reranker: str = ""


class VectorStoreProtocol(Protocol):
    """本地向量库和外部向量库共用的接口边界。"""

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        ...


class InMemoryVectorStore:
    """把 chunk 向量保存在内存中，并返回最相似的代码片段。"""

    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")

        for chunk, vector in zip(chunks, vectors):
            if not chunk.text.strip():
                raise ValueError("chunk 文本不能为空")

            self.items.append(
                {
                    "text": chunk.text,
                    "vector": _ensure_vector(vector),
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol_name": chunk.symbol_name,
                        "score_source": "vector",
                    },
                }
            )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []

        results: list[SearchResult] = []
        for item in self.items:
            score = cosine_similarity(query_vector, item["vector"])
            results.append(
                SearchResult(
                    text=str(item["text"]),
                    score=score,
                    metadata=dict(item["metadata"]),
                    score_source="vector",
                    vector_score=score,
                )
            )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self.items)


VectorStore = InMemoryVectorStore


def search_code(
    query: str,
    chunks: list[Chunk],
    top_k: int,
    embedding_provider: EmbeddingProvider,
    *,
    keyword_index: "KeywordIndex | None" = None,
    reranker: "Reranker | None" = None,
    vector_store: VectorStoreProtocol | None = None,
) -> list[SearchResult]:
    """使用向量检索、可选混合检索和可选重排来搜索代码 chunk。"""
    if not query.strip():
        raise ValueError("query 不能为空")
    if not chunks or top_k <= 0:
        return []

    chunk_texts = [chunk.text for chunk in chunks]
    chunk_vectors = embedding_provider.embed_texts(chunk_texts)
    query_vector = embedding_provider.embed_texts([query])[0]

    store = vector_store or InMemoryVectorStore()
    store.add_chunks(chunks, chunk_vectors)
    vector_results = store.search(query_vector=query_vector, top_k=top_k)

    if keyword_index is None:
        if reranker is not None:
            return reranker.rerank(query, vector_results, top_k)
        return vector_results

    from codebase_agent.rag.hybrid_search import rrf_fuse

    keyword_results = keyword_index.search(query, top_k=top_k)
    merged = rrf_fuse(vector_results, keyword_results)

    if reranker is not None:
        return reranker.rerank(query, merged, top_k)

    return merged[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = _ensure_vector(left)
    right_vector = _ensure_vector(right)

    if len(left_vector) != len(right_vector):
        raise ValueError("两个向量的维度必须一致")

    left_norm = sqrt(sum(value * value for value in left_vector))
    right_norm = sqrt(sum(value * value for value in right_vector))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left_vector, right_vector))
    return dot / (left_norm * right_norm)


def _ensure_vector(vector: list[float]) -> list[float]:
    if not vector:
        raise ValueError("vector 不能为空")

    clean_vector = []
    for value in vector:
        if not isinstance(value, (int, float)):
            raise ValueError("vector 只能包含数字")
        clean_vector.append(float(value))
    return clean_vector
