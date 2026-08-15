"""向量检索 + 关键词检索的混合检索辅助函数。"""

from __future__ import annotations

from codebase_agent.rag.vector_store import SearchResult


def rrf_fuse(
    vector_results: list[SearchResult],
    keyword_results: list[SearchResult],
    *,
    k: int = 60,
) -> list[SearchResult]:
    """使用 RRF（倒数排名融合）合并多路排序结果。"""
    rrf_scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    metadatas: dict[str, dict[str, object]] = {}
    vec_scores: dict[str, float | None] = {}
    kw_scores: dict[str, float | None] = {}
    sources: dict[str, set[str]] = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = _chunk_key(result)
        rrf_scores[chunk_id] = 1.0 / (k + rank)
        sources.setdefault(chunk_id, set()).add("vector")
        texts.setdefault(chunk_id, result.text)
        metadatas.setdefault(chunk_id, dict(result.metadata))
        vec_scores[chunk_id] = result.vector_score

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = _chunk_key(result)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(chunk_id, set()).add("keyword")
        texts.setdefault(chunk_id, result.text)
        metadatas.setdefault(chunk_id, dict(result.metadata))
        kw_scores[chunk_id] = result.keyword_score

    merged: list[SearchResult] = []
    for chunk_id, score in rrf_scores.items():
        source_set = sources[chunk_id]
        if len(source_set) == 2:
            score_source = "both"
        elif "vector" in source_set:
            score_source = "vector"
        else:
            score_source = "keyword"

        metadata = dict(metadatas[chunk_id])
        metadata["score_source"] = score_source
        if vec_scores.get(chunk_id) is not None:
            metadata["vector_score"] = vec_scores[chunk_id]
        if kw_scores.get(chunk_id) is not None:
            metadata["keyword_score"] = kw_scores[chunk_id]

        merged.append(
            SearchResult(
                text=texts[chunk_id],
                score=score,
                metadata=metadata,
                score_source=score_source,
                vector_score=vec_scores.get(chunk_id),
                keyword_score=kw_scores.get(chunk_id),
            )
        )

    merged.sort(key=lambda result: result.score, reverse=True)
    return merged


def _chunk_key(result: SearchResult) -> str:
    """为 SearchResult 生成稳定的去重键。"""
    return str(result.metadata.get("chunk_id", result.text))
