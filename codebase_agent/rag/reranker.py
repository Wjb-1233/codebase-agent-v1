"""RAG 候选结果重排。"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from codebase_agent.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """按 query 对候选结果重新排序，并返回前 top_k 条。"""

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        ...


class IdentityReranker:
    """稳定的兜底重排器。

    它不改变上游检索顺序，只给结果补充重排元数据，保证 API 响应能说明排序阶段。
    """

    name = "identity"

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        return [_copy_with_rerank_metadata(item, None, self.name) for item in candidates[:top_k]]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder 重排器。

    模型按需加载。加载或预测失败时使用 fallback，避免本地开发和 CI 环境中断主链路。
    """

    _DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    name = "cross_encoder"

    def __init__(
        self,
        model_name: str | None = None,
        fallback: Reranker | None = None,
    ) -> None:
        self._model_name = model_name or os.getenv("RERANKER_MODEL", self._DEFAULT_MODEL)
        self._model: object | None = None
        self._fallback = fallback or IdentityReranker()

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not candidates or top_k <= 0:
            return []

        try:
            model = self._ensure_model()
            pairs: list[list[str]] = [[query, candidate.text] for candidate in candidates]
            scores: list[float] = [float(score) for score in model.predict(pairs)]
        except Exception as exc:  # pragma: no cover - log path is verified by fallback behavior.
            logger.warning(
                "CrossEncoder 重排器不可用，回退到 IdentityReranker: %s",
                exc,
                exc_info=True,
            )
            return self._fallback.rerank(query, candidates, top_k)

        scored: list[tuple[float, SearchResult]] = list(zip(scores, candidates))
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            _copy_with_rerank_metadata(result, new_score, self.name)
            for new_score, result in scored[:top_k]
        ]

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        logger.info("正在加载 CrossEncoder 重排模型: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("CrossEncoder 重排模型已加载: %s", self._model_name)
        return self._model


def _copy_with_rerank_metadata(
    result: SearchResult,
    rerank_score: float | None,
    reranker_name: str,
) -> SearchResult:
    metadata = dict(result.metadata)
    if result.score_source:
        metadata["score_source"] = result.score_source
    if result.vector_score is not None:
        metadata["vector_score"] = result.vector_score
    if result.keyword_score is not None:
        metadata["keyword_score"] = result.keyword_score
    if rerank_score is not None:
        metadata["rerank_score"] = rerank_score
    metadata["reranker"] = reranker_name

    return SearchResult(
        text=result.text,
        score=rerank_score if rerank_score is not None else result.score,
        metadata=metadata,
        score_source=result.score_source,
        vector_score=result.vector_score,
        keyword_score=result.keyword_score,
        rerank_score=rerank_score,
        reranker=reranker_name,
    )
