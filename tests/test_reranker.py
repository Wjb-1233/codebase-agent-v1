"""重排器测试：IdentityReranker + CrossEncoderReranker（测试假模型）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_agent.rag.reranker import CrossEncoderReranker, IdentityReranker
from codebase_agent.rag.vector_store import SearchResult


# ── 辅助函数 ──────────────────────────────────────────────────────


def _sr(chunk_id: str, score: float, text: str = "") -> SearchResult:
    return SearchResult(text=text or chunk_id, score=score, metadata={"chunk_id": chunk_id})


# ── 直接返回型重排器 ────────────────────────────────────────────


class TestIdentityReranker:
    def test_returns_top_k_unchanged(self):
        rr = IdentityReranker()
        candidates = [_sr("a", 0.9), _sr("b", 0.8), _sr("c", 0.7)]
        result = rr.rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0].metadata["chunk_id"] == "a"
        assert result[1].metadata["chunk_id"] == "b"
        assert result[0].score == 0.9
        assert result[1].score == 0.8

    def test_top_k_exceeds_candidates_returns_all(self):
        rr = IdentityReranker()
        candidates = [_sr("a", 0.9)]
        result = rr.rerank("query", candidates, top_k=5)
        assert len(result) == 1

    def test_empty_candidates(self):
        rr = IdentityReranker()
        assert rr.rerank("query", [], top_k=5) == []


# ── CrossEncoder 重排器（测试假模型） ──────────────────────────


class TestCrossEncoderReranker:
    """CrossEncoderReranker 测试：模拟 sentence_transformers.CrossEncoder。

    不下载任何模型，pytest 不需要网络。
    """

    def test_sorts_by_cross_encoder_score(self):
        """模拟 CrossEncoder 返回 [0.2, 0.9, 0.5]，验证按高分重排。"""
        fake_model = MagicMock()
        # predict 返回类似 numpy 数组的列表。
        fake_model.predict.return_value = [0.2, 0.9, 0.5]

        with patch.object(CrossEncoderReranker, "_ensure_model", return_value=fake_model):
            rr = CrossEncoderReranker()
            candidates = [
                _sr("a", 0.9, "def connect_db"),
                _sr("b", 0.8, "def render_html"),
                _sr("c", 0.7, "def get_engine"),
            ]
            result = rr.rerank("database connection", candidates, top_k=3)

        # 应按 CrossEncoder 分数降序：b(0.9) > c(0.5) > a(0.2)
        assert result[0].metadata["chunk_id"] == "b"
        assert result[1].metadata["chunk_id"] == "c"
        assert result[2].metadata["chunk_id"] == "a"
        assert result[0].score == pytest.approx(0.9)
        assert result[0].rerank_score == pytest.approx(0.9)
        assert result[0].reranker == "cross_encoder"

    def test_top_k_truncation_after_rerank(self):
        """只返回 Top-K，即使候选更多。"""
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.8, 0.2, 0.6]

        with patch.object(CrossEncoderReranker, "_ensure_model", return_value=fake_model):
            rr = CrossEncoderReranker()
            candidates = [_sr("a", 0.9), _sr("b", 0.8), _sr("c", 0.7)]
            result = rr.rerank("query", candidates, top_k=2)

        assert len(result) == 2
        # 按测试分数排序：a(0.8) > c(0.6) > b(0.2)。
        assert result[0].metadata["chunk_id"] == "a"
        assert result[1].metadata["chunk_id"] == "c"

    def test_candidates_fewer_than_top_k_still_reranks(self):
        """候选少于 Top-K 时仍调用模型重排，但不截断。"""
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.5]

        with patch.object(CrossEncoderReranker, "_ensure_model", return_value=fake_model):
            rr = CrossEncoderReranker()
            candidates = [_sr("a", 0.9)]
            result = rr.rerank("query", candidates, top_k=5)

        assert len(result) == 1
        assert result[0].score == pytest.approx(0.5)  # score 替换为 rerank 分数
        fake_model.predict.assert_called_once()

    def test_empty_candidates_returns_empty(self):
        fake_model = MagicMock()
        with patch.object(CrossEncoderReranker, "_ensure_model", return_value=fake_model):
            rr = CrossEncoderReranker()
            assert rr.rerank("query", [], top_k=5) == []
        fake_model.predict.assert_not_called()

    def test_preserves_metadata_and_score_source(self):
        """重排后 metadata、score_source、vector_score 等字段不丢失。"""
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.7, 0.3]

        with patch.object(CrossEncoderReranker, "_ensure_model", return_value=fake_model):
            rr = CrossEncoderReranker()
            candidates = [
                SearchResult(
                    text="code a", score=0.9,
                    metadata={"chunk_id": "a", "file_path": "a.py"},
                    score_source="both", vector_score=0.9, keyword_score=8.0,
                ),
                SearchResult(
                    text="code b", score=0.8,
                    metadata={"chunk_id": "b", "file_path": "b.py"},
                    score_source="vector", vector_score=0.8, keyword_score=None,
                ),
            ]
            result = rr.rerank("query", candidates, top_k=2)

        # 第一名的 score 替换为重排分数。
        assert result[0].score == pytest.approx(0.7)
        # metadata 保留。
        assert result[0].metadata["chunk_id"] == "a"
        assert result[0].metadata["file_path"] == "a.py"
        # 原始检索分数保留
        assert result[0].vector_score == 0.9
        assert result[0].keyword_score == 8.0
        assert result[0].score_source == "both"
        assert result[0].rerank_score == pytest.approx(0.7)
        assert result[0].reranker == "cross_encoder"
        assert result[0].metadata["rerank_score"] == pytest.approx(0.7)
        assert result[0].metadata["reranker"] == "cross_encoder"

    def test_query_chunk_pairs_correct(self):
        """验证传给 CrossEncoder 的 query-document 对格式正确。"""
        fake_model = MagicMock()
        fake_model.predict.return_value = [1.0, 0.5]

        with patch.object(CrossEncoderReranker, "_ensure_model", return_value=fake_model):
            rr = CrossEncoderReranker()
            candidates = [_sr("a", 0.9, "db connect"), _sr("b", 0.8, "html render")]
            rr.rerank("database", candidates, top_k=2)

        # 验证传入模型的 query-document 对。
        pairs = fake_model.predict.call_args[0][0]
        assert len(pairs) == 2
        assert pairs[0] == ["database", "db connect"]
        assert pairs[1] == ["database", "html render"]

    def test_falls_back_to_identity_when_model_fails(self):
        with patch.object(CrossEncoderReranker, "_ensure_model", side_effect=RuntimeError("load failed")):
            rr = CrossEncoderReranker()
            candidates = [_sr("a", 0.9), _sr("b", 0.8)]
            result = rr.rerank("query", candidates, top_k=2)

        assert [item.metadata["chunk_id"] for item in result] == ["a", "b"]
        assert result[0].score == pytest.approx(0.9)
        assert result[0].reranker == "identity"
