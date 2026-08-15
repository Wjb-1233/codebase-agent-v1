"""RRF 融合与混合检索结果合并测试。"""

from __future__ import annotations

from codebase_agent.rag.hybrid_search import rrf_fuse, _chunk_key
from codebase_agent.rag.vector_store import SearchResult


# ── 测试辅助函数 ─────────────────────────────────────────────────


def _vr(chunk_id: str, score: float, text: str = "") -> SearchResult:
    """创建向量检索侧的 SearchResult。"""
    return SearchResult(
        text=text or chunk_id,
        score=score,
        metadata={"chunk_id": chunk_id, "file_path": "test.py"},
        score_source="vector",
        vector_score=score,
    )


def _kr(chunk_id: str, score: float, text: str = "") -> SearchResult:
    """创建关键词检索侧的 SearchResult。"""
    return SearchResult(
        text=text or chunk_id,
        score=score,
        metadata={"chunk_id": chunk_id, "file_path": "test.py"},
        score_source="keyword",
        keyword_score=score,
    )


# ── RRF 融合 ────────────────────────────────────────────────────


class TestRRFFusion:
    def test_both_paths_agree_top_item_wins(self):
        """两路检索都把同一个 chunk 排第一时，融合后它仍应排第一。"""
        vec = [_vr("a", 0.95), _vr("b", 0.80)]
        kw = [_kr("a", 15.0), _kr("b", 5.0)]
        result = rrf_fuse(vec, kw)

        assert result[0].metadata["chunk_id"] == "a"
        assert result[0].score_source == "both"
        assert result[0].vector_score == 0.95
        assert result[0].keyword_score == 15.0

    def test_disjoint_paths_both_present(self):
        """两路检索返回不同 chunk 时，融合结果应都保留。"""
        vec = [_vr("a", 0.90)]
        kw = [_kr("b", 10.0)]
        result = rrf_fuse(vec, kw)

        ids = {r.metadata["chunk_id"] for r in result}
        assert ids == {"a", "b"}

    def test_single_path_only(self):
        """只有一路有结果时，直接返回该路结果并重新标记排序分。"""
        vec = [_vr("x", 0.88)]
        result = rrf_fuse(vec, [])

        assert len(result) == 1
        assert result[0].metadata["chunk_id"] == "x"
        assert result[0].score_source == "vector"

    def test_empty_both_returns_empty(self):
        assert rrf_fuse([], []) == []

    def test_score_source_vector_only(self):
        result = rrf_fuse([_vr("x", 0.7)], [])
        assert result[0].score_source == "vector"
        assert result[0].vector_score == 0.7
        assert result[0].keyword_score is None

    def test_score_source_both_preserves_scores(self):
        result = rrf_fuse([_vr("shared", 0.85)], [_kr("shared", 20.0)])
        assert result[0].score_source == "both"
        assert result[0].vector_score == 0.85
        assert result[0].keyword_score == 20.0

    def test_rrf_score_is_not_raw_cosine_or_bm25(self):
        """RRF 会把原始分数替换成基于排名的融合分数。"""
        vec = [_vr("a", 0.95)]
        kw = []
        result = rrf_fuse(vec, kw)

        # k=60 时，第 1 名的 RRF 分数是 1/61，约等于 0.0164。
        assert 0.015 < result[0].score < 0.018

    def test_multiple_chunks_ranked_by_fusion(self):
        # a: vector rank 1, keyword rank 3  → RRF ≈ 1/61 + 1/63 = 0.0323
        # b: vector rank 2, keyword rank 1  → RRF ≈ 1/62 + 1/61 = 0.0328
        # c: vector rank 3, keyword rank 2  → RRF ≈ 1/63 + 1/62 = 0.0320
        vec = [_vr("a", 0.95), _vr("b", 0.90), _vr("c", 0.85)]
        kw = [_kr("b", 20.0), _kr("c", 15.0), _kr("a", 10.0)]
        result = rrf_fuse(vec, kw)

        # b 的综合排名最好，因此应该排第一。
        assert result[0].metadata["chunk_id"] == "b"


# ── chunk 去重键 ────────────────────────────────────────────────


class TestChunkKey:
    def test_uses_chunk_id_when_present(self):
        r = SearchResult(
            text="hello",
            score=0.5,
            metadata={"chunk_id": "abc-123", "file_path": "x.py"},
        )
        assert _chunk_key(r) == "abc-123"

    def test_falls_back_to_text(self):
        r = SearchResult(
            text="def hello(): pass",
            score=0.5,
            metadata={},
        )
        assert _chunk_key(r) == "def hello(): pass"
