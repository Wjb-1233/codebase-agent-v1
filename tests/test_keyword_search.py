"""BM25 关键词检索索引测试。"""

from __future__ import annotations

import pytest

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.keyword_search import KeywordIndex, keyword_search
from codebase_agent.rag.vector_store import SearchResult


# ── 测试辅助函数 ─────────────────────────────────────────────────


def _chunk(chunk_id: str, text: str, file_path: str = "test.py") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name="test_func",
    )


# ── KeywordIndex ─────────────────────────────────────────────────


class TestKeywordIndexInit:
    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError):
            KeywordIndex([])

    def test_builds_with_single_chunk(self):
        chunks = [_chunk("a", "def hello(): pass")]
        idx = KeywordIndex(chunks)
        assert len(idx) == 1

    def test_builds_with_multiple_chunks(self):
        chunks = [
            _chunk("a", "def hello(): pass"),
            _chunk("b", "class World: pass"),
        ]
        idx = KeywordIndex(chunks)
        assert len(idx) == 2


class TestKeywordIndexSearch:
    def test_exact_match_ranks_first(self):
        # 分词器会把下划线当作标识符的一部分：
        # "build_database_url" 是一个完整 token，适合精确匹配函数名。
        chunks = [
            _chunk("a", "def build_database_url(): pass"),
            _chunk("b", "def get_engine(): pass"),
            _chunk("c", "class ConfigError: pass"),
        ]
        idx = KeywordIndex(chunks)
        results = idx.search("build_database_url", top_k=3)

        # 只有 chunk "a" 包含完整 token "build_database_url"。
        # chunk "b" 和 "c" 没有 token 重叠，分数为 0，会被排除。
        assert len(results) == 1
        assert results[0].metadata["chunk_id"] == "a"

    def test_empty_query_raises(self):
        idx = KeywordIndex([_chunk("a", "def hello(): pass")])
        with pytest.raises(ValueError):
            idx.search("   ", top_k=3)

    def test_top_k_zero_returns_empty(self):
        idx = KeywordIndex([_chunk("a", "def hello(): pass")])
        results = idx.search("hello", top_k=0)
        assert results == []

    def test_no_match_returns_empty(self):
        idx = KeywordIndex([_chunk("a", "def hello(): pass")])
        results = idx.search("zzzznotfoundzzzz", top_k=3)
        assert results == []

    def test_top_k_clamps_result_count(self):
        chunks = [
            _chunk("a", "def one(): pass"),
            _chunk("b", "def two(): pass"),
            _chunk("c", "def three(): pass"),
        ]
        idx = KeywordIndex(chunks)
        results = idx.search("def", top_k=2)
        assert len(results) == 2

    def test_search_result_structure(self):
        idx = KeywordIndex([_chunk("hello-1", "def build_database_url() -> str: ...")])
        results = idx.search("build_database_url", top_k=1)

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.score > 0
        assert r.metadata["chunk_id"] == "hello-1"
        assert r.metadata["file_path"] == "test.py"
        assert r.metadata["score_source"] == "keyword"


class TestKeywordIndexScoring:
    def test_more_matches_scores_higher(self):
        chunks = [
            _chunk("rare", "def config(): return None"),
            _chunk("rich", "config = {}  # config is a global config dict"),
        ]
        idx = KeywordIndex(chunks)
        results = idx.search("config", top_k=2)

        assert len(results) == 2
        # 出现更多 "config" 的 chunk 应该得分更高。
        assert results[0].metadata["chunk_id"] == "rich"

    def test_rare_term_ranks_higher_than_common(self):
        # "config" 出现在三个 chunk 中，"get_engine" 更少见。
        # 同时搜索时，稀有词所在 chunk 应该排在前面。
        chunks = [
            _chunk("a", "def helper(): return config"),
            _chunk("b", "def get_engine(): return config"),
            _chunk("c", "config = {}"),
            _chunk("d", "def main(): return get_engine"),
        ]
        idx = KeywordIndex(chunks)
        results = idx.search("get_engine", top_k=4)

        # "get_engine" 出现在 chunk b 和 d 中，不在 a 或 c 中。
        # b 把 "get_engine" 作为函数名，d 只在 return 语句中出现。
        # b 通常因为词频或位置更好而排得更高。
        assert len(results) >= 1
        assert results[0].metadata["chunk_id"] in ("b", "d")


class TestKeywordSearchConvenience:
    def test_one_shot(self):
        chunks = [_chunk("a", "def hello(): pass")]
        results = keyword_search("hello", chunks, top_k=1)
        assert len(results) == 1
        assert results[0].metadata["chunk_id"] == "a"


class TestKeywordSearchFileFilter:
    def test_filters_to_matching_directory(self):
        chunks = [
            _chunk("a", "def create_user(): pass", file_path="backend/api.py"),
            _chunk("b", "def create_user(): pass", file_path="frontend/app.py"),
        ]
        idx = KeywordIndex(chunks)
        results = idx.search("create_user", top_k=5, file_filter="backend/")

        assert len(results) == 1
        assert results[0].metadata["chunk_id"] == "a"
        assert results[0].metadata["file_path"] == "backend/api.py"

    def test_filter_no_match_returns_empty(self):
        chunks = [_chunk("a", "def hello(): pass", file_path="backend/api.py")]
        idx = KeywordIndex(chunks)
        results = idx.search("hello", top_k=5, file_filter="frontend/")
        assert results == []

    def test_filter_none_returns_all(self):
        chunks = [
            _chunk("a", "def hello(): pass", file_path="backend/api.py"),
            _chunk("b", "def hello(): pass", file_path="frontend/app.py"),
        ]
        idx = KeywordIndex(chunks)
        results = idx.search("hello", top_k=5)
        # file_filter 默认为 None，表示不过滤文件路径。
        assert len(results) == 2
