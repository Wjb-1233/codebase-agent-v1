"""Parent Document 检索测试 — 构建 + 扩展 + 去重。"""

from __future__ import annotations

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.parent_document import (
    ParentDocument,
    build_parent_map,
    attach_parent_metadata,
    expand_with_parents,
)
from codebase_agent.rag.vector_store import SearchResult


# ── 辅助函数 ──────────────────────────────────────────────────────


def _ch(chunk_id: str, text: str, file_path: str = "test.py",
        start_line: int = 1, end_line: int = 5,
        symbol_name: str = "my_func") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
    )


def _sr(
    chunk_id: str, score: float, text: str = "",
    parent_id: str | None = None,
    file_path: str = "test.py",
) -> SearchResult:
    meta: dict[str, object] = {"chunk_id": chunk_id, "file_path": file_path}
    if parent_id:
        meta["parent_id"] = parent_id
    return SearchResult(text=text or chunk_id, score=score, metadata=meta)


def _parent(parent_id: str, text: str = "", file_path: str = "test.py") -> ParentDocument:
    return ParentDocument(
        parent_id=parent_id,
        text=text or parent_id,
        file_path=file_path,
        start_line=1,
        end_line=10,
        symbol_name=parent_id.split(":")[-1] if ":" in parent_id else "",
    )


# ── build_parent_map ─────────────────────────────────────────────


class TestBuildParentMap:
    def test_empty_chunks_returns_empty(self):
        assert build_parent_map([]) == {}

    def test_same_symbol_merged_into_one_parent(self):
        chunks = [
            _ch("a.py:def:foo:1", "def foo():\n    x = 1\n", file_path="a.py", start_line=1, end_line=2, symbol_name="foo"),
            _ch("a.py:def:foo:15", "    y = 2\n    return y\n", file_path="a.py", start_line=15, end_line=16, symbol_name="foo"),
        ]
        result = build_parent_map(chunks)
        assert len(result) == 1
        parent = result["a.py:foo"]
        assert parent.symbol_name == "foo"
        assert parent.start_line == 1
        assert parent.end_line == 16
        assert "x = 1" in parent.text
        assert "y = 2" in parent.text

    def test_different_symbols_separate_parents(self):
        chunks = [
            _ch("a.py:def:foo:1", "def foo(): pass\n", file_path="a.py", symbol_name="foo"),
            _ch("a.py:def:bar:10", "def bar(): pass\n", file_path="a.py", symbol_name="bar"),
        ]
        result = build_parent_map(chunks)
        assert len(result) == 2
        assert "a.py:foo" in result
        assert "a.py:bar" in result

    def test_no_symbol_name_falls_back_to_module(self):
        chunks = [
            _ch("mod.py:module:module:1", "import os\n", file_path="mod.py", symbol_name=""),
        ]
        result = build_parent_map(chunks)
        assert "mod.py:module" in result



    def test_repeated_code_lines_are_preserved(self):
        chunks = [
            _ch("a.py:def:foo:1", "def foo():\n    return None\n", file_path="a.py", start_line=1, end_line=2, symbol_name="foo"),
            _ch("a.py:def:foo:3", "    if True:\n        return None\n", file_path="a.py", start_line=3, end_line=4, symbol_name="foo"),
        ]

        parent = build_parent_map(chunks)["a.py:foo"]

        assert parent.text.count("return None") == 2

    def test_overlapping_line_ranges_are_not_duplicated(self):
        chunks = [
            _ch("a.py:def:foo:1", "def foo():\n    value = 1\n", file_path="a.py", start_line=1, end_line=2, symbol_name="foo"),
            _ch("a.py:def:foo:2", "    value = 1\n    return value\n", file_path="a.py", start_line=2, end_line=3, symbol_name="foo"),
        ]

        parent = build_parent_map(chunks)["a.py:foo"]

        assert parent.text.count("value = 1") == 1
        assert "return value" in parent.text

# ── expand_with_parents ──────────────────────────────────────────


class TestExpandWithParents:
    def test_same_parent_deduplicated_keeps_highest_score(self):
        results = [
            _sr("c1", 0.9, parent_id="a.py:foo"),
            _sr("c2", 0.7, parent_id="a.py:foo"),
        ]
        pmap = {"a.py:foo": _parent("a.py:foo", "def foo():\n    pass\n")}
        expanded = expand_with_parents(results, pmap, top_k=5)
        assert len(expanded) == 1
        assert expanded[0].score == 0.9
        assert "def foo()" in expanded[0].text
        assert expanded[0].metadata["parent_id"] == "a.py:foo"

    def test_parent_text_replaces_child_text(self):
        results = [_sr("c1", 0.95, text="short chunk", parent_id="a.py:foo")]
        pmap = {"a.py:foo": _parent("a.py:foo", "full function code here")}
        expanded = expand_with_parents(results, pmap, top_k=5)
        assert expanded[0].text == "full function code here"

    def test_parent_start_end_lines_updated(self):
        results = [_sr("c1", 0.8, parent_id="a.py:foo")]
        pmap = {"a.py:foo": ParentDocument(
            parent_id="a.py:foo", text="code", file_path="a.py",
            start_line=5, end_line=50,
        )}
        expanded = expand_with_parents(results, pmap, top_k=5)
        assert expanded[0].metadata["start_line"] == 5
        assert expanded[0].metadata["end_line"] == 50

    def test_missing_parent_id_passed_through_unchanged(self):
        results = [_sr("orphan", 0.8, text="orphan chunk")]
        expanded = expand_with_parents(results, {}, top_k=5)
        assert len(expanded) == 1
        assert expanded[0].text == "orphan chunk"
        assert expanded[0].score == 0.8

    def test_parent_id_not_in_map_passed_through(self):
        results = [_sr("c1", 0.8, parent_id="missing:func")]
        expanded = expand_with_parents(results, {"other:func": _parent("other:func")}, top_k=5)
        assert len(expanded) == 1
        assert expanded[0].metadata["chunk_id"] == "c1"

    def test_empty_parent_map_returns_unchanged(self):
        results = [_sr("c1", 0.8), _sr("c2", 0.7)]
        expanded = expand_with_parents(results, {}, top_k=5)
        assert len(expanded) == 2
        assert [r.metadata["chunk_id"] for r in expanded] == ["c1", "c2"]

    def test_empty_results_returns_empty(self):
        assert expand_with_parents([], {"a.py:foo": _parent("a.py:foo")}, top_k=5) == []

    def test_top_k_respected_after_dedup(self):
        results = [
            _sr("c1", 0.9, parent_id="a.py:foo"),
            _sr("c2", 0.8, parent_id="a.py:bar"),
            _sr("c3", 0.7, parent_id="a.py:baz"),
        ]
        pmap = {f"a.py:{name}": _parent(f"a.py:{name}") for name in ("foo", "bar", "baz")}
        expanded = expand_with_parents(results, pmap, top_k=2)
        assert len(expanded) == 2

    def test_different_parents_kept_separate(self):
        results = [
            _sr("c1", 0.9, parent_id="a.py:foo"),
            _sr("c2", 0.8, parent_id="a.py:bar"),
        ]
        pmap = {
            "a.py:foo": _parent("a.py:foo", "foo code"),
            "a.py:bar": _parent("a.py:bar", "bar code"),
        }
        expanded = expand_with_parents(results, pmap, top_k=5)
        assert len(expanded) == 2

    def test_mix_parent_and_orphan(self):
        results = [
            _sr("c1", 0.9, parent_id="a.py:foo"),
            _sr("orphan", 0.7, text="orphan text"),
        ]
        pmap = {"a.py:foo": _parent("a.py:foo", "foo code")}
        expanded = expand_with_parents(results, pmap, top_k=5)
        assert len(expanded) == 2
        texts = [r.text for r in expanded]
        assert "foo code" in texts
        assert "orphan text" in texts

class TestAttachParentMetadata:
    def test_attaches_parent_trace_without_replacing_text(self):
        results = [_sr("c1", 0.8, text="child chunk", parent_id="a.py:foo")]
        pmap = {"a.py:foo": ParentDocument(
            parent_id="a.py:foo",
            text="full parent code",
            file_path="a.py",
            start_line=10,
            end_line=20,
            symbol_name="foo",
        )}

        enriched = attach_parent_metadata(results, pmap)

        assert enriched[0].text == "child chunk"
        assert enriched[0].metadata["parent_id"] == "a.py:foo"
        assert enriched[0].metadata["parent_start_line"] == 10
        assert enriched[0].metadata["parent_end_line"] == 20


