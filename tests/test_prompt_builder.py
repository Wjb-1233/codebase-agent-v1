"""Tests for build_rag_prompt pure function."""

import pytest

from codebase_agent.rag.prompt_builder import build_rag_prompt


class FakeChunk:
    def __init__(self, text, file_path, start_line, end_line):
        self.text = text
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line


def _make_chunks():
    return [
        FakeChunk("def handle_error(e):\n    raise", "app.py", 10, 12),
        FakeChunk("def connect_db():\n    return 'ok'", "db.py", 5, 7),
    ]


def test_build_rag_prompt_contains_question():
    prompt = build_rag_prompt("how to connect?", _make_chunks())
    assert "how to connect?" in prompt


def test_build_rag_prompt_contains_chunk_text():
    prompt = build_rag_prompt("test", _make_chunks())
    assert "def handle_error" in prompt
    assert "def connect_db" in prompt


def test_build_rag_prompt_contains_source_label():
    prompt = build_rag_prompt("test", _make_chunks())
    assert "[来源:app.py:10-12]" in prompt


def test_build_rag_prompt_empty_chunks_returns_fallback():
    prompt = build_rag_prompt("test", [])
    assert "没有找到相关代码片段" in prompt


def test_build_rag_prompt_includes_no_fabrication_rule():
    prompt = build_rag_prompt("test", _make_chunks())
    assert "不要编造" in prompt


def test_build_rag_prompt_includes_output_format():
    prompt = build_rag_prompt("test", _make_chunks())
    assert "先给结论" in prompt


def test_build_rag_prompt_handles_none_chunks():
    prompt = build_rag_prompt("test", None)
    assert "没有找到相关代码片段" in prompt


def test_build_rag_prompt_rejects_blank_question():
    with pytest.raises(ValueError, match="question"):
        build_rag_prompt("   ", _make_chunks())
