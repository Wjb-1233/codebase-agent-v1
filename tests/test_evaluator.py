import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.evaluator import (
    ComparisonResult,
    EvalCase,
    GenerationEvalCase,
    compare_retrieval,
    evaluate_generation,
    hit_at_k,
    reciprocal_rank,
    run_evaluation,
)
from scripts.evaluate_rag import load_cases


def _fake_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lower = text.lower()
        if "connect" in lower or "database" in lower or "db" in lower:
            vectors.append([1.0, 0.0, 0.0])
        elif "render" in lower or "html" in lower or "ui" in lower:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


def _make_fake_provider():
    provider = MagicMock()
    provider.embed_texts = _fake_embed
    return provider


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="db.py:func:connect_db:1",
            text="def connect_db():\n    return 'connected'\n",
            file_path="db.py",
            start_line=1,
            end_line=2,
            symbol_name="connect_db",
        ),
        Chunk(
            chunk_id="ui.py:func:render_html:1",
            text="def render_html():\n    return '<div/>'\n",
            file_path="ui.py",
            start_line=1,
            end_line=2,
            symbol_name="render_html",
        ),
    ]


def test_hit_and_reciprocal_rank():
    retrieved_files = ["file1.py", "file2.py", "file3.py"]

    assert hit_at_k("file1.py", retrieved_files, 3) == 1
    assert reciprocal_rank("file1.py", retrieved_files) == 1.0
    assert hit_at_k("file3.py", retrieved_files, 2) == 0
    assert reciprocal_rank("file3.py", retrieved_files) == pytest.approx(1 / 3)
    assert reciprocal_rank("missing.py", retrieved_files) == 0.0


def test_run_evaluation_single_hit(chunks: list[Chunk]):
    cases = [EvalCase("how to connect database", "db.py", top_k=2)]

    result = run_evaluation(cases, chunks, _make_fake_provider())

    assert result["total"] == 1
    assert result["hit_count"] == 1
    assert result["hit_at_1"] == 1.0
    assert result["hit_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["failed_samples"] == []


def test_run_evaluation_mixed(chunks: list[Chunk]):
    cases = [
        EvalCase("how to connect database", "db.py", top_k=2),
        EvalCase("how to render html", "ui.py", top_k=2),
        EvalCase("what is chunker", "chunker.py", top_k=2),
    ]

    result = run_evaluation(cases, chunks, _make_fake_provider())

    assert result["total"] == 3
    assert result["hit_count"] == 2
    assert result["hit_at_k"] == pytest.approx(2 / 3)
    assert result["mrr"] > 0.5
    assert len(result["failed_samples"]) == 1
    failed = result["failed_samples"][0]
    assert failed.expected_file == "chunker.py"
    assert failed.failure_reason == "期望文件未出现在前 2 个检索结果中"


def test_invalid_case_does_not_stop_valid_case(chunks: list[Chunk]):
    cases = [
        EvalCase("broken", "", top_k=0, validation_error="缺少 expected_file"),
        EvalCase("how to connect database", "db.py", top_k=2),
    ]

    result = run_evaluation(cases, chunks, _make_fake_provider())

    assert result["total"] == 2
    assert result["hit_count"] == 1
    assert result["hit_at_k"] == 0.5
    assert result["failed_samples"][0].failure_reason == "缺少 expected_file"


def test_embedding_count_error_is_recorded(chunks: list[Chunk]):
    provider = MagicMock()
    provider.embed_texts.side_effect = [
        [[1.0, 0.0]],
        [[1.0, 0.0]],
    ]

    result = run_evaluation(
        [EvalCase("how to connect database", "db.py", top_k=2)],
        chunks,
        provider,
    )

    assert result["hit_count"] == 0
    assert len(result["failed_samples"]) == 1
    assert result["failed_samples"][0].failure_reason == (
        "检索执行失败（ValueError）: chunks 和 vectors 数量必须一致"
    )


def test_load_cases_keeps_malformed_sample(tmp_path: Path):
    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text(
        json.dumps(
            [
                {"question": "connect database", "expected_file": "db.py", "top_k": 2},
                {"question": "缺少期望文件"},
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(eval_path)

    assert len(cases) == 2
    assert cases[0].validation_error is None
    assert "expected_file" in cases[1].validation_error


def test_run_evaluation_empty_cases():
    result = run_evaluation([], [], _make_fake_provider())

    assert result["total"] == 0
    assert result["hit_count"] == 0
    assert result["hit_at_1"] == 0.0
    assert result["hit_at_k"] == 0.0
    assert result["mrr"] == 0.0


# ── 对比评估 ────────────────────────────────────────────────────


def test_compare_retrieval_runs_both_modes(chunks: list[Chunk]):
    from codebase_agent.rag.keyword_search import KeywordIndex
    from codebase_agent.rag.reranker import IdentityReranker

    cases = [EvalCase("how to connect database", "db.py", top_k=2)]
    ki = KeywordIndex(chunks)
    rr = IdentityReranker()

    result = compare_retrieval(
        cases, chunks, _make_fake_provider(),
        keyword_index=ki, reranker=rr, use_parent_document=False,
    )

    assert result.total == 1
    assert "hit_count" in result.basic
    assert "hit_count" in result.advanced
    assert "mrr" in result.basic
    assert "mrr" in result.advanced
    assert result.basic_mode == "vector-only"
    assert result.advanced_mode == "hybrid+rerank+parent"


def test_compare_retrieval_advanced_not_worse_than_basic(chunks: list[Chunk]):
    from codebase_agent.rag.keyword_search import KeywordIndex
    from codebase_agent.rag.reranker import IdentityReranker

    cases = [
        EvalCase("how to connect database", "db.py", top_k=2),
        EvalCase("how to render html", "ui.py", top_k=2),
    ]
    ki = KeywordIndex(chunks)
    rr = IdentityReranker()

    result = compare_retrieval(
        cases, chunks, _make_fake_provider(),
        keyword_index=ki, reranker=rr, use_parent_document=False,
    )

    # 使用测试提供器和直接返回型重排器时，两种模式的分数应该一致。
    assert result.basic["mrr"] == result.advanced["mrr"]


def test_run_evaluation_with_hybrid_params(chunks: list[Chunk]):
    from codebase_agent.rag.keyword_search import KeywordIndex
    from codebase_agent.rag.reranker import IdentityReranker

    cases = [EvalCase("how to connect database", "db.py", top_k=2)]
    ki = KeywordIndex(chunks)
    rr = IdentityReranker()

    result = run_evaluation(
        cases, chunks, _make_fake_provider(),
        keyword_index=ki, reranker=rr,
    )

    assert result["total"] == 1
    assert result["hit_count"] == 1
    assert result["mrr"] == 1.0


def test_run_evaluation_with_parent_document(chunks: list[Chunk]):
    cases = [EvalCase("how to connect database", "db.py", top_k=2)]

    result = run_evaluation(
        cases, chunks, _make_fake_provider(),
        use_parent_document=True,
    )

    assert result["total"] == 1
    # 父文档扩展不应破坏命中判断。
    assert result["hit_count"] == 1


def test_evaluate_generation_matches_chinese_expected_keyword():
    case = GenerationEvalCase(
        question="数据库连接在哪里实现？",
        answer="数据库连接在 get_engine 中实现，并通过 create_engine 创建连接。",
        contexts=["def get_engine(database_url): return create_engine(database_url)"],
        expected_keywords=["get_engine", "create_engine", "数据库"],
    )

    result = evaluate_generation(case)

    assert result.missing_keywords == []
    assert result.faithfulness == 1.0
    assert result.answer_relevance >= 0.5
