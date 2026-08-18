"""POST /chat 接口测试。"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import codebase_agent.backend.main as main_module
from codebase_agent.backend.main import app, get_embedding_provider, get_llm_provider
from codebase_agent.exceptions import ConfigError, EmbeddingError, LLMError
from codebase_agent.rag.evaluator import GenerationEvalResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides(monkeypatch):
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    monkeypatch.delenv("GENERATION_EVALUATOR", raising=False)
    main_module._reranker = None
    yield
    app.dependency_overrides.clear()
    main_module._reranker = None


# ── 测试辅助函数 ─────────────────────────────────────────

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


def _make_fake_embedding():
    provider = MagicMock()
    provider.embed_texts = _fake_embed
    return provider


def _make_fake_llm(answer="基于代码片段的模拟回答。"):
    provider = MagicMock()
    provider.generate.return_value = answer
    return provider


def _two_files():
    return [
        {"file_path": "db.py", "content": "def connect_db():\n    return 'connected'\n"},
        {"file_path": "ui.py", "content": "def render_html():\n    return '<div/>'\n"},
    ]


# ── 正常路径 ────────────────────────────────────────────

def test_chat_returns_answer_and_chunks():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    fake_llm = _make_fake_llm()
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm

    resp = client.post(
        "/chat",
        json={"question": "connect to database", "top_k": 3, "files": _two_files()},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert len(body["answer"]) > 0
    assert "retrieved_chunks" in body
    assert len(body["retrieved_chunks"]) >= 1
    prompt = fake_llm.generate.call_args.args[0]
    assert "db.py" in prompt
    assert "connect_db" in prompt


def test_chat_retrieved_chunks_have_all_fields():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={
            "question": "render html",
            "top_k": 1,
            "files": [{"file_path": "ui.py", "content": "def render_html():\n    pass\n"}],
        },
    )

    assert resp.status_code == 200
    item = resp.json()["retrieved_chunks"][0]

    assert isinstance(item["text"], str) and len(item["text"]) > 0
    assert isinstance(item["score"], float)
    assert 0.0 <= item["score"] <= 1.01
    assert isinstance(item["chunk_id"], str) and len(item["chunk_id"]) > 0
    assert isinstance(item["file_path"], str)
    assert isinstance(item["start_line"], int)
    assert isinstance(item["end_line"], int)
    assert "symbol_name" in item
    assert item["parent_id"] == "ui.py:render_html"
    assert item["parent_start_line"] == 1
    assert item["parent_end_line"] >= 1
    assert item["score_source"] in {"vector", "keyword", "both"}
    assert item["reranker"] == "identity"


def test_chat_top_k_respected():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={"question": "test", "top_k": 1, "files": _two_files()},
    )

    assert resp.status_code == 200
    assert len(resp.json()["retrieved_chunks"]) <= 1


# ── 边界场景：空输入和非法输入 ─────────────────────────

def test_chat_empty_question_returns_422():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={"question": "", "files": _two_files()},
    )

    assert resp.status_code == 422


def test_chat_blank_question_returns_422():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={"question": "   ", "files": _two_files()},
    )

    assert resp.status_code == 422


def test_chat_empty_files_returns_200_with_fallback():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    fake_llm = _make_fake_llm()
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm

    resp = client.post(
        "/chat",
        json={"question": "how to connect", "files": []},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_chunks"] == []
    fake_llm.generate.assert_not_called()


def test_chat_top_k_zero_rejected():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={"question": "test", "top_k": 0, "files": _two_files()},
    )

    assert resp.status_code == 422


def test_chat_no_valid_chunks_returns_200():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={
            "question": "test",
            "files": [{"file_path": "empty.py", "content": ""}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["retrieved_chunks"] == []


# ── 边界场景：外部提供器异常 ───────────────────────────

def test_chat_llm_error_returns_502():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()

    bad_llm = MagicMock()
    bad_llm.generate.side_effect = LLMError("simulated LLM failure")
    app.dependency_overrides[get_llm_provider] = lambda: bad_llm

    resp = client.post(
        "/chat",
        json={
            "question": "connect to db",
            "files": [{"file_path": "a.py", "content": "def foo():\n    pass\n"}],
        },
    )

    assert resp.status_code == 502


def test_chat_embedding_error_returns_502():
    bad_embed = MagicMock()
    bad_embed.embed_texts.side_effect = EmbeddingError("simulated embedding failure")
    app.dependency_overrides[get_embedding_provider] = lambda: bad_embed

    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={
            "question": "test",
            "files": [{"file_path": "a.py", "content": "def foo():\n    pass\n"}],
        },
    )

    assert resp.status_code == 502


def test_chat_llm_config_error_returns_500():
    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()

    bad_llm = MagicMock()
    bad_llm.generate.side_effect = ConfigError("missing key")
    app.dependency_overrides[get_llm_provider] = lambda: bad_llm

    resp = client.post(
        "/chat",
        json={
            "question": "connect to db",
            "files": [{"file_path": "a.py", "content": "def foo():\n    pass\n"}],
        },
    )

    assert resp.status_code == 500


def test_chat_embedding_config_error_returns_500():
    bad_embed = MagicMock()
    bad_embed.embed_texts.side_effect = ConfigError("missing key")
    app.dependency_overrides[get_embedding_provider] = lambda: bad_embed
    app.dependency_overrides[get_llm_provider] = lambda: _make_fake_llm()

    resp = client.post(
        "/chat",
        json={
            "question": "test",
            "files": [{"file_path": "a.py", "content": "def foo():\n    pass\n"}],
        },
    )

    assert resp.status_code == 500


def test_chat_stream_returns_sse_chunks():
    class StreamingLLM:
        def __init__(self):
            self.last_prompt = None

        def generate(self, prompt: str) -> str:
            self.last_prompt = prompt
            return "streamed answer"

        def stream(self, prompt: str):
            self.last_prompt = prompt
            yield "streamed "
            yield "answer"

    app.dependency_overrides[get_embedding_provider] = lambda: _make_fake_embedding()
    streaming_llm = StreamingLLM()
    app.dependency_overrides[get_llm_provider] = lambda: streaming_llm

    with client.stream(
        "POST",
        "/chat/stream",
        json={"question": "connect to database", "top_k": 2, "files": _two_files()},
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: chunk" in body
    assert "streamed answer" in body
    assert "event: done" in body
    assert "retrieved_chunks" in body
    assert "vector_backend" in body


def test_generation_evaluation_api_scores_grounded_answer():
    resp = client.post(
        "/evaluate/generation",
        json={
            "question": "How does connect_db return database status?",
            "answer": "connect_db returns connected database status",
            "contexts": ["def connect_db():\n    return 'connected'  # database status"],
            "expected_keywords": ["connect_db", "connected"],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluator"] == "heuristic"
    assert data["faithfulness"] >= 0.6
    assert data["answer_relevance"] >= 0.5
    assert data["missing_keywords"] == []


def test_generation_evaluation_api_can_use_llm_judge(monkeypatch):
    monkeypatch.setenv("GENERATION_EVALUATOR", "llm_judge")

    def fake_llm_judge(case):
        return GenerationEvalResult(
            question=case.question,
            faithfulness=0.91,
            answer_relevance=0.86,
            passed=True,
            unsupported_claims=[],
            missing_keywords=[],
            notes="线上模型评审通过",
            evaluator="llm_judge",
        )

    monkeypatch.setattr(main_module, "evaluate_generation_with_llm_judge", fake_llm_judge)

    resp = client.post(
        "/evaluate/generation",
        json={
            "question": "数据库连接在哪里实现？",
            "answer": "数据库连接在 get_engine 中实现。",
            "contexts": ["def get_engine(database_url): return create_engine(database_url)"],
            "expected_keywords": ["get_engine"],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluator"] == "llm_judge"
    assert data["faithfulness"] == 0.91
    assert data["answer_relevance"] == 0.86
    assert data["notes"] == "线上模型评审通过"
