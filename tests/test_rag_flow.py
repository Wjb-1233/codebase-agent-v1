"""检索接口与问答接口的端到端回归测试。"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from codebase_agent.backend.main import app, get_embedding_provider, get_llm_provider


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def _fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lower = text.lower()
        if "database" in lower or "connect" in lower or "db" in lower:
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


def test_search_then_chat_return_the_same_top_source():
    embedding_provider = MagicMock()
    embedding_provider.embed_texts = _fake_embed
    llm_provider = MagicMock()
    llm_provider.generate.return_value = "数据库连接定义在 db.py。"
    app.dependency_overrides[get_embedding_provider] = lambda: embedding_provider
    app.dependency_overrides[get_llm_provider] = lambda: llm_provider

    files = [
        {"file_path": "db.py", "content": "def connect_database():\n    return 'ok'\n"},
        {"file_path": "ui.py", "content": "def render_page():\n    return '<main/>'\n"},
    ]
    search_response = client.post(
        "/search",
        json={"query": "where is the database connection", "top_k": 2, "files": files},
    )
    chat_response = client.post(
        "/chat",
        json={"question": "where is the database connection", "top_k": 2, "files": files},
    )

    assert search_response.status_code == 200
    assert chat_response.status_code == 200
    search_source = search_response.json()["results"][0]["file_path"]
    chat_source = chat_response.json()["retrieved_chunks"][0]["file_path"]
    assert search_source == chat_source == "db.py"
    assert chat_response.json()["answer"] == "数据库连接定义在 db.py。"
    assert "db.py" in llm_provider.generate.call_args.args[0]
