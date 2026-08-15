import pytest


@pytest.fixture(autouse=True)
def isolate_database_url(monkeypatch):
    """测试默认使用显式依赖，避免本机 .env 污染用例结果。"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_EMBEDDING_MODEL", raising=False)
