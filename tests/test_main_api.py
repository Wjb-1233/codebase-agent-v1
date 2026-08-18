from unittest.mock import AsyncMock, MagicMock, patch
from codebase_agent.exceptions import NetworkError
from fastapi.testclient import TestClient

from codebase_agent.backend.main import app, get_db_path, get_max_files
from codebase_agent.backend.database import init_db, save_analysis, list_recent_analyses
client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_success():
    resp1 = MagicMock()
    resp1.status_code = 200
    resp1.json.return_value = {"default_branch": "main"}

    resp2 = MagicMock()
    resp2.status_code = 200
    resp2.json.return_value = {
        "tree": [
            {"path": "main.py", "type": "blob"},
            {"path": "app.py", "type": "blob"},
        ]
    }

    fake_client = AsyncMock()
    fake_client.get = AsyncMock(side_effect=[resp1, resp2])

    with patch(
        "codebase_agent.backend.github_client.httpx.AsyncClient",
        return_value=fake_client,
    ):
        resp = client.post(
            "/analyze", json={"repo_url": "https://github.com/test/repo"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["files"] == ["main.py", "app.py"]
    assert data["count"] == 2


def test_analyze_missing_repo_url():
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422
def test_analyze_missing_repo_url_400():
    resp1 = client.post("/analyze", json={"repo_url":""})
    resp2 = client.post("/analyze", json={"repo_url":"    "})
    assert resp1.status_code==400
    assert resp2.status_code==400
def test_analyze_network_error_returns_502():
    mock_gh = MagicMock()
    mock_gh.__aenter__ = AsyncMock(side_effect=NetworkError("连接失败"))
    mock_gh.__aexit__ = AsyncMock()
    with patch(
        "codebase_agent.backend.main.GitHubClient",
        return_value=mock_gh,
    ):
        resp = client.post("/analyze", json={"repo_url": "https://github.com/test/repo"})
    assert resp.status_code == 502
def test_analyze_too_many_files_returns_422():
    app.dependency_overrides[get_max_files] = lambda: 1
    try:
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"default_branch": "main"}

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {
            "tree": [
                {"path": "main.py", "type": "blob"},
                {"path": "app.py", "type": "blob"},
                {"path": "utils.py", "type": "blob"},
            ]
        }

        fake_client = AsyncMock()
        fake_client.get = AsyncMock(side_effect=[resp1, resp2])

        with patch(
            "codebase_agent.backend.github_client.httpx.AsyncClient",
            return_value=fake_client,
        ):
            resp = client.post("/analyze", json={"repo_url": "https://github.com/test/repo"})

        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()

def test_analyze_saves_and_history_returns_record(tmp_path):
    db_path = str(tmp_path / "test_history.db")
    app.dependency_overrides[get_db_path] = lambda: db_path
    try:
        init_db(db_path)
        with patch("codebase_agent.backend.main.GitHubClient") as mock_github_cls:
            mock_instance = AsyncMock()
            mock_instance.get_file_tree = AsyncMock(return_value=["main.py", "app.py", "utils.py"])
            mock_instance.__aenter__.return_value = mock_instance
            mock_github_cls.return_value = mock_instance
            payload = {"repo_url": "https://github.com/test/sample"}
            resp = client.post("/analyze", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["files"] == ["main.py", "app.py", "utils.py"]
            assert data["count"] == 3
        history_resp = client.get("/history?limit=10")
        assert history_resp.status_code == 200
        history_list = history_resp.json()
        assert len(history_list) == 1
        record = history_list[0]
        assert record["repo_url"] == "https://github.com/test/sample"
        assert record["count"] == 3
    finally:
        app.dependency_overrides.clear()
def test_history_empty_db(tmp_path):
    db_path = str(tmp_path / "test_history_empty.db")
    app.dependency_overrides[get_db_path] = lambda: db_path
    try:
        init_db(db_path)
        resp = client.get("/history")
        assert resp.status_code==200
        data=resp.json()
        assert data==[]
    finally:
        app.dependency_overrides.clear()

def test_history_respects_limit(tmp_path):
    db_path = str(tmp_path / "test_history_limit.db")
    app.dependency_overrides[get_db_path] = lambda: db_path
    try:
        init_db(db_path)
        for i in range(5):
            save_analysis(db_path, f"https://github.com/test/repo{i}", i * 10)
        resp = client.get("/history?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        # limit=1 被正确遵守，只返回一条记录
        assert "repo_url" in data[0]
        assert "count" in data[0]
    finally:
        app.dependency_overrides.clear()
