from unittest.mock import patch, MagicMock

import pytest

from codebase_agent.backend.database import (
    build_database_url,
    init_db,
    save_analysis,
    list_recent_analyses,
)


def test_build_database_url_prefers_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:////data/analysis.db")

    assert build_database_url("local.db") == "sqlite:////data/analysis.db"


def test_build_database_url_builds_sqlite_path(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_path = tmp_path / "local.db"

    assert build_database_url(str(db_path)) == f"sqlite:///{db_path}"


def test_init_db_and_save(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    save_analysis(db_path, "https://github.com/a/b", 42)
    results = list_recent_analyses(db_path)
    assert len(results) == 1
    assert results[0]["repo_url"] == "https://github.com/a/b"
    assert results[0]["count"] == 42


def test_list_recent_order(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    save_analysis(db_path, "https://github.com/a/d", 30)
    save_analysis(db_path, "https://github.com/a/e", 63)
    save_analysis(db_path, "https://github.com/a/f", 42)
    results = list_recent_analyses(db_path, 2)
    assert len(results) == 2
    assert results[0]["repo_url"] == "https://github.com/a/f"


def test_list_recent_empty_db(tmp_path):
    db_path = str(tmp_path / "test_empty.db")
    init_db(db_path)
    result = list_recent_analyses(db_path)
    assert result == []


def test_save_analysis_rollback_on_commit_failure():
    """commit 失败 → rollback 被调用 → 异常向上传播。"""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.commit.side_effect = Exception("磁盘写入失败")

    with patch(
        "codebase_agent.backend.database.Session", return_value=mock_session
    ):
        with pytest.raises(Exception, match="磁盘写入失败"):
            save_analysis("dummy.db", "https://github.com/test/repo", 42)

        mock_session.rollback.assert_called_once()
