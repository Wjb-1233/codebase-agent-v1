import asyncio
import json
import logging

import pytest

from scripts import github_fetcher
from codebase_agent.exceptions import ConfigError, GitHubAPIError, NetworkError, RateLimitError


class FakeGitHubClient:
    def __init__(self, repo_url):
        self.repo_url = repo_url
        self.owner = "sample"
        self.repo_name = "project"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get_file_tree(self):
        return ["README.md", "src/app.py"]


def test_main_writes_output_and_logs_timed_operations(monkeypatch, tmp_path, caplog):
    output_path = tmp_path / "repo_files.json"
    monkeypatch.setattr(github_fetcher, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(
        "sys.argv",
        [
            "scripts/github_fetcher.py",
            "https://github.com/sample/project",
            "--output",
            str(output_path),
        ],
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(github_fetcher.main())

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["repo_url"] == "https://github.com/sample/project"
    assert data["owner"] == "sample"
    assert data["repo"] == "project"
    assert data["total_files"] == 2
    assert [item["path"] for item in data["files"]] == ["README.md", "src/app.py"]
    assert "拉取文件树" in caplog.text
    assert "写入输出 JSON" in caplog.text


@pytest.mark.parametrize(
    "error, expected_message",
    [
        (RateLimitError(reset_time=None), "限流"),
        (NetworkError("Request timed out"), "网络错误"),
        (GitHubAPIError(status_code=404, message="Repository not found"), "github api 错误"),
    ],
)
def test_main_exits_with_friendly_message_for_fetch_errors(
    monkeypatch, tmp_path, capsys, error, expected_message
):
    output_path = tmp_path / "repo_files.json"

    class RaisingGitHubClient(FakeGitHubClient):
        async def get_file_tree(self):
            raise error

    monkeypatch.setattr(github_fetcher, "GitHubClient", RaisingGitHubClient)
    monkeypatch.setattr(
        "sys.argv",
        [
            "scripts/github_fetcher.py",
            "https://github.com/sample/project",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(github_fetcher.main())

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert expected_message in captured.err.lower()
    assert not output_path.exists()


def test_main_exits_with_friendly_message_for_config_error(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "repo_files.json"

    class InvalidGitHubClient:
        def __init__(self, repo_url):
            raise ConfigError("Invalid GitHub repository URL")

    monkeypatch.setattr(github_fetcher, "GitHubClient", InvalidGitHubClient)
    monkeypatch.setattr(
        "sys.argv",
        [
            "scripts/github_fetcher.py",
            "not-a-url",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(github_fetcher.main())

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "配置错误" in captured.err
    assert not output_path.exists()
