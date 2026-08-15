from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from codebase_agent.backend.github_client import GitHubClient
from codebase_agent.exceptions import ConfigError, GitHubAPIError, NetworkError, RateLimitError


def test_url_parsing():
    client = GitHubClient("https://github.com/fastapi/fastapi")
    assert client.owner == "fastapi"
    assert client.repo_name == "fastapi"


def test_invalid_url_raises_config_error():
    with pytest.raises(ConfigError) as exc_info:
        GitHubClient("not a url")

    assert "GitHub 仓库地址无效" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_file_success():
    """正常返回文件列表"""
    mock_resp1 = Mock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {"default_branch": "main"}

    mock_resp2 = Mock()
    mock_resp2.status_code = 200
    mock_resp2.json.return_value = {
        "tree": [
            {"path": "a.py", "type": "blob"},
            {"path": "b.py", "type": "blob"},
        ]
    }

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_resp1, mock_resp2]

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            files = await client.get_file_tree()

    assert files == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_rate_limit_raises_rate_limit_error():
    """API 403 -> 抛限流异常"""
    mock_resp1 = Mock()
    mock_resp1.status_code = 403
    mock_resp1.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp1

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.get_file_tree()
    assert exc_info.value.code == 403


@pytest.mark.asyncio
async def test_timeout_raises_network_error():
    """网络超时 -> 抛网络异常"""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.TimeoutException("timed out")

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(NetworkError) as exc_info:
                await client.get_file_tree()

    assert "超时" in str(exc_info.value)


@pytest.mark.asyncio
async def test_connect_error_raises_network_error():
    """httpx.ConnectError（连接被拒/DNS失败）→ 应转换为 NetworkError"""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(NetworkError) as exc_info:
                await client.get_file_tree()

    assert "超时" in str(exc_info.value)


@pytest.mark.asyncio
async def test_repo_not_found_raises_github_api_error():
    """repo 不存在 -> 抛 GitHub API 异常"""
    mock_resp1 = Mock()
    mock_resp1.status_code = 404
    mock_resp1.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp1

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.get_file_tree()

    assert exc_info.value.status_code == 404
    assert exc_info.value.code == 404


@pytest.mark.asyncio
async def test_rate_limit_429_raises_rate_limit_error():
    """文件树接口返回 429 → RateLimitError"""
    mock_resp1 = Mock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {"default_branch": "main"}

    mock_resp2 = Mock()
    mock_resp2.status_code = 429
    mock_resp2.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_resp1, mock_resp2]

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.get_file_tree()

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_file_tree_500_raises_github_api_error():
    """文件树接口返回 500 → GitHubAPIError"""
    mock_resp1 = Mock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {"default_branch": "main"}

    mock_resp2 = Mock()
    mock_resp2.status_code = 500
    mock_resp2.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_resp1, mock_resp2]

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.get_file_tree()

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_file_tree_connection_error_raises_network_error():
    """文件树请求连接断开 → NetworkError"""
    mock_resp1 = Mock()
    mock_resp1.status_code = 200
    mock_resp1.json.return_value = {"default_branch": "main"}

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_resp1, httpx.TimeoutException("timed out")]

    with patch("httpx.AsyncClient", return_value=mock_client):
        async with GitHubClient("https://github.com/test/repo") as client:
            with pytest.raises(NetworkError) as exc_info:
                await client.get_file_tree()

    assert "超时" in str(exc_info.value)


# ═══ get_file_contents 加量测试 ═══


@pytest.mark.asyncio
async def test_get_file_contents_success():
    """并发抓取多个文件 → 返回 {path: content} dict"""
    async with GitHubClient("https://github.com/test/repo") as client:
        mock = AsyncMock()
        mock.side_effect = ["content_a", "content_b"]
        client.get_file_content = mock
        result = await client.get_file_contents(["a.py", "b.py"], limit=2)
        assert result == {"a.py": "content_a", "b.py": "content_b"}


@pytest.mark.asyncio
async def test_get_file_contents_skips_failures():
    """一个文件失败 → 跳过它，其余正常返回"""
    async with GitHubClient("https://github.com/test/repo") as client:
        mock = AsyncMock()
        mock.side_effect = ["ok", NetworkError("boom")]
        client.get_file_content = mock
        result = await client.get_file_contents(["a.py", "b.py"], limit=2)
        assert result == {"a.py": "ok"}


@pytest.mark.asyncio
async def test_get_file_contents_connect_error_skipped():
    """httpx.ConnectError → 被 fetch_one 捕获 → 跳过，不泄露原始异常"""
    async with GitHubClient("https://github.com/test/repo") as client:
        mock = AsyncMock()
        mock.side_effect = httpx.ConnectError("connection refused")
        client.get_file_content = mock
        result = await client.get_file_contents(["a.py"], limit=1)
        assert result == {}
