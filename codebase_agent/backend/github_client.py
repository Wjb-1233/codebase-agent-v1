import asyncio
import base64
import os
from logging import getLogger
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

from codebase_agent.exceptions import ConfigError, GitHubAPIError, NetworkError, RateLimitError
load_dotenv()
logger = getLogger(__name__)


class GitHubClient:
    """异步 GitHub API 客户端。

    用法:
        async with GitHubClient(repo_url) as client:
            files = await client.get_file_tree()
            content = await client.get_file_content(files[0])
    """

    def __init__(self, repo_url: str) -> None:
        self.token = os.getenv("GITHUB_TOKEN")
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) >= 2:
            self.owner = path_parts[0]
            self.repo_name = path_parts[1]
        else:
            raise ConfigError("GitHub 仓库地址无效")
        self._client = None

    async def __aenter__(self):
        # 未配置 GITHUB_TOKEN 时不发送 Authorization 头，
        # 否则 "Bearer None" 会被 GitHub API 判为 401，匿名配额（60 次/小时）无法使用。
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._client = httpx.AsyncClient(timeout=10, headers=headers)
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()

    async def validate_repo(self) -> dict:
        api_url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}"
        try:
            logger.info("验证仓库: %s", api_url)
            response = await self._client.get(api_url)
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.error("请求超时: %s", api_url)
            raise NetworkError("访问 GitHub API 超时")
        if response.status_code == 403 or response.status_code == 429:
            logger.error("API 限流: status=%d", response.status_code)
            raise RateLimitError(reset_time=None)
        if response.status_code == 200:
            return response.json()
        raise GitHubAPIError(
            status_code=response.status_code,
            message="仓库不存在或当前 Token 无权访问",
        )

    async def get_file_tree(self) -> list[str]:
        repo_info = await self.validate_repo()
        if not repo_info:
            logger.error("仓库不存在或无法访问: %s/%s", self.owner, self.repo_name)
            raise GitHubAPIError(
                status_code=404, message="仓库不存在或当前 Token 无权访问"
            )
        branch = repo_info.get("default_branch", "main")
        url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/git/trees/{branch}?recursive=1"
        logger.info("获取文件树: %s", url)
        try:
            resp = await self._client.get(url)
        except (httpx.TimeoutException, httpx.ConnectError):
            raise NetworkError("访问 GitHub API 超时")
        if resp.status_code != 200:
            raise GitHubAPIError(
                status_code=resp.status_code, message="拉取仓库文件树失败"
            )
        tree = resp.json()["tree"]
        return [item["path"] for item in tree if item["type"] == "blob"]

    async def get_file_content(self, file_path: str) -> str:
        url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/contents/{file_path}"
        logger.info("获取文件内容: %s", url)
        try:
            resp = await self._client.get(url)
        except (httpx.TimeoutException, httpx.ConnectError):
            raise NetworkError("访问 GitHub API 超时")
        if resp.status_code != 200:
            raise GitHubAPIError(
                status_code=resp.status_code, message="拉取文件内容失败"
            )
        content = resp.json()
        if content["encoding"] == "base64":
            return base64.b64decode(content["content"]).decode("utf-8")
        return content["content"]
    async def get_file_contents(
            self, file_paths: list[str], limit: int = 5
        ) -> dict[str, str]:
            semaphore = asyncio.Semaphore(limit)

            async def fetch_one(path):
                async with semaphore:
                    try:
                        content = await self.get_file_content(path)
                        return (path, content)
                    except Exception as e:
                        logger.warning("跳过 %s: %s", path, e)
                        return (path, None)
            tasks = [fetch_one(path) for path in file_paths]
            results = await asyncio.gather(*tasks)
            return {p: c for p, c in results if c is not None}
    async def get_py_files(self) -> list[str]:
        tree = await self.get_file_tree()
        return [f for f in tree if f.endswith('.py')]
