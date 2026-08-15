import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from codebase_agent.backend.github_client import GitHubClient
from codebase_agent.exceptions import ConfigError, GitHubAPIError, NetworkError, RateLimitError
from codebase_agent.utils import timed_operation


def _error_message(error):
    if isinstance(error, RateLimitError):
        return f"GitHub API 限流: {error}"
    if isinstance(error, NetworkError):
        return f"网络错误: {error}"
    if isinstance(error, ConfigError):
        return f"配置错误: {error}"
    if isinstance(error, GitHubAPIError):
        return f"GitHub API 错误: {error}"
    return str(error)


async def main():
    parser = argparse.ArgumentParser(description="从 GitHub 仓库拉取文件列表并保存为 JSON")
    parser.add_argument("repo_url", help="GitHub 仓库地址")
    parser.add_argument("--output", default="output/repo_files.json", help="输出 JSON 文件路径")
    args = parser.parse_args()

    try:
        async with GitHubClient(args.repo_url) as client:
            with timed_operation("拉取文件树"):
                file_list = await client.get_file_tree()
    except (RateLimitError, NetworkError, ConfigError, GitHubAPIError) as error:
        print(_error_message(error), file=sys.stderr)
        raise SystemExit(1)

    fetched_at = datetime.now().isoformat()
    data = {
        "repo_url": args.repo_url,
        "owner": client.owner,
        "repo": client.repo_name,
        "fetched_at": fetched_at,
        "total_files": len(file_list),
        "files": [
            {"path": file_path, "type": Path(file_path).suffix, "size": None}
            for file_path in file_list
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with timed_operation("写入输出 JSON"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
