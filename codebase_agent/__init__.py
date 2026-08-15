"""codebase_agent — 代码分析工具包。

提供代码分析、GitHub 客户端、工具函数和自定义异常。
"""

from codebase_agent.code_analyzer import analyze
from codebase_agent.exceptions import (
    CodebaseError,
    ConfigError,
    EmbeddingError,
    GitHubAPIError,
    NetworkError,
    RateLimitError,
)
from codebase_agent.utils import cache_result, timed_operation

__all__ = [
    "analyze",
    "cache_result",
    "timed_operation",
    "CodebaseError",
    "ConfigError",
    "EmbeddingError",
    "GitHubAPIError",
    "NetworkError",
    "RateLimitError",
]
