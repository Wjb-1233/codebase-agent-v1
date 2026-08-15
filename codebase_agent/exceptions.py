class CodebaseError(Exception):
    """codebase-agent 项目异常的基类。"""

    def __init__(self, message="代码库分析过程发生错误", code=500):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.code}] {self.message}"


class GitHubAPIError(CodebaseError):
    """GitHub API 返回错误时抛出的异常。"""

    def __init__(self, status_code, message="GitHub API 错误"):
        self.status_code = status_code
        super().__init__(message=message, code=status_code or 500)


class RateLimitError(GitHubAPIError):
    """GitHub API 触发限流时抛出的异常。"""

    def __init__(self, reset_time):
        self.reset_time = reset_time
        super().__init__(status_code=403, message="GitHub API 请求频率超限")


class NetworkError(GitHubAPIError):
    """访问 GitHub API 遇到网络问题时抛出的异常。"""

    def __init__(self, message="访问 GitHub API 时发生网络错误"):
        super().__init__(status_code=None, message=message)


class ConfigError(CodebaseError):
    """配置错误时抛出的异常。"""

    def __init__(self, message="配置错误"):
        super().__init__(message=message, code=400)


class EmbeddingError(CodebaseError):
    """生成 embedding 失败时抛出的异常。"""

    def __init__(self, message="Embedding 生成失败"):
        super().__init__(message=message, code=502)


class LLMError(CodebaseError):
    """LLM 请求失败时抛出的异常。"""

    def __init__(self, message="LLM 请求失败"):
        super().__init__(message=message, code=502)
