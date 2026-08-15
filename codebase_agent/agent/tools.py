"""Agent 可调用的三个只读工具。"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.embeddings import EmbeddingProvider
from codebase_agent.rag.vector_store import SearchResult, search_code as _search_code


def list_files(project_root: str, file_pattern: str = "*.py", **kwargs: object) -> list[str]:
    """列出项目根目录及其子目录中匹配 pattern 的文件。

    返回按字母排序的相对路径列表（"/" 分隔）。
    """
    root = Path(project_root)
    if not root.is_dir():
        return []

    matched: list[str] = []
    for entry in sorted(root.rglob(file_pattern)):
        if entry.is_file():
            relative = str(entry.relative_to(root)).replace(os.sep, "/")
            matched.append(relative)
    return matched


def get_file_content(path: str, project_root: str, **kwargs: object) -> str:
    """读取项目内指定文件的内容（UTF-8）。

    调用前由 executor 校验路径安全——本函数假设 path 已通过安全校验。
    """
    full_path = (Path(project_root) / path).resolve()
    if not full_path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    return full_path.read_text(encoding="utf-8")


def search_code(
    query: str,
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    top_k: int = 5,
) -> list[SearchResult]:
    """语义搜索代码库——直接复用 rag.vector_store.search_code 全链路。

    核心规则：
      真实走 chunk → embedding → cosine → Top-K 排序，
      pytest 可使用假的 embedding 提供器隔离外部 API，
      但不能写死检索结果。
    """
    return _search_code(
        query=query,
        chunks=chunks,
        top_k=top_k,
        embedding_provider=embedding_provider,
    )
