"""代码检索使用的父文档扩展。

检索阶段先搜索较小的子 chunk；生成阶段再把命中的子 chunk 扩展到周围的函数、
类或模块上下文，让 LLM 看到足够完整的代码证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.vector_store import SearchResult


@dataclass(frozen=True)
class ParentDocument:
    """由一个或多个子 chunk 组装出来的较大代码上下文。"""

    parent_id: str
    text: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ""
    child_ids: tuple[str, ...] = field(default_factory=tuple)


def build_parent_map(chunks: list[Chunk]) -> dict[str, ParentDocument]:
    """按文件和符号把 chunk 分组为父文档。"""
    if not chunks:
        return {}

    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(_parent_key(chunk), []).append(chunk)

    parent_map: dict[str, ParentDocument] = {}
    for parent_id, group in groups.items():
        ordered = sorted(group, key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_id))
        start_line = min(chunk.start_line for chunk in ordered)
        end_line = max(chunk.end_line for chunk in ordered)
        symbol_name = ordered[0].symbol_name

        parent_map[parent_id] = ParentDocument(
            parent_id=parent_id,
            text=_merge_chunk_text(ordered),
            file_path=ordered[0].file_path,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            child_ids=tuple(chunk.chunk_id for chunk in ordered),
        )

    return parent_map


def attach_parent_metadata(
    results: list[SearchResult],
    parent_map: dict[str, ParentDocument],
) -> list[SearchResult]:
    """保留子 chunk 文本，同时挂上父文档 ID 和行号范围，方便追踪。"""
    if not results or not parent_map:
        return results

    enriched: list[SearchResult] = []
    for result in results:
        parent_id = _result_parent_id(result)
        if not parent_id or parent_id not in parent_map:
            enriched.append(result)
            continue

        parent = parent_map[parent_id]
        metadata = dict(result.metadata)
        metadata["parent_id"] = parent_id
        metadata["parent_start_line"] = parent.start_line
        metadata["parent_end_line"] = parent.end_line

        enriched.append(_copy_result(result, metadata=metadata))

    return enriched


def expand_with_parents(
    results: list[SearchResult],
    parent_map: dict[str, ParentDocument],
    top_k: int,
) -> list[SearchResult]:
    """用父文档文本替换子 chunk 文本，并按父文档 ID 去重。"""
    if not results or not parent_map or top_k <= 0:
        return results[:top_k] if top_k > 0 else []

    best: dict[str, SearchResult] = {}
    order: list[str] = []

    for result in results:
        parent_id = _result_parent_id(result)
        if not parent_id or parent_id not in parent_map:
            key = str(result.metadata.get("chunk_id", result.text))
            if key not in best:
                best[key] = result
                order.append(key)
            elif result.score > best[key].score:
                best[key] = result
            continue

        parent = parent_map[parent_id]
        metadata = dict(result.metadata)
        metadata["parent_id"] = parent_id
        metadata["parent_start_line"] = parent.start_line
        metadata["parent_end_line"] = parent.end_line
        metadata["start_line"] = parent.start_line
        metadata["end_line"] = parent.end_line

        expanded = _copy_result(result, text=parent.text, metadata=metadata)

        if parent_id not in best:
            best[parent_id] = expanded
            order.append(parent_id)
        elif expanded.score > best[parent_id].score:
            best[parent_id] = expanded

    return [best[key] for key in order][:top_k]


def _merge_chunk_text(chunks: list[Chunk]) -> str:
    """按行号范围合并 chunk，同时保留合理的重复代码行。"""
    merged_lines: list[str] = []
    last_end_line = 0

    for chunk in chunks:
        lines = chunk.text.splitlines()
        if chunk.start_line <= last_end_line:
            overlap = last_end_line - chunk.start_line + 1
            lines = lines[overlap:]
        if lines:
            merged_lines.extend(lines)
        last_end_line = max(last_end_line, chunk.end_line)

    return "\n".join(merged_lines)


def _result_parent_id(result: SearchResult) -> str:
    parent_id = str(result.metadata.get("parent_id", ""))
    if parent_id:
        return parent_id

    file_path = str(result.metadata.get("file_path", ""))
    symbol_name = str(result.metadata.get("symbol_name", ""))
    if not file_path:
        return ""
    return f"{file_path}:{symbol_name or 'module'}"


def _copy_result(
    result: SearchResult,
    *,
    text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    return SearchResult(
        text=result.text if text is None else text,
        score=result.score,
        metadata=dict(result.metadata) if metadata is None else metadata,
        score_source=result.score_source,
        vector_score=result.vector_score,
        keyword_score=result.keyword_score,
        rerank_score=result.rerank_score,
        reranker=result.reranker,
    )


def _parent_key(chunk: Chunk) -> str:
    symbol = chunk.symbol_name or "module"
    return f"{chunk.file_path}:{symbol}"
