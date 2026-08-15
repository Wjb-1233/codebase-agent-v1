"""RAG 输入层使用的代码切块工具。"""

from __future__ import annotations

from dataclasses import dataclass
import re


MAX_SYMBOL_LINES = 50
CHUNK_SIZE = 30
_SYMBOL_RE = re.compile(r"(?:async\s+)?(def|class)\s+(\w+)")


@dataclass(frozen=True)
class Chunk:
    """一段可以被向量化和检索的代码片段。"""

    chunk_id: str
    text: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ""


def chunk_file(file_path: str, code_text: str) -> list[Chunk]:
    """围绕顶层函数和类，把一个源码文件切成多个 chunk。"""
    lines = code_text.splitlines()
    if not lines or not code_text.strip():
        return []

    symbols = _extract_top_level_symbols(lines)
    if not symbols:
        return _build_chunks(lines, file_path, "module", "module", 0, len(lines))

    chunks: list[Chunk] = []
    for symbol in symbols:
        chunks.extend(
            _build_chunks(
                lines=lines,
                file_path=file_path,
                symbol_type=symbol["type"],
                symbol_name=symbol["name"],
                start=symbol["start"],
                end=symbol["end"],
            )
        )
    return chunks


def _extract_top_level_symbols(lines: list[str]) -> list[dict[str, object]]:
    symbols: list[dict[str, object]] = []

    for line_number, line in enumerate(lines):
        if line.startswith((" ", "\t")):
            continue

        match = _SYMBOL_RE.match(line)
        if not match:
            continue

        if symbols:
            symbols[-1]["end"] = line_number

        symbol_type, symbol_name = match.groups()
        symbols.append(
            {
                "type": symbol_type,
                "name": symbol_name,
                "start": line_number,
                "end": len(lines),
            }
        )

    return symbols


def _build_chunks(
    lines: list[str],
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start: int,
    end: int,
) -> list[Chunk]:
    if end - start <= MAX_SYMBOL_LINES:
        return [_make_chunk(lines, file_path, symbol_type, symbol_name, start, end)]

    chunks = []
    for chunk_start in range(start, end, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, end)
        chunks.append(_make_chunk(lines, file_path, symbol_type, symbol_name, chunk_start, chunk_end))
    return chunks


def _make_chunk(
    lines: list[str],
    file_path: str,
    symbol_type: str,
    symbol_name: str,
    start: int,
    end: int,
) -> Chunk:
    return Chunk(
        chunk_id=f"{file_path}:{symbol_type}:{symbol_name}:{start}",
        text="\n".join(lines[start:end]),
        file_path=file_path,
        start_line=start + 1,
        end_line=end,
        symbol_name=symbol_name,
    )
