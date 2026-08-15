"""代码 chunk 的 BM25 关键词检索，用来补充向量检索。

KeywordIndex 会提前构建索引，适合同一批 chunk 被多次查询的场景；
keyword_search() 适合一次性快速查询。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.vector_store import SearchResult

# ── 分词器 ─────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")


def _tokenize(text: str) -> list[str]:
    """把文本切成小写标识符 token。"""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


# ── BM25 索引 ─────────────────────────────────────────────────


class KeywordIndex:
    """基于代码 chunk 预构建的 BM25 索引。

    构建一次、多次查询；构建后索引不再修改，便于跨请求复用。
    """

    # BM25 超参数，使用常见默认值。
    _k1: float = 1.5
    _b: float = 0.75

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks 不能为空")

        self._chunks = chunks
        self._doc_count = len(chunks)

        # 每个文档只做一次分词。
        self._doc_tokens: list[list[str]] = [_tokenize(ch.text) for ch in chunks]
        self._doc_lengths: list[int] = [len(tokens) for tokens in self._doc_tokens]

        # IDF：统计每个词出现在多少文档中。
        df: dict[str, int] = defaultdict(int)
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] += 1
        self._idf: dict[str, float] = {
            term: math.log((self._doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

        # 平均文档长度，用于 BM25 长度归一化。
        self._avgdl: float = (
            sum(self._doc_lengths) / self._doc_count if self._doc_count else 0.0
        )

    # ── 对外接口 ─────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10,file_filter: str | None = None) -> list[SearchResult]:
        """按 BM25 分数返回最多 top_k 个 chunk。"""
        if not query.strip():
            raise ValueError("query 不能为空")
        if top_k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            ch = self._chunks[idx]
            if file_filter and not ch.file_path.startswith(file_filter):
                      continue   # 跳过不匹配目录的 chunk
            score = self._bm25_score(query_tokens, doc_tokens, idx)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in scores[:top_k]:
            ch = self._chunks[idx]
            results.append(
                SearchResult(
                    text=ch.text,
                    score=score,
                    metadata={
                        "chunk_id": ch.chunk_id,
                        "file_path": ch.file_path,
                        "start_line": ch.start_line,
                        "end_line": ch.end_line,
                        "symbol_name": ch.symbol_name,
                        "score_source": "keyword",
                    },
                    score_source="keyword",
                    keyword_score=score,
                )
            )
        return results

    def __len__(self) -> int:
        return self._doc_count

    # ── BM25 评分 ───────────────────────────────────────────

    def _bm25_score(
        self, query_tokens: list[str], doc_tokens: list[str], doc_idx: int
    ) -> float:
        """计算单个文档的 BM25 分数。"""
        dl = self._doc_lengths[doc_idx]
        # 当前文档的词频表。
        tf: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1

        score: float = 0.0
        for term in query_tokens:
            idf = self._idf.get(term, 0.0)
            f = tf[term]
            if f == 0:
                continue
            numerator = f * (self._k1 + 1.0)
            denominator = f + self._k1 * (1.0 - self._b + self._b * (dl / self._avgdl))
            score += idf * numerator / denominator
        return score


# ── 便捷函数 ─────────────────────────────────────────────────


def keyword_search(query: str, chunks: list[Chunk], top_k: int = 10) -> list[SearchResult]:
    """一次性关键词检索，会在函数内部临时构建索引。

    如果同一批 chunk 要查询多次，优先显式创建 KeywordIndex 后复用。
    """
    index = KeywordIndex(chunks)
    return index.search(query, top_k=top_k)
