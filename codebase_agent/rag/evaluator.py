"""可解释的检索评估辅助工具。

提供单模式评估（基础向量检索）和双模式对比
（基础 vs 高级，即混合检索 + 重排 + 父文档扩展）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codebase_agent.exceptions import LLMError
from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.embeddings import EmbeddingProvider
from codebase_agent.rag.llm import LLMProvider, OpenAILLMProvider
from codebase_agent.rag.vector_store import search_code

if TYPE_CHECKING:
    from codebase_agent.rag.keyword_search import KeywordIndex
    from codebase_agent.rag.reranker import Reranker
    from codebase_agent.rag.parent_document import ParentDocument, build_parent_map, expand_with_parents


def hit_at_k(expected_file: str, retrieved_files: list[str], k: int) -> int:
    """期望文件出现在前 k 个结果中返回 1，否则返回 0。"""
    if k <= 0:
        return 0
    return int(expected_file in retrieved_files[:k])


def reciprocal_rank(expected_file: str, retrieved_files: list[str]) -> float:
    """返回期望文件首次出现的倒数排名。"""
    try:
        rank = retrieved_files.index(expected_file) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_file: str
    top_k: int = 5
    validation_error: str | None = None


@dataclass(frozen=True)
class EvalResult:
    question: str
    expected_file: str
    hit: int
    reciprocal_rank: float
    actual_files: list[str]
    failure_reason: str | None = None


def run_evaluation(
    cases: list[EvalCase],
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    *,
    keyword_index: "KeywordIndex | None" = None,
    reranker: "Reranker | None" = None,
    use_parent_document: bool = False,
) -> dict[str, object]:
    """评估所有案例，单条失败不中断整批。

    传入 *keyword_index* 和 *reranker* 时使用 hybrid 检索
    （向量 + 关键词 → RRF → 重排）。*use_parent_document* 为
    ``True`` 时，结果先扩展为完整 parent 上下文再检查命中。
    """
    results: list[EvalResult] = []

    # 启用父文档扩展时预构建一次父文档映射，所有案例复用
    parent_map: dict[str, object] | None = None
    if use_parent_document:
        from codebase_agent.rag.parent_document import build_parent_map as _bpm
        parent_map = _bpm(chunks)

    for case in cases:
        if case.validation_error:
            results.append(_failed_result(case, case.validation_error))
            continue

        try:
            search_results = search_code(
                query=case.question,
                chunks=chunks,
                top_k=case.top_k,
                embedding_provider=embedding_provider,
                keyword_index=keyword_index,
                reranker=reranker,
            )

            # 请求时扩展为父文档上下文
            if use_parent_document and parent_map:
                from codebase_agent.rag.parent_document import expand_with_parents as _exp
                search_results = _exp(search_results, parent_map, case.top_k)

            retrieved_files = [str(result.metadata["file_path"]) for result in search_results]
            hit = hit_at_k(case.expected_file, retrieved_files, case.top_k)
            failure_reason = None
            if not hit:
                failure_reason = f"期望文件未出现在前 {case.top_k} 个检索结果中"

            results.append(
                EvalResult(
                    question=case.question,
                    expected_file=case.expected_file,
                    hit=hit,
                    reciprocal_rank=reciprocal_rank(case.expected_file, retrieved_files),
                    actual_files=retrieved_files,
                    failure_reason=failure_reason,
                )
            )
        except Exception as exc:
            reason = f"检索执行失败（{type(exc).__name__}）: {exc}"
            results.append(_failed_result(case, reason))

    total = len(results)
    hit_count = sum(result.hit for result in results)
    hit_at_1_count = sum(
        int(bool(result.actual_files) and result.actual_files[0] == result.expected_file)
        for result in results
    )
    failed_samples = [result for result in results if result.hit == 0]

    return {
        "total": total,
        "hit_count": hit_count,
        "hit_at_1": hit_at_1_count / total if total else 0.0,
        "hit_at_k": hit_count / total if total else 0.0,
        "mrr": (
            sum(result.reciprocal_rank for result in results) / total
            if total
            else 0.0
        ),
        "results": results,
        "failed_samples": failed_samples,
    }


def _failed_result(case: EvalCase, reason: str) -> EvalResult:
    return EvalResult(
        question=case.question,
        expected_file=case.expected_file,
        hit=0,
        reciprocal_rank=0.0,
        actual_files=[],
        failure_reason=reason,
    )


# ── 对比评估 ──────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """基础 vs 高级检索的并排对比评估结果。"""

    total: int
    basic: dict[str, object]
    advanced: dict[str, object]
    basic_mode: str = "vector-only"
    advanced_mode: str = "hybrid+rerank+parent"


def compare_retrieval(
    cases: list[EvalCase],
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    *,
    keyword_index: "KeywordIndex",
    reranker: "Reranker",
    use_parent_document: bool = True,
) -> ComparisonResult:
    """对同一批案例同时运行基础和高级评估，返回并排对比。

    返回 ``ComparisonResult``，包含两套完整指标。
    """
    basic = run_evaluation(
        cases,
        chunks,
        embedding_provider,
        keyword_index=None,
        reranker=None,
        use_parent_document=False,
    )

    advanced = run_evaluation(
        cases,
        chunks,
        embedding_provider,
        keyword_index=keyword_index,
        reranker=reranker,
        use_parent_document=use_parent_document,
    )

    basic_results: list[EvalResult] = basic.get("results", [])
    advanced_results: list[EvalResult] = advanced.get("results", [])

    return ComparisonResult(
        total=len(cases),
        basic=basic,
        advanced=advanced,
    )


@dataclass(frozen=True)
class GenerationEvalCase:
    """生成质量评估样本。"""

    question: str
    answer: str
    contexts: list[str]
    expected_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationEvalResult:
    """RAGAS 风格生成质量评估结果。"""

    question: str
    faithfulness: float
    answer_relevance: float
    passed: bool
    unsupported_claims: list[str]
    missing_keywords: list[str]
    notes: str = ""
    evaluator: str = "heuristic"


def evaluate_generation(case: GenerationEvalCase) -> GenerationEvalResult:
    """评估 RAG 答案是否有上下文支撑，并且是否回答了问题。

    这里刻意避免网络调用，让 CI 和本地验收都可复现。它近似两个常见 RAGAS 维度：
    - faithfulness：答案里的说法应该能被检索上下文支撑。
    - answer_relevance：答案应该覆盖问题和期望关键词。
    """
    question_terms = _meaningful_terms(case.question)
    answer_terms = _meaningful_terms(case.answer)
    context_terms = _meaningful_terms("\n".join(case.contexts))
    expected_terms = {_normalize_term(term) for term in case.expected_keywords if term.strip()}

    if not case.answer.strip():
        return GenerationEvalResult(
            question=case.question,
            faithfulness=0.0,
            answer_relevance=0.0,
            passed=False,
            unsupported_claims=[],
            missing_keywords=sorted(expected_terms),
            notes="空答案",
            evaluator="heuristic",
        )

    answer_text = _normalize_text(case.answer)
    context_text = _normalize_text("\n".join(case.contexts))

    # 中文没有空格分词，轻量离线评估不引入分词器；faithfulness 主要检查代码标识符、
    # API 名、英文技术词是否被上下文支撑，避免把“数据库连接在”这类中文短语误判为幻觉。
    factual_answer_terms = {
        term for term in answer_terms - _STOPWORDS if _contains_ascii_token(term)
    }
    if not factual_answer_terms:
        factual_answer_terms = answer_terms - _STOPWORDS
    unsupported_terms = sorted(
        term for term in factual_answer_terms if term not in context_terms and term not in context_text
    )
    faithfulness = _ratio(len(factual_answer_terms) - len(unsupported_terms), len(factual_answer_terms))

    target_terms = (question_terms | expected_terms) - _STOPWORDS
    covered_terms = {
        term for term in target_terms if term in answer_terms or term in answer_text
    }
    answer_relevance = _ratio(len(covered_terms), len(target_terms)) if target_terms else 1.0

    missing_keywords = sorted(
        term for term in expected_terms if term not in answer_terms and term not in answer_text
    )
    passed = faithfulness >= 0.6 and answer_relevance >= 0.5 and not missing_keywords
    notes = "通过" if passed else "需要检查"

    return GenerationEvalResult(
        question=case.question,
        faithfulness=round(faithfulness, 4),
        answer_relevance=round(answer_relevance, 4),
        passed=passed,
        unsupported_claims=unsupported_terms[:20],
        missing_keywords=missing_keywords,
        notes=notes,
        evaluator="heuristic",
    )


class LLMJudgeGenerationEvaluator:
    """使用线上模型评审 RAG 生成质量。"""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        *,
        faithfulness_threshold: float = 0.75,
        relevance_threshold: float = 0.7,
    ) -> None:
        model = os.getenv("LLM_JUDGE_MODEL") or None
        self.llm = llm or OpenAILLMProvider(model=model)
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold

    def evaluate(self, case: GenerationEvalCase) -> GenerationEvalResult:
        if not case.answer.strip():
            return evaluate_generation(case)

        payload = self._parse_json(self.llm.generate(_build_generation_judge_prompt(case)))
        faithfulness = _clamp_score(payload.get("faithfulness"))
        answer_relevance = _clamp_score(payload.get("answer_relevance"))
        unsupported_claims = _string_list(payload.get("unsupported_claims"))
        missing_keywords = _string_list(payload.get("missing_keywords"))
        notes = str(payload.get("notes") or "").strip() or "线上模型评审完成"
        passed = (
            faithfulness >= self.faithfulness_threshold
            and answer_relevance >= self.relevance_threshold
            and not unsupported_claims
            and not missing_keywords
        )

        return GenerationEvalResult(
            question=case.question,
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            passed=passed,
            unsupported_claims=unsupported_claims[:20],
            missing_keywords=missing_keywords,
            notes=notes,
            evaluator="llm_judge",
        )

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, object]:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise LLMError("生成质量评审未返回 JSON")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("生成质量评审返回的 JSON 无效") from exc
        if not isinstance(data, dict):
            raise LLMError("生成质量评审返回结构无效")
        return data


def evaluate_generation_with_llm_judge(
    case: GenerationEvalCase,
    llm: LLMProvider | None = None,
) -> GenerationEvalResult:
    """调用线上模型进行生成质量评审。"""
    return LLMJudgeGenerationEvaluator(llm=llm).evaluate(case)


def _build_generation_judge_prompt(case: GenerationEvalCase) -> str:
    contexts = "\n\n".join(f"[证据 {index + 1}]\n{context}" for index, context in enumerate(case.contexts))
    keywords = ", ".join(case.expected_keywords) if case.expected_keywords else "无"
    return f"""你是 RAG 生成质量评审员。请只根据给定证据评估答案，不要补充外部知识。

评分维度：
1. faithfulness：答案中的关键事实是否都能被证据支撑，0 到 1。
2. answer_relevance：答案是否直接回答用户问题并覆盖期望关键词，0 到 1。

判定要求：
- 如果答案出现证据中没有的关键事实，把这些内容写入 unsupported_claims。
- 如果期望关键词没有被答案覆盖，把这些词写入 missing_keywords。
- 不要因为答案更长就给高分。
- 只返回一个 JSON 对象，不要 Markdown，不要解释正文。

JSON 格式：
{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "unsupported_claims": [],
  "missing_keywords": [],
  "notes": "一句中文评审结论"
}}

用户问题：
{case.question}

期望关键词：
{keywords}

检索证据：
{contexts or "无"}

待评估答案：
{case.answer}
"""


def _clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMError("生成质量评审分数无效") from exc
    return min(1.0, max(0.0, score))


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise LLMError("生成质量评审列表字段无效")
    return [str(item).strip() for item in value if str(item).strip()]


def evaluate_generation_batch(cases: list[GenerationEvalCase]) -> dict[str, object]:
    """对固定样本集运行确定性的生成质量评估。"""
    results = [evaluate_generation(case) for case in cases]
    total = len(results)
    pass_count = sum(1 for result in results if result.passed)
    return {
        "total": total,
        "pass_count": pass_count,
        "pass_rate": pass_count / total if total else 0.0,
        "avg_faithfulness": sum(result.faithfulness for result in results) / total if total else 0.0,
        "avg_answer_relevance": sum(result.answer_relevance for result in results) / total if total else 0.0,
        "results": results,
    }


_STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "use",
    "what", "when", "where", "why", "with", "you", "your",
}


def _meaningful_terms(text: str) -> set[str]:
    terms: set[str] = set()
    current: list[str] = []
    for char in text.lower():
        if char.isalnum() or char in {"_", "-"}:
            current.append(char)
        else:
            _flush_term(current, terms)
    _flush_term(current, terms)
    return {term for term in terms if len(term) > 1}


def _flush_term(current: list[str], terms: set[str]) -> None:
    if not current:
        return
    term = _normalize_term("".join(current))
    if term:
        terms.add(term)
    current.clear()


def _normalize_term(term: str) -> str:
    return term.strip().lower().replace(".", "_")


def _normalize_text(text: str) -> str:
    return _normalize_term(text).replace(" ", "")


def _contains_ascii_token(term: str) -> bool:
    return any(char.isascii() and (char.isalnum() or char == "_") for char in term)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))
