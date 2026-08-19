"""使用固定问题集运行 RAG 检索评估。

用法:
    python scripts/evaluate_rag.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from codebase_agent.rag.chunker import Chunk, chunk_file
from codebase_agent.rag.embeddings import OpenAIEmbeddingProvider
from codebase_agent.rag.evaluator import EvalCase, run_evaluation
from codebase_agent.exceptions import ConfigError


def load_cases(json_path: Path) -> list[EvalCase]:
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("评估集必须是 JSON 数组")

    cases: list[EvalCase] = []
    for index, item in enumerate(raw, start=1):
        try:
            if not isinstance(item, dict):
                raise ValueError("评估样本必须是 JSON 对象")

            question = item["question"]
            expected_file = item["expected_file"]
            top_k = item.get("top_k", 5)
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question 必须是非空字符串")
            if not isinstance(expected_file, str) or not expected_file.strip():
                raise ValueError("expected_file 必须是非空字符串")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k 必须是正整数")

            cases.append(
                EvalCase(
                    question=question.strip(),
                    expected_file=expected_file.strip(),
                    top_k=top_k,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            question = f"<第 {index} 条无效样本>"
            if isinstance(item, dict) and isinstance(item.get("question"), str):
                question = item["question"] or question
            cases.append(
                EvalCase(
                    question=question,
                    expected_file="",
                    top_k=0,
                    validation_error=f"第 {index} 条样本无效: {exc}",
                )
            )
    return cases


def collect_chunks(code_dir: Path) -> list[Chunk]:
    """扫描目录下所有 .py 文件，分块后返回所有 chunk。"""
    chunks = []
    skip = {"__pycache__", "venv", "tests", "output", "data", ".git", "scripts"}
    for py_file in sorted(code_dir.rglob("*.py")):
        if skip & set(py_file.parts):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(py_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
        chunks.extend(chunk for chunk in chunk_file(rel, content) if chunk.text.strip())
    return chunks


def main() -> int:
    eval_path = PROJECT_ROOT / "data" / "eval_set.json"
    if not eval_path.exists():
        print(f"错误: 找不到评估集 {eval_path}", file=sys.stderr)
        return 1

    print("正在读取评估集 ...")
    try:
        cases = load_cases(eval_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"错误: 评估集读取失败: {exc}", file=sys.stderr)
        return 1
    print(f"  已读取 {len(cases)} 条样本")

    code_dir = PROJECT_ROOT / "codebase_agent"
    print(f"正在切分目录下的代码文件: {code_dir} ...")
    all_chunks = collect_chunks(code_dir)
    print(f"  已生成 {len(all_chunks)} 个 chunk")

    print("正在创建 embedding 提供器 ...")
    try:
        provider = OpenAIEmbeddingProvider()
    except ConfigError as exc:
        print(f"错误: 无法创建 embedding 提供器: {exc}", file=sys.stderr)
        return 1

    print("正在运行评估 ...")
    result = run_evaluation(cases, all_chunks, provider)

    print()
    print("=" * 50)
    print(f"样本总数={result['total']}  命中数={result['hit_count']}")
    print(
        f"Hit@1={result['hit_at_1']:.4f}  "
        f"Hit@K={result['hit_at_k']:.4f}  MRR={result['mrr']:.4f}"
    )
    print(f"失败样本数={len(result['failed_samples'])}")
    print("=" * 50)
    print("\n全部结果:")

    if result["failed_samples"]:
        print("\n失败样本:")
        for i, s in enumerate(result["failed_samples"], 1):
            print(f"  {i}. {s.question[:60]}")
            print(f"     期望文件={s.expected_file}  失败原因={s.failure_reason}")
    print("\n排名结果（从差到好）:")
    ranked_results = sorted(result["results"], key=lambda item: item.reciprocal_rank)
    for i, r in enumerate(ranked_results, 1):
        print(f"  {i}. {r.question[:60]}")
        print(f"     期望文件={r.expected_file}  倒数排名分={r.reciprocal_rank}")
        print(f"     实际前 3 个文件={r.actual_files[:3]}")
    if result["total"] > 0 and result["hit_count"] == 0 and len(result["failed_samples"]) == result["total"]:
        print("\n本次评估全部失败，通常是外部 embedding 服务不可用。")
        print("为避免覆盖上一次可展示的评估快照，本次不写入 data/eval_results.json。")
        return 1

    # 保存 JSON
    out = PROJECT_ROOT / "data" / "eval_results.json"
    out.write_text(
        json.dumps(
            {
                "total": result["total"],
                "hit_count": result["hit_count"],
                "hit_at_1": result["hit_at_1"],
                "hit_at_k": result["hit_at_k"],
                "mrr": result["mrr"],
                "chunk_count": len(all_chunks),
                "embedding_model": provider.model,
                "all_results": [
                    {
                        "question": r.question,
                        "expected_file": r.expected_file,
                        "hit": r.hit,
                        "reciprocal_rank": r.reciprocal_rank,
                        "actual_top_files": r.actual_files[:3],
                    }
                    for r in result["results"]
                ],
                "failed_samples": [
                    {
                        "question": s.question,
                        "expected_file": s.expected_file,
                        "actual_top_files": s.actual_files[:3],
                        "reciprocal_rank": s.reciprocal_rank,
                        "failure_reason": s.failure_reason,
                    }
                    for s in result["failed_samples"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n已保存到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
