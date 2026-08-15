"""构造代码库 RAG 问答使用的 prompt。"""

from typing import Protocol, Sequence


class PromptChunk(Protocol):
    """渲染一个检索 chunk 时需要的字段。"""

    text: str
    file_path: str
    start_line: int
    end_line: int


def build_rag_prompt(
    question: str,
    chunks: Sequence[PromptChunk] | None,
) -> str:
    """把用户问题和检索到的 chunk 合成一个带证据约束的 prompt。"""
    if not question.strip():
        raise ValueError("question 不能为空")

    if not chunks:
        context = "没有找到相关代码片段。"
    else:
        parts = [
            f"[来源:{chunk.file_path}:{chunk.start_line}-{chunk.end_line}]\n{chunk.text}"
            for chunk in chunks
        ]
        context = "\n\n".join(parts)

    return f"""你是代码库问答助手，只能根据给出的代码片段回答问题。

=== 代码参考 ===
{context}

=== 用户问题 ===
{question}

如果信息不足，请明确说明不知道，不要编造。
回答时先给结论，再列出引用来源。"""
