"""短期记忆：滑动窗口 + 长度截断，把最近 N 轮对话做成可注入的上下文。

今天只做滑动窗口——不碰长期记忆、向量记忆、用户画像。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ── 数据模型 ──

class ConversationTurn:
    """一轮对话：role + content。

    role 必须是 "user"、"assistant" 或 "system"。
    content 不能为空字符串，构建上下文时会自动过滤。
    """

    __slots__ = ("role", "content")

    def __init__(self, *, role: str, content: str) -> None:
        if not role:
            raise ValueError("role 不能为空")
        if not isinstance(role, str):
            raise TypeError(f"role 应为 str，实际为 {type(role).__name__}")
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"role 只能是 user/assistant/system，得到: {role}")
        self.role: str = role
        self.content: str = content

    def __repr__(self) -> str:
        return f"ConversationTurn(role={self.role!r}, content={self.content[:40]!r}...)"


@dataclass(frozen=True)
class MemoryContext:
    """短期记忆构造结果——可直接注入 prompt 或放进 AgentState。"""

    formatted: str                       # 格式化后的多轮对话文本
    memory_used: bool                    # 是否实际使用了历史（history 非空）
    memory_turns: int                    # 实际保留的轮数
    truncated: bool = False              # 是否因字符上限截断


# ── 核心函数 ──

def build_short_term_context(
    history: Sequence[ConversationTurn] | None,
    *,
    max_turns: int = 10,
    max_chars: int = 4000,
) -> MemoryContext:
    """把对话历史压缩成短期记忆上下文。

    Args:
        history: 全部对话轮次（按时间升序排列）。
        max_turns: 最多保留最近多少轮。
        max_chars: 格式化后文本的总字符数上限。

    处理流程：
        1. 取最近 max_turns 轮
        2. 过滤掉 content 为空的轮次
        3. 从旧到新拼接为 "[role]: content\\n" 格式
        4. 如果总字符数超过 max_chars，从最旧一条截断

    返回：
        MemoryContext: formatted 可直接注入 prompt 的文本片段。

    边界场景：
        - history 为 None 或空列表 → memory_used=False, turns=0
        - 所有 content 都为空 → 同上
        - 单条 content 就超过 max_chars → 从该条头部截断
        - max_turns=0 → 不保留任何历史
    """
    if not history or max_turns <= 0:
        return MemoryContext(formatted="", memory_used=False, memory_turns=0)

    # 1. 取最近 N 轮
    recent = list(history[-max_turns:])

    # 2. 过滤空内容
    filtered = [turn for turn in recent if turn.content]

    if not filtered:
        return MemoryContext(formatted="", memory_used=False, memory_turns=0)

    # 3. 拼接
    lines: list[str] = []
    for turn in filtered:
        lines.append(f"[{turn.role}]: {turn.content}")

    # 4. 长度截断：从最旧一条开始删，直到不超过 max_chars
    truncated = False
    while lines:
        total = sum(len(line) + 1 for line in lines)  # +1 for newline
        if total <= max_chars:
            break
        # 最旧一条过长 → 截断它
        if len(lines[0]) > max_chars:
            lines[0] = lines[0][:max_chars]
            truncated = True
            break
        # 否则直接删除最旧一条
        lines.pop(0)
        truncated = True

    formatted = "\n".join(lines)

    return MemoryContext(
        formatted=formatted,
        memory_used=True,
        memory_turns=len(lines),
        truncated=truncated,
    )
