"""Agent 运行器：状态管理、工具循环、失败处理。

这是 Agent 的核心编排层。它不自己调用模型、不自己执行工具——
它只负责：持有 state → 问模型"下一步做什么" → 把决策交给 executor →
把结果写回 state → 循环 → 直到有答案或超步数。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from codebase_agent.agent.executor import dispatch
from codebase_agent.agent.memory import ConversationTurn, build_short_term_context
from codebase_agent.agent.tool_contracts import ToolEvent, ToolResult
from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.embeddings import EmbeddingProvider
from codebase_agent.rag.vector_store import SearchResult


# ── 数据模型 ──

@dataclass
class AgentToolCall:
    """模型决策的一次工具调用记录。"""
    tool_name: str
    arguments: dict[str, object]
    reason: str = ""


@dataclass
class AgentDecision:
    """模型一次决策的结果：要么直接回答，要么调用工具。"""
    is_direct_answer: bool
    content: str | None = None          # 直接回答时使用
    tool_name: str | None = None        # 工具调用时使用
    arguments: dict[str, object] | None = None  # 工具调用时使用


class AgentModelProvider:
    """模型决策的抽象边界。

    在 runner 眼中，模型就是两个方法：
      decide(state) → 下一步是调工具还是直接回答
      answer(state) → 基于已有工具结果生成最终答案
    """

    def decide(self, state: AgentState) -> AgentDecision:
        raise NotImplementedError

    def answer(self, state: AgentState) -> str:
        raise NotImplementedError


@dataclass
class AgentState:
    """一次 Agent 运行中所有节点共享的数据。

    它是 runner 循环中的唯一数据载体。
    每一步只读 state、只写 state，不依赖全局变量。
    """

    question: str
    answer: str = ""
    status: str = "running"  # running → completed / partial / failed
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    events: list[ToolEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    step_count: int = 0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    memory_context: str = ""    # 短期记忆上下文（格式化文本，注入 prompt）
    memory_used: bool = False   # 是否使用了短期记忆
    memory_turns: int = 0       # 保留了多少轮历史


@dataclass
class AgentRunResult:
    """run_agent 的返回值——API 层直接映射到 AgentRunResponse。"""
    question: str
    answer: str
    status: str
    tool_calls: list[AgentToolCall]
    events: list[ToolEvent]
    errors: list[str]
    trace_id: str
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    memory_used: bool = False
    memory_turns: int = 0


# ── 核心运行循环 ──

def run_agent(
    question: str,
    project_root: str,
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    model_provider: AgentModelProvider,
    max_steps: int = 3,
    history: list[ConversationTurn] | None = None,
    max_memory_turns: int = 3,
    max_memory_chars: int = 4000,
) -> AgentRunResult:
    """完整 Agent 运行循环。

    ┌─ while step_count < max_steps ──────────────────────┐
    │  1. model_provider.decide(state) → 拿到决策          │
    │  2. 如果是直接回答 → 写入 state.answer → 跳出循环      │
    │  3. 否则 → executor.dispatch() 执行工具               │
    │  4. ToolResult + ToolEvent 写入 state                │
    │  5. 如果工具是 search_code，提取检索结果为 retrieved_chunks │
    │  6. 失败 → 写入 state.errors                         │
    │  7. step_count += 1                                  │
    └─────────────────────────────────────────────────────┘
    循环结束后：
      - 超步数 → status="partial"，但仍尝试生成答案
      - 有答案 → status="completed"
    """

    # ── 短期记忆：构建上下文 ──
    memory_ctx = build_short_term_context(
        history, max_turns=max_memory_turns, max_chars=max_memory_chars
    )

    state = AgentState(
        question=question,
        memory_context=memory_ctx.formatted,
        memory_used=memory_ctx.memory_used,
        memory_turns=memory_ctx.memory_turns,
    )

    # ── 工具循环 ──
    while state.step_count < max_steps:
        decision = model_provider.decide(state)

        # 模型决定直接回答
        if decision.is_direct_answer:
            state.answer = decision.content or ""
            state.status = "completed"
            break

        # 模型决定调工具
        tool_name = decision.tool_name or ""
        arguments = decision.arguments or {}
        call_signature = (tool_name, tuple(sorted((str(k), str(v)) for k, v in arguments.items())))
        previous_signatures = {
            (call.tool_name, tuple(sorted((str(k), str(v)) for k, v in call.arguments.items())))
            for call in state.tool_calls
        }
        if call_signature in previous_signatures:
            state.status = "partial"
            state.errors.append("repeated_tool_call_blocked")
            break

        state.tool_calls.append(
            AgentToolCall(
                tool_name=tool_name,
                arguments=dict(arguments),
                reason=f"step {state.step_count + 1}",
            )
        )

        result, event = dispatch(
            tool_name=tool_name,
            arguments=dict(arguments),
            project_root=project_root,
            trace_id=state.trace_id,
            chunks=chunks,
            embedding_provider=embedding_provider,
        )

        state.tool_results.append(result)
        state.events.append(event)

        if not result.success:
            error_msg = result.error_type or "unknown_error"
            detail = result.error_message or ""
            state.errors.append(f"{error_msg}: {detail}" if detail else error_msg)

        state.step_count += 1

    # ── 抽取 search_code 的检索结果 ──
    retrieved_chunks: list[SearchResult] = []
    for tc, tr in zip(state.tool_calls, state.tool_results):
        if tc.tool_name == "search_code" and tr.success:
            # search_code 的 output 是 list[SearchResult]
            if isinstance(tr.output, list):
                for item in tr.output:
                    if isinstance(item, SearchResult):
                        retrieved_chunks.append(item)

    # ── 循环后处理 ──
    # 情况 A：直接回答（已在循环中处理，status=completed）
    # 情况 B：超步数但尚未有答案
    if state.step_count >= max_steps and not state.answer:
        state.status = "partial"
        state.errors.append("max_steps_exceeded")
        if retrieved_chunks:
            state.answer = _build_partial_search_answer(retrieved_chunks)

    # 情况 C：正常结束（模型在某步决定直接回答）——不需要额外处理

    # 如果还没有答案，让模型基于工具结果生成
    if not state.answer:
        try:
            state.answer = model_provider.answer(state)
            if state.status != "partial":
                state.status = "completed"
        except Exception as exc:
            state.answer = ""
            state.errors.append(f"answer_generation_failed: {exc}")
            if state.status != "partial":
                state.status = "failed"

    return AgentRunResult(
        question=state.question,
        answer=state.answer,
        status=state.status,
        tool_calls=state.tool_calls,
        events=state.events,
        errors=state.errors,
        trace_id=state.trace_id,
        retrieved_chunks=retrieved_chunks,
        memory_used=state.memory_used,
        memory_turns=state.memory_turns,
    )


def _build_partial_search_answer(retrieved_chunks: list[SearchResult]) -> str:
    """真实模型未收束时，基于已有检索证据给出稳定可读的降级答案。"""
    top_items = retrieved_chunks[:3]
    lines = ["模型已达到最大工具步数，以下是基于已检索证据的部分结论："]
    for index, item in enumerate(top_items, start=1):
        metadata = item.metadata
        file_path = metadata.get("file_path", "unknown")
        start_line = metadata.get("start_line", 0)
        end_line = metadata.get("end_line", 0)
        preview = " ".join(item.text.strip().split())[:160]
        lines.append(
            f"{index}. {file_path}:{start_line}-{end_line}，相关分数 {item.score:.3f}，代码片段：{preview}"
        )
    return "\n".join(lines)
