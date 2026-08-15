"""短期记忆单元测试。

覆盖：最近 N 轮 / 空 history / 空内容过滤 / 超长截断 / role 校验 / 边界。
"""

from __future__ import annotations

import pytest

from codebase_agent.agent.memory import (
    ConversationTurn,
    MemoryContext,
    build_short_term_context,
)


# ═══════════════════════ ConversationTurn ═══════════════════════

class TestConversationTurn:
    def test_user_role(self):
        t = ConversationTurn(role="user", content="你好")
        assert t.role == "user"
        assert t.content == "你好"

    def test_assistant_role(self):
        t = ConversationTurn(role="assistant", content="你好！有什么可以帮你")
        assert t.role == "assistant"

    def test_system_role(self):
        t = ConversationTurn(role="system", content="你是一个助手")
        assert t.role == "system"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role 只能是"):
            ConversationTurn(role="admin", content="...")

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="role 不能为空"):
            ConversationTurn(role="", content="...")

    def test_empty_content_allowed_for_storage(self):
        """空 content 在构造时不报错，但在 build_short_term_context 中会被过滤。"""
        t = ConversationTurn(role="user", content="")
        assert t.content == ""


# ═══════════════════════ 最近 N 轮 ═══════════════════════

def test_keeps_last_n_turns():
    history = [
        ConversationTurn(role="user", content="Q1"),
        ConversationTurn(role="assistant", content="A1"),
        ConversationTurn(role="user", content="Q2"),
        ConversationTurn(role="assistant", content="A2"),
        ConversationTurn(role="user", content="Q3"),
        ConversationTurn(role="assistant", content="A3"),
    ]
    ctx = build_short_term_context(history, max_turns=4)

    assert ctx.memory_used is True
    assert ctx.memory_turns == 4
    assert "Q3" in ctx.formatted
    assert "A3" in ctx.formatted
    assert "Q1" not in ctx.formatted  # 被丢弃的最旧轮次

    print(f"\n[最近 4 轮]\n{ctx.formatted}")


def test_max_turns_zero_returns_empty():
    history = [ConversationTurn(role="user", content="hello")]
    ctx = build_short_term_context(history, max_turns=0)

    assert ctx.memory_used is False
    assert ctx.memory_turns == 0
    assert ctx.formatted == ""


# ═══════════════════════ 空 history ═══════════════════════

def test_none_history():
    ctx = build_short_term_context(None)
    assert ctx.memory_used is False
    assert ctx.memory_turns == 0
    assert ctx.formatted == ""


def test_empty_list_history():
    ctx = build_short_term_context([])
    assert ctx.memory_used is False
    assert ctx.memory_turns == 0
    assert ctx.formatted == ""


# ═══════════════════════ 空内容过滤 ═══════════════════════

def test_filters_empty_content():
    history = [
        ConversationTurn(role="user", content="Q1"),
        ConversationTurn(role="assistant", content=""),
        ConversationTurn(role="user", content=""),
        ConversationTurn(role="assistant", content="A1"),
    ]
    ctx = build_short_term_context(history, max_turns=10)

    assert ctx.memory_used is True
    assert ctx.memory_turns == 2
    assert "Q1" in ctx.formatted
    assert "A1" in ctx.formatted


def test_all_content_empty_returns_false():
    history = [
        ConversationTurn(role="user", content=""),
        ConversationTurn(role="assistant", content=""),
    ]
    ctx = build_short_term_context(history, max_turns=10)

    assert ctx.memory_used is False
    assert ctx.memory_turns == 0


# ═══════════════════════ 超长截断 ═══════════════════════

def test_total_chars_exceeded_drops_oldest():
    long_content = "A" * 1500
    history = [
        ConversationTurn(role="user", content="Q1 - old"),
        ConversationTurn(role="assistant", content="A1 - old"),
        ConversationTurn(role="user", content=long_content),
        ConversationTurn(role="assistant", content="A2 - recent"),
    ]
    # max_chars 只够放最后两条 → 最旧两条被丢弃
    ctx = build_short_term_context(history, max_turns=10, max_chars=100)

    assert ctx.memory_used is True
    assert "Q1 - old" not in ctx.formatted
    assert "A1 - old" not in ctx.formatted
    # long_content 被截断到 max_chars，只剩前缀部分
    assert "A" * 90 in ctx.formatted  # 截断后保留约 90 个 A（扣除 "[user]: " 前缀）
    assert "A2 - recent" in ctx.formatted
    assert ctx.truncated is True
    assert ctx.memory_turns == 2


def test_single_turn_exceeds_max_chars_head_truncated():
    """单条 content 超过 max_chars → 从头部截断该条。"""
    long_content = "X" * 300
    history = [ConversationTurn(role="user", content=long_content)]

    ctx = build_short_term_context(history, max_turns=10, max_chars=250)

    assert ctx.memory_used is True
    assert ctx.memory_turns == 1
    assert ctx.truncated is True
    assert len(ctx.formatted) <= 250
    print(f"\n[{len(ctx.formatted)} chars] {ctx.formatted[:100]}...")


# ═══════════════════════ 长度刚好 ═══════════════════════

def test_exact_fit_no_truncation():
    history = [
        ConversationTurn(role="user", content="short"),
        ConversationTurn(role="assistant", content="reply"),
    ]
    ctx = build_short_term_context(history, max_turns=10, max_chars=99999)

    assert ctx.memory_used is True
    assert ctx.memory_turns == 2
    assert ctx.truncated is False


# ═══════════════════════ formatted 格式 ═══════════════════════

def test_formatted_format():
    history = [
        ConversationTurn(role="user", content="数据库在哪"),
        ConversationTurn(role="assistant", content="在 database.py 中"),
    ]
    ctx = build_short_term_context(history)

    expected = "[user]: 数据库在哪\n[assistant]: 在 database.py 中"
    assert ctx.formatted == expected
