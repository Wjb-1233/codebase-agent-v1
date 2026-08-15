"""确定性路由基线 —— 按规则匹配用户意图到对应工具。

当前阶段：可测试的规则路由，不依赖大模型。
后续演进：可替换为 LangGraph 或模型 Tool Calling 路由节点。
"""

from __future__ import annotations

import re

from codebase_agent.agent.models import RouterResult


# 意图模式（按优先级：list_files → get_file_content → search_code）
_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "list_files",
        "用户想列出项目中的文件",
        re.compile(
            r"(有哪些|列出|看看|显示|遍历).*(?:py(?:thon)?\s*)?(?:文件|代码|源码|目录|结构)"
        ),
    ),
    (
        "get_file_content",
        "用户想读取/打开指定文件",
        re.compile(
            r"(?:打开|读取|看看|显示|查看|读).*(?:\.py|backend|main|app|config|database|utils|client)"
        ),
    ),
    (
        "search_code",
        "用户想按键词或语义搜索代码",
        re.compile(
            r"(?:搜索|查找|在哪|哪里|怎么|逻辑|实现|功能|"
            r"认证|登录|注册|数据库|部署|配置|docker|sql|fastapi|pydantic|pytest|测试|"
            r"代码|分析)"
        ),
    ),
]


def route(question: str) -> RouterResult:
    """根据用户问题，按规则匹配返回 RouterResult。

    匹配顺序：空/寒暄 → list_files → get_file_content → search_code → direct_answer。
    """
    if not question or not question.strip():
        return RouterResult(
            tool_name="direct_answer",
            arguments={},
            reason="空问题，无需调用工具",
            is_direct_answer=True,
        )

    question_stripped = question.strip()
    question_lower = question_stripped.lower()

    # 寒暄直接路由
    if re.match(r"^(hi|hello|你好|嗨|早上好|晚上好)\b", question_lower):
        return RouterResult(
            tool_name="direct_answer",
            arguments={"question": question_stripped},
            reason="寒暄信息，无需调用工具",
            is_direct_answer=True,
        )

    # 按优先级匹配工具
    for tool_name, reason_desc, pattern in _PATTERNS:
        if pattern.search(question_lower):
            args = _build_default_args(tool_name, question_stripped)
            return RouterResult(
                tool_name=tool_name,
                arguments=args,
                reason=reason_desc,
            )

    # 默认：直接回答
    return RouterResult(
        tool_name="direct_answer",
        arguments={"question": question_stripped},
        reason="无明确工具意图，走直接回答",
        is_direct_answer=True,
    )


def _build_default_args(tool_name: str, question: str) -> dict[str, object]:
    """为不同工具构建默认参数。"""
    if tool_name == "list_files":
        return {"file_pattern": "*.py"}
    if tool_name == "get_file_content":
        path_match = re.search(r"([\w/.-]+\.py)", question)
        return {"path": path_match.group(1) if path_match else ""}
    if tool_name == "search_code":
        return {"query": question, "top_k": 5}
    return {}
