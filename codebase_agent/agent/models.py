"""Agent 工具系统的数据模型：统一结果、审计事件、工具定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ToolResult:
    """工具执行的统一结果。

    关键区分：
      success=True + output=[]  → 功能正常但空结果（HTTP 200）
      success=False             → 执行失败（HTTP 4xx/5xx）
    """

    success: bool
    output: object | None = None
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def ok(cls, output: object) -> ToolResult:
        return cls(success=True, output=output)

    @classmethod
    def fail(cls, error_type: str, error_message: str) -> ToolResult:
        return cls(success=False, error_type=error_type, error_message=error_message)


@dataclass(frozen=True)
class ToolEvent:
    """一次工具调用的不可变审计事件。

    脱敏规则：
      - input_summary: 只记录摘要（文件名不记录完整路径，query 截断到 80 字）
      - output_summary: 只记录"返回 N 条结果"/"文本 (N 字符)"
      - 绝不记录：完整文件内容、API key、越权路径原文
    """

    tool_name: str
    input_summary: dict[str, object] = field(default_factory=dict)
    output_summary: str | None = None
    success: bool = True
    error_type: str | None = None
    duration_ms: float = 0.0
    trace_id: str = ""


@dataclass(frozen=True)
class ToolDefinition:
    """工具的接口声明——给路由器/模型看的"使用说明书"。"""

    name: str
    description: str
    parameters: dict[str, dict[str, object]] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    function: Callable[..., object] | None = None


@dataclass(frozen=True)
class RouterResult:
    """路由器的输出：选择了哪个工具、什么参数、为什么这样选。"""

    tool_name: str
    arguments: dict[str, object]
    reason: str = ""
    is_direct_answer: bool = False
