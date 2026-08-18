"""Agent 工具层 —— 可测试的工具执行底座。

导入所有公开接口，方便外部 `from codebase_agent.agent import ToolResult, ...`。
"""

from codebase_agent.agent.tool_contracts import RouterResult, ToolDefinition, ToolEvent, ToolResult
from codebase_agent.agent.tools import get_file_content, list_files, search_code
from codebase_agent.agent.executor import dispatch, get_registry
from codebase_agent.agent.router import route
from codebase_agent.agent.runner import (
    AgentDecision,
    AgentModelProvider,
    AgentRunResult,
    AgentState,
    AgentToolCall,
    run_agent,
)

__all__ = [
    "ToolResult",
    "ToolEvent",
    "ToolDefinition",
    "RouterResult",
    "list_files",
    "get_file_content",
    "search_code",
    "dispatch",
    "get_registry",
    "route",
    "AgentState",
    "AgentToolCall",
    "AgentDecision",
    "AgentModelProvider",
    "AgentRunResult",
    "run_agent",
]
