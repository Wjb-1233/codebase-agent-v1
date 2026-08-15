"""统一工具执行器：白名单注册表、参数校验、安全边界、事件脱敏。"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

from codebase_agent.agent.models import ToolDefinition, ToolEvent, ToolResult


# ── 注册表 ──
_registry: dict[str, ToolDefinition] = {}


def register(tool: ToolDefinition) -> None:
    """向注册表登记一个可用工具。"""
    _registry[tool.name] = tool


def get_registry() -> dict[str, ToolDefinition]:
    """返回当前注册表副本（只读）。"""
    return dict(_registry)


# ── 参数校验 ──
def validate_args(tool: ToolDefinition, arguments: dict[str, object]) -> list[str]:
    """校验参数：必填字段、类型、未知参数。"""
    errors: list[str] = []

    # 1. 必填参数检查
    for param_name in tool.required_params:
        if param_name not in arguments or arguments[param_name] is None:
            errors.append(f"缺少必填参数: {param_name}")
            continue
        param_schema = tool.parameters.get(param_name, {})
        expected_type = param_schema.get("type", "")
        value = arguments[param_name]

        if expected_type == "str" and not isinstance(value, str):
            errors.append(f"参数 {param_name} 应为字符串，实际为 {type(value).__name__}")
        elif expected_type == "int" and not isinstance(value, int):
            errors.append(f"参数 {param_name} 应为整数，实际为 {type(value).__name__}")
        elif expected_type == "list" and not isinstance(value, list):
            errors.append(f"参数 {param_name} 应为列表，实际为 {type(value).__name__}")

    # 2. 未知参数检查
    for key in arguments:
        if key not in tool.parameters:
            errors.append(f"未知参数: {key}")

    return errors


# ── 安全边界 ──
def check_path_safety(path: str, project_root: str) -> tuple[bool, str]:
    """检查路径是否在项目根目录内。

    返回 (安全: bool, 脱敏路径描述: str)。
    规范解析后比对前缀——不是简单的字符串 ".." 检查。
    """
    root = Path(project_root).resolve()
    try:
        full = (root / path).resolve()
    except (ValueError, OSError):
        return False, "[REDACTED: invalid path]"

    try:
        full.relative_to(root)
    except ValueError:
        return False, "[REDACTED: path_traversal]"

    return True, path


# ── 事件摘要（脱敏）──
def _summarize_input(arguments: dict[str, object]) -> dict[str, object]:
    """生成脱敏的参数摘要。"""
    summary: dict[str, object] = {}
    for key, value in arguments.items():
        if key == "path" and isinstance(value, str):
            summary[key] = value.rsplit("/", 1)[-1] if "/" in value else value
        elif key == "query" and isinstance(value, str):
            summary[key] = value[:80]
        elif isinstance(value, (str, int, float, bool)):
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = f"[{len(value)} items]"
        else:
            summary[key] = type(value).__name__
    return summary


def _summarize_output(output: object) -> str | None:
    """生成脱敏的输出摘要。"""
    if output is None:
        return "无输出"
    if isinstance(output, list):
        return f"返回 {len(output)} 条结果"
    if isinstance(output, str):
        size = len(output)
        if size <= 120:
            return output
        return f"文本 ({size} 字符)"
    return type(output).__name__


# ── dispatch（核心）──
def dispatch(
    tool_name: str,
    arguments: dict[str, object],
    project_root: str,
    trace_id: str | None = None,
    **deps: object,
) -> tuple[ToolResult, ToolEvent]:
    """统一工具调度入口。返回 (ToolResult, ToolEvent) 对。

    流程：白名单 → 参数校验 → 安全边界 → 执行 + 计时 → 结果 + 事件。
    deps 用于注入项目级依赖（chunks, embedding_provider 等）。
    """
    start_time = time.time()
    tid = trace_id or uuid.uuid4().hex[:12]

    def _fail_event(error_type: str, error_message: str = "") -> ToolEvent:
        return ToolEvent(
            tool_name=tool_name,
            input_summary=_summarize_input(arguments),
            success=False,
            error_type=error_type,
            duration_ms=(time.time() - start_time) * 1000,
            trace_id=tid,
        )

    # 1. 白名单
    if tool_name not in _registry:
        event = _fail_event("unknown_tool", f"未知工具: {tool_name}")
        return ToolResult.fail("unknown_tool", f"未知工具: {tool_name}"), event

    tool_def = _registry[tool_name]

    # 2. 参数校验
    validation_errors = validate_args(tool_def, arguments)
    if validation_errors:
        msg = "; ".join(validation_errors)
        event = _fail_event("invalid_argument", msg)
        return ToolResult.fail("invalid_argument", msg), event

    # 3. 安全边界：路径校验
    if "path" in arguments and isinstance(arguments["path"], str):
        safe, _ = check_path_safety(str(arguments["path"]), project_root)
        if not safe:
            event = _fail_event("permission_denied")
            return ToolResult.fail("permission_denied", "路径越权——拒绝访问"), event

    # 4. 执行
    fn = tool_def.function
    if fn is None:
        event = _fail_event("tool_not_configured")
        return ToolResult.fail("tool_not_configured", f"工具 {tool_name} 未绑定实现"), event

    # 合并参数：arguments + project_root（仅对需要的工具注入）+ deps
    merged_args = dict(arguments)
    if tool_def.name in ("list_files", "get_file_content"):
        merged_args["project_root"] = project_root
    for dep_name, dep_value in deps.items():
        if dep_name not in merged_args:
            merged_args[dep_name] = dep_value

    try:
        output = fn(**merged_args)
        elapsed = (time.time() - start_time) * 1000
        event = ToolEvent(
            tool_name=tool_name,
            input_summary=_summarize_input(arguments),
            output_summary=_summarize_output(output),
            success=True,
            duration_ms=elapsed,
            trace_id=tid,
        )
        return ToolResult.ok(output), event
    except Exception as exc:
        elapsed = (time.time() - start_time) * 1000
        error_type = type(exc).__name__
        event = ToolEvent(
            tool_name=tool_name,
            input_summary=_summarize_input(arguments),
            success=False,
            error_type=error_type,
            duration_ms=elapsed,
            trace_id=tid,
        )
        return ToolResult.fail(error_type, str(exc)), event


# ── 启动时注册 ──
def _bootstrap_registry() -> None:
    from codebase_agent.agent.tools import get_file_content, list_files, search_code

    _registry.clear()

    register(
        ToolDefinition(
            name="list_files",
            description="列出项目目录中匹配通配符的文件。例如 '*.py' 列出所有 Python 文件。",
            parameters={"file_pattern": {"type": "str"}},
            required_params=[],
            function=list_files,
        )
    )

    register(
        ToolDefinition(
            name="get_file_content",
            description="读取项目内指定文件的内容。参数 path 是相对于项目根目录的路径。",
            parameters={"path": {"type": "str"}},
            required_params=["path"],
            function=get_file_content,
        )
    )

    register(
        ToolDefinition(
            name="search_code",
            description="按语义搜索代码库，返回最相关的代码片段及其相似度分数。",
            parameters={
                "query": {"type": "str"},
                "top_k": {"type": "int"},
            },
            required_params=["query"],
            function=search_code,
        )
    )


_bootstrap_registry()
