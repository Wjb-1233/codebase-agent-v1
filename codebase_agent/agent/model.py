"""Agent 模型提供器：决策与答案生成。

这里的"模型"指 LLM。runner 不关心底层供应商，只通过
AgentModelProvider 协议调用模型，生产环境默认使用 OpenAI 兼容接口。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from codebase_agent.agent.runner import AgentDecision, AgentModelProvider, AgentState
from codebase_agent.exceptions import ConfigError, LLMError


# ── 真实 OpenAI 工具调用提供器 ──

# OpenAI 工具定义对应的 JSON Schema 类型
_OPENAI_TOOLS_SCHEMA: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出项目目录中匹配通配符的文件。例如 '*.py' 列出所有 Python 文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_pattern": {
                        "type": "string",
                        "description": "文件通配符，默认 '*.py'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_content",
            "description": "读取项目内指定文件的内容。参数 path 是相对于项目根目录的路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于项目根目录的文件路径",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "按语义搜索代码库，返回最相关的代码片段及其相似度分数。适合查找「某个功能在哪实现的」这类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或自然语言描述",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# 给模型看的系统 prompt——告诉它"你是什么、你能做什么、不能做什么"
_SYSTEM_PROMPT = """你是一个代码库分析助手。你可以使用以下工具来探索项目：

- list_files: 列出项目中的文件
- get_file_content: 读取指定文件内容
- search_code: 按语义搜索代码

规则：
1. 如果用户问题可以直接回答（寒暄、常识），不需要调工具。
2. 如果需要查看代码，先用 list_files 或 search_code 了解结构，再用 get_file_content 读具体文件。
3. 工具返回空结果不是错误——告诉用户"没找到"即可。
4. 如果需要读文件，path 必须是项目内的相对路径，不能包含 ../ 或绝对路径。
5. 最多使用 3 步完成分析，给出最终结论。
6. 如果已有工具结果足够回答，不要重复调用相同工具，直接给出中文结论。"""


class OpenAIAgentProvider(AgentModelProvider):
    """基于 OpenAI 对话补全接口的工具调用提供器。

    如果 API key 不存在 → 抛 ConfigError（不是运行时崩溃）。
    测试时用假模型提供器，不碰这个。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        load_dotenv(override=True)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ConfigError("OPENAI_API_KEY 未配置，无法使用 Agent 模型提供器")
        self._model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def _client(self):
        """延迟导入——不在模块加载时引入 openai 依赖。"""
        from openai import OpenAI  # noqa: PLC0415
        client_kwargs = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        return OpenAI(**client_kwargs)

    def decide(self, state: AgentState) -> AgentDecision:
        """调用 OpenAI 对话补全接口，让模型决定下一步：调用工具或直接回答。"""
        client = self._client()

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_decision_prompt(state)},
                ],
                tools=_OPENAI_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as exc:
            raise LLMError(f"Agent 模型调用失败: {exc}") from exc

        choice = response.choices[0]
        message = choice.message

        # 模型选择调工具
        if message.tool_calls and len(message.tool_calls) > 0:
            tool_call = message.tool_calls[0]
            import json

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            return AgentDecision(
                is_direct_answer=False,
                tool_name=tool_call.function.name,
                arguments=arguments,
            )

        # 模型直接回答（不调工具）
        return AgentDecision(
            is_direct_answer=True,
            content=message.content or "",
        )

    def answer(self, state: AgentState) -> str:
        """基于工具执行结果，让模型生成最终答案。"""
        client = self._client()

        # 构建消息：system + 用户原始问题 + 每步工具结果
        messages: list[dict[str, object]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": state.question},
        ]

        for i, (tc, tr) in enumerate(zip(state.tool_calls, state.tool_results)):
            tool_result_text = (
                f"成功: {tr.output}" if tr.success
                else f"失败 ({tr.error_type}): {tr.error_message}"
            )
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": str(tc.arguments),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": str(tool_result_text),
            })

        messages.append({
            "role": "user",
            "content": "现在不要再调用工具，也不要输出工具调用标记。请只根据已有工具结果，用中文给出最终代码分析结论。",
        })

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:
            raise LLMError(f"Agent 答案生成失败: {exc}") from exc

        content = response.choices[0].message.content
        return content or "（模型未返回有效答案）"


def _build_decision_prompt(state: AgentState) -> str:
    """把用户问题和已有工具结果放进下一步决策上下文。"""
    lines = [f"用户问题：{state.question}"]
    if state.memory_context:
        lines.append(f"短期记忆：\n{state.memory_context}")
    if state.tool_calls:
        lines.append("已有工具结果：")
        for index, (tool_call, tool_result) in enumerate(
            zip(state.tool_calls, state.tool_results),
            start=1,
        ):
            summary = _summarize_tool_output(tool_result.output)
            status = "成功" if tool_result.success else f"失败：{tool_result.error_message}"
            lines.append(
                f"{index}. {tool_call.tool_name}({tool_call.arguments}) -> {status}；结果摘要：{summary}"
            )
        lines.append("如果这些结果已经足够回答，请直接回答，不要重复调用相同工具。")
    return "\n".join(lines)


def _summarize_tool_output(output: object) -> str:
    if isinstance(output, list):
        items = output[:3]
        previews = []
        for item in items:
            metadata = getattr(item, "metadata", {})
            file_path = metadata.get("file_path", "") if isinstance(metadata, dict) else ""
            text = getattr(item, "text", str(item))
            previews.append(f"{file_path}: {' '.join(str(text).split())[:120]}")
        return " | ".join(previews) if previews else "空列表"
    return " ".join(str(output).split())[:300]
