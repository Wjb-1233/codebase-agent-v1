from __future__ import annotations

from collections.abc import Iterator
from typing import Sequence

from codebase_agent.agent.runner import AgentDecision, AgentModelProvider, AgentState


class FakeLLMProvider:
    """测试使用的确定性 LLM provider。"""

    def __init__(self, fixed_response: str = "基于检索代码片段生成的模拟回答") -> None:
        self.fixed_response = fixed_response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.fixed_response

    def stream(self, prompt: str) -> Iterator[str]:
        self.last_prompt = prompt
        for index in range(0, len(self.fixed_response), 12):
            yield self.fixed_response[index:index + 12]


class FakeAgentModelProvider(AgentModelProvider):
    """按预设列表返回决策，覆盖 Agent 循环分支。"""

    def __init__(
        self,
        decisions: Sequence[AgentDecision] | None = None,
        final_answer: str = "基于工具执行结果的分析。",
    ) -> None:
        self._decisions = list(decisions or [])
        self._index = 0
        self.final_answer = final_answer

    def decide(self, state: AgentState) -> AgentDecision:
        if self._index < len(self._decisions):
            decision = self._decisions[self._index]
            self._index += 1
            return decision
        return AgentDecision(is_direct_answer=True, content=self.final_answer)

    def answer(self, state: AgentState) -> str:
        return self.final_answer
