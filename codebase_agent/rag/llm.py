"""RAG 答案生成层使用的 LLM 提供器。"""

from __future__ import annotations

import os
from typing import Callable, Iterator, Protocol

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from codebase_agent.exceptions import ConfigError, LLMError


DEFAULT_CHAT_MODEL = "gpt-4o-mini"


class LLMProvider(Protocol):
    """普通回答和流式回答共用的 LLM 边界接口。"""

    def generate(self, prompt: str) -> str:
        """根据 prompt 返回完整文本回答。"""
        ...

    def stream(self, prompt: str) -> Iterator[str]:
        """为流式客户端逐段返回回答。"""
        ...


class OpenAILLMProvider:
    """基于 OpenAI 的 LLM 提供器实现。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        client: object | None = None,
        client_factory: Callable[..., object] | None = None,
    ) -> None:
        load_dotenv(override=True)
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ConfigError("未配置 OPENAI_API_KEY")

        factory = client_factory or OpenAI
        client_kwargs = {"api_key": resolved_api_key}
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = factory(**client_kwargs)

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
        except AuthenticationError as e:
            raise ConfigError("OpenAI API Key 无效") from e
        except (RateLimitError, APITimeoutError, APIError) as e:
            raise LLMError("LLM 调用失败") from e

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError("LLM 返回结构无效") from e
        if not content or not content.strip():
            raise LLMError("LLM 返回了空响应")
        return content.strip()

    def stream(self, prompt: str) -> Iterator[str]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
                stream=True,
            )
        except AuthenticationError as e:
            raise ConfigError("OpenAI API Key 无效") from e
        except (RateLimitError, APITimeoutError, APIError) as e:
            raise LLMError("LLM 流式调用失败") from e

        yielded = False
        for chunk in response:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError, TypeError) as e:
                raise LLMError("LLM 流式返回结构无效") from e
            if delta:
                yielded = True
                yield str(delta)
        if not yielded:
            raise LLMError("LLM 流式返回了空响应")


class FakeLLMProvider:
    """测试和本地示例使用的确定性提供器。"""

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
