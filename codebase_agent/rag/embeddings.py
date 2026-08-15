"""RAG 检索层使用的 embedding 提供器。"""

from __future__ import annotations

import os
from typing import Callable, Protocol, Sequence

from dotenv import load_dotenv

from codebase_agent.exceptions import ConfigError, EmbeddingError


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingProvider(Protocol):
    """把文本转换为向量的边界接口。"""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """为每段输入文本返回一个 embedding 向量。"""


class OpenAIEmbeddingProvider:
    """基于 OpenAI 的 embedding 提供器。

    RAG 其他模块只依赖这个提供器边界，不直接引入 OpenAI SDK。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        client: object | None = None,
        client_factory: Callable[..., object] | None = None,
    ) -> None:
        load_dotenv(override=True)
        self.model = (
            model
            or os.getenv("EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ConfigError("使用外部 Embeddings 必须配置 EMBEDDING_API_KEY 或 OPENAI_API_KEY")

        if client_factory is None:
            from openai import OpenAI

            client_factory = OpenAI

        client_kwargs = {"api_key": resolved_api_key}
        resolved_base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = client_factory(**client_kwargs)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        clean_texts = _validate_texts(texts)
        if not clean_texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_texts,
            )
        except Exception as exc:  # pragma: no cover - exact SDK exceptions vary.
            raise EmbeddingError("OpenAI Embeddings 请求失败") from exc

        data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        return [list(item.embedding) for item in data]


class SentenceTransformerEmbeddingProvider:
    """基于本地 sentence-transformers 的 embedding 提供器。"""

    def __init__(
        self,
        model_name: str | None = None,
        model_instance: object | None = None,
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        load_dotenv(override=True)
        self.model_name = model_name or os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            DEFAULT_LOCAL_EMBEDDING_MODEL,
        )

        if model_instance is not None:
            self.model = model_instance
            return

        if model_factory is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ConfigError(
                    "未安装 sentence-transformers，无法使用本地 embedding"
                ) from exc
            model_factory = SentenceTransformer

        self.model = model_factory(self.model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        clean_texts = _validate_texts(texts)
        if not clean_texts:
            return []

        try:
            embeddings = self.model.encode(
                clean_texts,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError("本地 Embeddings 请求失败") from exc

        return [_to_float_vector(vector) for vector in embeddings]


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        raise ValueError("texts 必须是字符串序列，不能直接传入单个字符串")

    clean_texts = list(texts)
    for index, text in enumerate(clean_texts):
        if not isinstance(text, str):
            raise ValueError(f"texts[{index}] 必须是字符串")
        if not text.strip():
            raise ValueError(f"texts[{index}] 不能为空")
    return clean_texts


def _to_float_vector(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
