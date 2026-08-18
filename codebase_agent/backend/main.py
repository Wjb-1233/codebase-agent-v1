import hashlib
import json
import logging
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from codebase_agent.backend.database import init_db, list_recent_analyses, save_analysis
from codebase_agent.backend.github_client import GitHubClient
from codebase_agent.code_graph import CodeGraphInputFile, build_code_graph
from codebase_agent.exceptions import (
    ConfigError,
    EmbeddingError,
    GitHubAPIError,
    LLMError,
    NetworkError,
    RateLimitError,
)
from codebase_agent.rag.chunker import Chunk, chunk_file
from codebase_agent.rag.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from codebase_agent.rag.llm import LLMProvider, OpenAILLMProvider
from codebase_agent.rag.prompt_builder import build_rag_prompt
from codebase_agent.rag.vector_store import VectorStoreProtocol, search_code
from codebase_agent.rag.keyword_search import KeywordIndex
from codebase_agent.rag.reranker import CrossEncoderReranker, IdentityReranker
from codebase_agent.rag.parent_document import attach_parent_metadata, build_parent_map, expand_with_parents
from codebase_agent.rag.qdrant_store import QdrantVectorStore, VectorStoreError
from codebase_agent.rag.evaluator import (
    GenerationEvalCase,
    evaluate_generation,
    evaluate_generation_with_llm_judge,
)
from codebase_agent.agent.memory_store import AgentMemoryStore
from codebase_agent.agent.runner import AgentModelProvider, AgentRunResult, AgentToolCall, run_agent as _run_agent
from codebase_agent.agent.model import OpenAIAgentProvider

app = FastAPI()


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_max_files() -> int:
    return 500


def get_db_path() -> str:
    return os.getenv("DATABASE_PATH", "analysis.db")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logging.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({elapsed:.2f}s)"
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class AnalyzeRequest(BaseModel):
    repo_url: str


class AnalyzeResponse(BaseModel):
    files: list[str]
    count: int


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    max_files: int = Depends(get_max_files),
    db_path: str = Depends(get_db_path),
) -> AnalyzeResponse:
    if request.repo_url is None or not request.repo_url.strip():
        raise HTTPException(status_code=400, detail="repo_url 不能为空")
    try:
        async with GitHubClient(request.repo_url) as client:
            files = await client.get_file_tree()
    except NetworkError:
        raise HTTPException(502, detail="GitHub API 不可用")
    except GitHubAPIError:
        raise HTTPException(502, detail="GitHub API 返回错误")
    except RateLimitError:
        raise HTTPException(502, detail="GitHub API 限流，稍后重试")
    except ConfigError:
        raise HTTPException(400, detail="配置错误")
    if len(files) > max_files:
        raise HTTPException(status_code=422, detail="文件太多")
    init_db(db_path)
    save_analysis(db_path, request.repo_url, len(files))
    return AnalyzeResponse(files=files, count=len(files))


@app.get("/history")
def history(limit: int = 20, db_path: str = Depends(get_db_path)):
    init_db(db_path)
    return list_recent_analyses(db_path, limit)


class SearchFileInput(BaseModel):
    file_path: str
    content: str


class CodeGraphRequest(BaseModel):
    files: list[SearchFileInput]


class CodeGraphNodeItem(BaseModel):
    id: str
    label: str
    type: str
    file_path: str = ""
    line: int = 0


class CodeGraphEdgeItem(BaseModel):
    source: str
    target: str
    type: str
    label: str = ""


class CodeGraphImportItem(BaseModel):
    file_path: str
    module: str
    line: int
    target_file: str = ""
    is_internal: bool = False


class CodeGraphFileItem(BaseModel):
    file_path: str
    language: str
    symbols: list[CodeGraphNodeItem]
    imports: list[CodeGraphImportItem]
    parse_error: str = ""


class CodeGraphResponse(BaseModel):
    nodes: list[CodeGraphNodeItem]
    edges: list[CodeGraphEdgeItem]
    files: list[CodeGraphFileItem]
    summary: dict[str, int]


@app.post("/code-graph", response_model=CodeGraphResponse)
def code_graph(request: CodeGraphRequest) -> CodeGraphResponse:
    if not request.files:
        raise HTTPException(status_code=400, detail="请提供要分析的代码文件")

    graph = build_code_graph(
        [
            CodeGraphInputFile(file_path=file.file_path, content=file.content)
            for file in request.files
        ]
    )
    return CodeGraphResponse(
        nodes=[CodeGraphNodeItem(**node.__dict__) for node in graph.nodes],
        edges=[CodeGraphEdgeItem(**edge.__dict__) for edge in graph.edges],
        files=[
            CodeGraphFileItem(
                file_path=file.file_path,
                language=file.language,
                symbols=[CodeGraphNodeItem(**symbol.__dict__) for symbol in file.symbols],
                imports=[CodeGraphImportItem(**import_item.__dict__) for import_item in file.imports],
                parse_error=file.parse_error,
            )
            for file in graph.files
        ],
        summary=graph.summary,
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)
    files: list[SearchFileInput]


class SearchResultItem(BaseModel):
    text: str
    score: float
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ""
    parent_id: str = ""
    parent_start_line: int = 0
    parent_end_line: int = 0
    score_source: str = ""
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    reranker: str = ""


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResultItem]
    vector_backend: str = "memory"


def get_embedding_provider() -> EmbeddingProvider:
    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider in {"", "openai"}:
        return OpenAIEmbeddingProvider()
    if provider in {"local", "sentence_transformers", "sentence-transformers"}:
        return SentenceTransformerEmbeddingProvider()
    raise ConfigError(f"不支持的 EMBEDDING_PROVIDER: {provider}")


def build_vector_store_for_files(files: list[SearchFileInput]) -> tuple[VectorStoreProtocol | None, str]:
    """根据环境变量返回当前选择的向量库。"""
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory").strip().lower()
    if backend in {"", "memory", "in_memory", "in-memory"}:
        return None, "memory"
    if backend != "qdrant":
        raise ConfigError(f"不支持的 VECTOR_STORE_BACKEND: {backend}")

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ConfigError("未安装 qdrant-client") from exc

    base_collection = os.getenv("QDRANT_COLLECTION", "code_chunks").strip() or "code_chunks"
    collection_name = f"{base_collection}_{_files_digest(files)}"
    vector_size = _env_int("QDRANT_VECTOR_SIZE", 1536)
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    ), "qdrant"


def _files_digest(files: list[SearchFileInput]) -> str:
    digest = hashlib.sha1()
    for file in sorted(files, key=lambda item: item.file_path):
        digest.update(file.file_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数") from exc
    if value <= 0:
        raise ConfigError(f"{name} 必须是正整数")
    return value

# ── Reranker 依赖 ─────────────────────────────────────────────


_reranker: CrossEncoderReranker | IdentityReranker | None = None


def _env_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def get_reranker() -> CrossEncoderReranker | IdentityReranker:
    """返回当前配置的重排器。

    本地和 CI 稳定性优先：默认使用 ``IdentityReranker``，因为真实 CrossEncoder 模型较大，
    首次使用可能需要下载。设置 ``RERANKER_ENABLED=true`` 或
    ``RERANKER_PROVIDER=cross_encoder`` 后才启用真实重排。
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    provider = os.getenv("RERANKER_PROVIDER", "identity").strip().lower()
    if _env_enabled(os.getenv("RERANKER_ENABLED")) or provider in {"cross_encoder", "cross-encoder"}:
        _reranker = CrossEncoderReranker(fallback=IdentityReranker())
        logging.info("Reranker: 使用 CrossEncoderReranker，失败时回退到 IdentityReranker")
    else:
        _reranker = IdentityReranker()
        logging.info("Reranker: 默认使用 IdentityReranker")
    return _reranker


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_search_result_item(result) -> SearchResultItem:
    metadata = result.metadata
    return SearchResultItem(
        text=result.text,
        score=result.score,
        chunk_id=str(metadata.get("chunk_id", "")),
        file_path=str(metadata.get("file_path", "")),
        start_line=int(metadata.get("start_line", 0)),
        end_line=int(metadata.get("end_line", 0)),
        symbol_name=str(metadata.get("symbol_name", "")),
        parent_id=str(metadata.get("parent_id", "")),
        parent_start_line=int(metadata.get("parent_start_line", 0)),
        parent_end_line=int(metadata.get("parent_end_line", 0)),
        score_source=result.score_source or str(metadata.get("score_source", "")),
        vector_score=result.vector_score if result.vector_score is not None else _optional_float(metadata.get("vector_score")),
        keyword_score=result.keyword_score if result.keyword_score is not None else _optional_float(metadata.get("keyword_score")),
        rerank_score=result.rerank_score if result.rerank_score is not None else _optional_float(metadata.get("rerank_score")),
        reranker=result.reranker or str(metadata.get("reranker", "")),
    )

@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    reranker: CrossEncoderReranker | IdentityReranker = Depends(get_reranker),
) -> SearchResponse:
    if not request.files:
        raise HTTPException(status_code=400, detail="请提供要检索的代码文件")

    all_chunks: list[Chunk] = []
    for file in request.files:
        all_chunks.extend(chunk_file(file.file_path, file.content))

    if not all_chunks:
        return SearchResponse(query=request.query, top_k=request.top_k, results=[])

    try:
        vector_store, vector_backend = build_vector_store_for_files(request.files)
        keyword_index = KeywordIndex(all_chunks)
        search_results = search_code(
            query=request.query,
            chunks=all_chunks,
            top_k=request.top_k,
            embedding_provider=embedding_provider,
            keyword_index=keyword_index,
            reranker=reranker,
            vector_store=vector_store,
        )
    except (ConfigError, VectorStoreError):
        raise HTTPException(status_code=500, detail="检索服务配置错误")
    except EmbeddingError:
        raise HTTPException(status_code=502, detail="向量检索服务暂时不可用")

    parent_map = build_parent_map(all_chunks)
    search_results = attach_parent_metadata(search_results, parent_map)
    items = [_to_search_result_item(result) for result in search_results]

    return SearchResponse(query=request.query, top_k=request.top_k, results=items, vector_backend=vector_backend)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=20)
    files: list[SearchFileInput] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[SearchResultItem]
    vector_backend: str = "memory"


def get_llm_provider() -> LLMProvider:
    return OpenAILLMProvider()


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    llm: LLMProvider = Depends(get_llm_provider),
    reranker: CrossEncoderReranker | IdentityReranker = Depends(get_reranker),
) -> ChatResponse:
    all_chunks: list[Chunk] = []
    for file in request.files:
        all_chunks.extend(chunk_file(file.file_path, file.content))

    if not all_chunks:
        return ChatResponse(
            answer="没有找到相关代码文件。",
            retrieved_chunks=[],
        )

    try:
        vector_store, vector_backend = build_vector_store_for_files(request.files)
        keyword_index = KeywordIndex(all_chunks)
        search_results = search_code(
            query=request.question,
            chunks=all_chunks,
            top_k=request.top_k,
            embedding_provider=embedding_provider,
            keyword_index=keyword_index,
            reranker=reranker,
            vector_store=vector_store,
        )
    except (ConfigError, VectorStoreError):
        raise HTTPException(status_code=500, detail="检索服务配置错误")
    except EmbeddingError:
        raise HTTPException(status_code=502, detail="向量检索服务暂时不可用")

    # 将子 chunk 扩展为父文档，给 LLM 完整函数上下文
    parent_map = build_parent_map(all_chunks)
    search_results = expand_with_parents(search_results, parent_map, request.top_k)

    items = [_to_search_result_item(result) for result in search_results]

    prompt = build_rag_prompt(request.question, items)

    try:
        answer = llm.generate(prompt)
    except ConfigError:
        raise HTTPException(status_code=500, detail="LLM 服务配置错误")
    except LLMError:
        raise HTTPException(status_code=502, detail="LLM 服务暂时不可用")

    return ChatResponse(answer=answer, retrieved_chunks=items, vector_backend=vector_backend)

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    llm: LLMProvider = Depends(get_llm_provider),
    reranker: CrossEncoderReranker | IdentityReranker = Depends(get_reranker),
) -> StreamingResponse:
    all_chunks: list[Chunk] = []
    for file in request.files:
        all_chunks.extend(chunk_file(file.file_path, file.content))

    if not all_chunks:
        def empty_stream():
            payload = {"answer": "", "retrieved_chunks": [], "vector_backend": "memory"}
            yield _sse_event("done", payload)
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    try:
        vector_store, vector_backend = build_vector_store_for_files(request.files)
        keyword_index = KeywordIndex(all_chunks)
        search_results = search_code(
            query=request.question,
            chunks=all_chunks,
            top_k=request.top_k,
            embedding_provider=embedding_provider,
            keyword_index=keyword_index,
            reranker=reranker,
            vector_store=vector_store,
        )
    except (ConfigError, VectorStoreError):
        raise HTTPException(status_code=500, detail="检索服务配置错误")
    except EmbeddingError:
        raise HTTPException(status_code=502, detail="向量检索服务暂时不可用")

    parent_map = build_parent_map(all_chunks)
    search_results = expand_with_parents(search_results, parent_map, request.top_k)
    items = [_to_search_result_item(result) for result in search_results]
    prompt = build_rag_prompt(request.question, items)

    def event_stream():
        answer_parts: list[str] = []
        try:
            for delta in llm.stream(prompt):
                answer_parts.append(delta)
                yield _sse_event("chunk", {"delta": delta})
        except ConfigError as exc:
            yield _sse_event("error", {"error_type": "config_error", "detail": str(exc)})
            return
        except LLMError as exc:
            yield _sse_event("error", {"error_type": "llm_error", "detail": str(exc)})
            return

        yield _sse_event(
            "done",
            {
                "answer": "".join(answer_parts),
                "retrieved_chunks": [item.model_dump() for item in items],
                "vector_backend": vector_backend,
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class GenerationEvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


class GenerationEvaluationResponse(BaseModel):
    question: str
    evaluator: str
    faithfulness: float
    answer_relevance: float
    passed: bool
    unsupported_claims: list[str]
    missing_keywords: list[str]
    notes: str = ""


@app.post("/evaluate/generation", response_model=GenerationEvaluationResponse)
def evaluate_generation_api(request: GenerationEvaluationRequest) -> GenerationEvaluationResponse:
    case = GenerationEvalCase(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        expected_keywords=request.expected_keywords,
    )
    evaluator_mode = _generation_evaluator_mode()

    try:
        if evaluator_mode == "heuristic":
            result = evaluate_generation(case)
        elif evaluator_mode == "llm_judge":
            result = evaluate_generation_with_llm_judge(case)
        else:
            raise HTTPException(status_code=500, detail=f"不支持的生成质量评估模式：{evaluator_mode}")
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail="生成质量评估器配置错误") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="生成质量评估器暂时不可用") from exc

    return GenerationEvaluationResponse(
        question=result.question,
        evaluator=result.evaluator,
        faithfulness=result.faithfulness,
        answer_relevance=result.answer_relevance,
        passed=result.passed,
        unsupported_claims=result.unsupported_claims,
        missing_keywords=result.missing_keywords,
        notes=result.notes,
    )


def _generation_evaluator_mode() -> str:
    mode = os.getenv("GENERATION_EVALUATOR", "heuristic").strip().lower()
    aliases = {
        "": "heuristic",
        "offline": "heuristic",
        "rule": "heuristic",
        "rules": "heuristic",
        "llm-judge": "llm_judge",
        "llm_judge": "llm_judge",
        "online": "llm_judge",
    }
    return aliases.get(mode, mode)

# ── Agent /agent/run ──

class ConversationTurnItem(BaseModel):
    """单轮对话条目——API 接收的 history 格式。"""
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(default="")


class AgentRunRequest(BaseModel):
    """Agent 运行请求——模型自主决定调用哪些工具。"""
    question: str = Field(min_length=1)
    files: list[SearchFileInput] = Field(default_factory=list)
    top_k: int = Field(default=5, gt=0, le=50)
    max_steps: int = Field(default=3, gt=0, le=10)
    history: list[ConversationTurnItem] = Field(default_factory=list)
    session_id: str | None = Field(default=None, max_length=128)


class AgentToolCallItem(BaseModel):
    """单次工具调用的脱敏记录（面向 API 响应）。"""
    tool_name: str
    arguments: dict[str, object]
    reason: str = ""


class AgentEventItem(BaseModel):
    """单次工具事件的脱敏摘要（面向 API 响应）。"""
    tool_name: str
    success: bool
    error_type: str | None = None
    duration_ms: float = 0.0
    trace_id: str = ""


class AgentRunResponse(BaseModel):
    """Agent 运行响应——包含完整调用轨迹。"""
    answer: str
    tool_calls: list[AgentToolCallItem] = []
    events: list[AgentEventItem] = []
    errors: list[str] = []
    status: str
    trace_id: str
    retrieved_chunks: list[SearchResultItem] = []
    memory_used: bool = False
    memory_turns: int = 0
    session_id: str | None = None
    memory_scope: str = "request"



def _safe_request_file_path(file_path: str) -> Path:
    path = Path(file_path)
    if not file_path.strip() or path.is_absolute() or ".." in path.parts:
        raise HTTPException(status_code=400, detail="请求文件中包含不安全路径")
    return path


def _write_request_files(project_root: str, files: list[SearchFileInput]) -> None:
    root = Path(project_root).resolve()
    for file in files:
        relative_path = _safe_request_file_path(file.file_path)
        destination = (root / relative_path).resolve()
        try:
            destination.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=400, detail="请求文件中包含不安全路径")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(file.content, encoding="utf-8")

def get_agent_memory_store() -> AgentMemoryStore:
    return AgentMemoryStore()

def get_agent_model_provider() -> AgentModelProvider | None:
    """依赖注入，测试时可用 dependency_overrides 替换为假模型。"""
    try:
        return OpenAIAgentProvider()
    except ConfigError:
        # 没有 API key 时返回 None，由端点判断并返回可读错误
        return None


@app.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(
    request: AgentRunRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    _agent_provider: AgentModelProvider | None = Depends(get_agent_model_provider),
    memory_store: AgentMemoryStore = Depends(get_agent_memory_store),
) -> AgentRunResponse:
    """Agent 运行端点：模型自主决定用什么工具探索代码库。

    和 /chat 的区别：
      /chat 是固定检索→问答管线。
      /agent/run 是模型自己决定：先列文件还是先搜索、读哪个文件、什么时候停下。
    """
    # 切片阶段：files → chunks
    all_chunks: list[Chunk] = []
    for file in request.files:
        all_chunks.extend(chunk_file(file.file_path, file.content))

    # 短期记忆：API 的 ConversationTurnItem → runner 的 ConversationTurn
    from codebase_agent.agent.memory import ConversationTurn

    history_turns: list[ConversationTurn] = [
        ConversationTurn(role=turn.role, content=turn.content)
        for turn in request.history
    ]
    memory_scope = "request"
    if request.session_id:
        try:
            persisted_turns = memory_store.list_turns(request.session_id, limit=20)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        history_turns = persisted_turns + history_turns
        memory_scope = "session"

    # 决定模型提供器。
    # 如果测试时通过 dependency_overrides 注入了提供器，用它。
    # 否则检查真实提供器是否可用。
    if _agent_provider is None:
        raise HTTPException(
            status_code=500,
            detail="Agent 模型提供器未配置（缺少 OPENAI_API_KEY）",
        )

    try:
        with TemporaryDirectory(prefix="codebase-agent-") as project_root:
            _write_request_files(project_root, request.files)
            result: AgentRunResult = _run_agent(
                question=request.question,
                project_root=project_root,
                chunks=all_chunks,
                embedding_provider=embedding_provider,
                model_provider=_agent_provider,
                max_steps=request.max_steps,
                history=history_turns,
            )
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Agent 配置错误: {exc}")
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Agent 模型服务不可用: {exc}")

    # 映射 AgentRunResult → AgentRunResponse（脱敏处理）
    if request.session_id and result.answer:
        try:
            memory_store.append_turns(
                request.session_id,
                [
                    ConversationTurn(role="user", content=request.question),
                    ConversationTurn(role="assistant", content=result.answer),
                ],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    tool_call_items: list[AgentToolCallItem] = [
        AgentToolCallItem(
            tool_name=tc.tool_name,
            arguments=tc.arguments,
            reason=tc.reason,
        )
        for tc in result.tool_calls
    ]

    event_items: list[AgentEventItem] = [
        AgentEventItem(
            tool_name=ev.tool_name,
            success=ev.success,
            error_type=ev.error_type,
            duration_ms=ev.duration_ms,
            trace_id=ev.trace_id,
        )
        for ev in result.events
    ]

    # 把检索片段映射为公开 API 响应结构。
    retrieved_items = [_to_search_result_item(item) for item in result.retrieved_chunks]
    return AgentRunResponse(
        answer=result.answer,
        tool_calls=tool_call_items,
        events=event_items,
        errors=result.errors,
        status=result.status,
        trace_id=result.trace_id,
        retrieved_chunks=retrieved_items,
        memory_used=result.memory_used,
        memory_turns=result.memory_turns,
        session_id=request.session_id,
        memory_scope=memory_scope,
    )
