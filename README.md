# codebase-agent

面向代码库理解场景的 AI 应用项目，支持 GitHub 仓库分析、RAG 检索问答、Agent 工具调用、代码结构图、流式响应和检索/生成质量评估。

当前通过 FastAPI 提供 HTTP 接口，底层通过 GitHub API 获取仓库文件树，分析结果通过 SQLAlchemy 持久化；本地默认 SQLite，容器/生产可通过 DATABASE_URL 切换 PostgreSQL。

---
## 项目概览

这个项目包含后端 API、React/Vite 前端操作台、Agent 状态流说明、系统架构图和多语言静态代码结构分析能力，可用于代码库理解、检索问答、工具调用和质量评估。

### 本地启动

```powershell
# 后端
cd codebase-agent
venv\Scripts\python.exe -m uvicorn codebase_agent.backend.main:app --reload --host 127.0.0.1 --port 8000

# 前端
cd frontend
npm install
npm run dev
```

前端默认访问后端 `http://127.0.0.1:8000`，可在页面右上角修改“后端地址”。

### 前端控制台能力

- RAG 问答：调用 `/chat`，展示回答、引用来源、分数、父文档信息和 `vector_backend`。
- 流式问答：调用 `/chat/stream`，展示 SSE `chunk/done/error` 流程。
- Agent 分析：调用 `/agent/run`，展示回答、工具调用、执行事件、追踪 ID 和会话记忆状态。
- 代码结构图：调用 `/code-graph`，展示文件节点、函数/类/方法符号和 import 依赖关系；Python 使用 AST，JavaScript/TypeScript、Java、Go 使用轻量静态解析规则。
- 生成评估：调用 `/evaluate/generation`，展示忠实度、答案相关性、无证据支撑内容和缺失关键词。

### 架构图

```mermaid
flowchart LR
    FE["React/Vite 前端控制台"] --> API["FastAPI 接口层"]
    API --> RAG["RAG 检索问答链路"]
    API --> AG["Agent 执行链路"]
    API --> CG["代码结构图接口"]
    API --> EV["评估接口"]
    RAG --> EMB["OpenAI Embeddings"]
    RAG --> VS{"向量库"}
    VS --> MEM["内存向量库"]
    VS --> QD["Qdrant 向量库"]
    RAG --> LLM["LLM / 流式输出"]
    AG --> TOOLS["列文件 / 读文件 / 搜代码"]
    AG --> SESSION["会话记忆"]
    CG --> AST["Python AST + 多语言轻量解析"]
    EV --> METRICS["Hit@K / MRR / 忠实度 / 相关性"]
```

### 架构文档

- 系统架构图：`docs/architecture.md`
- Agent 状态流说明：`docs/agent-langgraph-flow.md`
- 前端工程：`frontend/`

---


## 当前功能

- `GET /health` — 健康检查
- `POST /analyze` — 提交 GitHub 仓库地址，拉取文件列表并记录分析历史
- `GET /history?limit=20` — 查看最近分析记录
- `POST /code-graph` — 基于请求文件快照做多语言静态结构分析，返回文件、符号和 import 依赖图

---

## 目录结构

```
codebase-agent/
├── codebase_agent/
│   ├── backend/
│   │   ├── main.py              # FastAPI 应用入口（路由、中间件、依赖注入）
│   │   ├── github_client.py     # GitHub API 客户端（异步，文件树拉取）
│   │   ├── models.py             # SQLAlchemy ORM 模型（AnalysisRecord）
│   │   └── database.py          # SQLAlchemy 持久化（建表、写入、查询）
│   ├── rag/
│   │   ├── chunker.py            # 代码文件按函数/类边界分块
│   │   ├── embeddings.py         # OpenAI Embedding 封装
│   │   ├── vector_store.py       # 本地向量检索与 Top-K 排序
│   │   ├── keyword_search.py     # 关键词检索
│   │   ├── hybrid_search.py      # 向量 + 关键词混合检索
│   │   ├── reranker.py           # 默认重排 / CrossEncoder 重排
│   │   ├── parent_document.py    # 父文档上下文扩展
│   │   ├── qdrant_store.py       # Qdrant 适配层
│   │   └── evaluator.py          # 检索与生成质量评估
│   ├── agent/
│   │   ├── runner.py             # Agent 主循环
│   │   ├── tools.py              # list_files / get_file_content / search_code
│   │   ├── llm_provider.py       # Agent 调用 OpenAI 兼容模型
│   │   ├── tool_contracts.py     # Agent 工具结果、事件和路由数据结构
│   │   └── memory_store.py       # 会话记忆持久化
│   ├── code_graph.py             # 多语言静态代码结构图
│   ├── exceptions.py            # 自定义异常（网络错误 / 限流错误等）
│   └── utils.py                 # 通用工具（定时器、缓存装饰器）
├── frontend/                    # React/Vite 前端操作台
├── docs/                        # 架构图与 Agent 流程图说明
├── scripts/
│   ├── validate_project.py      # 一键自检脚本
│   ├── evaluate_rag.py          # 固定评估集检索评估
│   ├── github_fetcher.py        # GitHub 文件列表导出工具
│   ├── json_reader.py           # 仓库文件 JSON 摘要工具
│   ├── api_tester.py            # HTTP 接口测试工具
│   └── file_stats.py            # 文本文件统计工具
├── data/                        # 固定评估集与评估结果
├── tests/                       # pytest 测试
├── requirements.txt
├── requirements-dev.txt         # 测试和本地开发依赖
├── Dockerfile                   # 容器镜像构建
├── .dockerignore                # 容器构建排除文件
├── compose.yaml                 # Docker Compose 服务编排
├── .env.example                 # 环境变量模板
└── .gitignore
```

---

## 环境准备

- Python 3.10+

### Windows

```powershell
# 1. 进入项目目录
cd codebase-agent

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
venv\Scripts\activate

# 4. 安装运行依赖
pip install -r requirements.txt

# 5. 配置环境变量
copy .env.example .env
# 编辑 .env，把 GITHUB_TOKEN 替换成你的 GitHub Personal Access Token
# 如果使用 RAG Embeddings，再填入 OPENAI_API_KEY
```

---

## Docker 启动（推荐）

不需要手动配置 Python 环境，容器内已经包含所有依赖。

### 前置要求

- [Docker](https://docs.docker.com/get-docker/) 和 Docker Compose V2（`docker compose`）

### 快速启动

```powershell
# 1. 进入项目目录
cd codebase-agent

# 2. 复制环境变量模板，填入你的 GitHub Token
copy .env.example .env
# 编辑 .env：GITHUB_TOKEN=your_github_token_here

# 3. 构建并启动
docker compose up --build
```

### 验证

```powershell
# 健康检查
curl http://localhost:8000/health
# → {"status": "ok"}

# 或浏览器打开 Swagger 文档
# http://localhost:8000/docs
```

### 停止

```powershell
docker compose down
```

### 本地 venv 启动 vs 容器启动

| | 本地 venv | Docker 容器 |
|---|---|---|
| 环境配置 | 手动装 Python、创建 venv、pip install | `docker compose up --build` 一条命令 |
| 隔离性 | 依赖全局 Python 环境 | 容器内独立运行，不污染宿主机 |
| 跨机器复现 | "在我的机器上能跑" | 任何装了 Docker 的机器都能跑 |
| 适用场景 | 本地开发调试 | 部署、CI、环境复现 |

### 容器启动排错

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| `docker: command not found` | Docker 未安装 | 安装 Docker Desktop 或 Docker Engine |
| `port is already allocated` | 宿主机 8000 端口被占用 | 改 `compose.yaml` 中端口为 `"8001:8000"`，或停掉占用进程 |
| 容器启动后立刻退出 | 依赖安装失败或代码报错 | `docker compose logs app` 查看容器日志 |
| `ConfigError: GITHUB_TOKEN not set` | `.env` 未创建或 Token 未填写 | 检查 `.env` 文件，确保 `GITHUB_TOKEN` 有值 |
| 容器内数据库路径错误 | `DATABASE_URL` 指向不存在的目录 | 检查 `compose.yaml` 中 `environment.DATABASE_URL` 值 |

---

## 运行测试

```powershell
# 本地首次运行测试前安装开发依赖
pip install -r requirements-dev.txt

# Windows 推荐使用项目 venv，并指定本地临时目录，避免系统 Temp 权限影响测试
venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .test-tmp\pytest-readme
```

当前项目验证以实际运行结果为准，不在 README 中固定写通过数量。

---

## 启动 API

```powershell
# 确保 venv 已激活
python -m uvicorn codebase_agent.backend.main:app --reload
```

启动后访问：
- Swagger 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

---

## 接口示例

### 健康检查

```powershell
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

### 提交分析

```powershell
curl -X POST http://127.0.0.1:8000/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"repo_url\": \"https://github.com/psf/requests\"}"
```

```json
{"files": ["requests/__init__.py", "requests/api.py", "..."], "count": 42}
```

### 查看历史

```powershell
curl "http://127.0.0.1:8000/history?limit=3"
```

```json
[{"id": 1, "repo_url": "https://github.com/psf/requests", "count": 42, "create_at": "2026-06-29 12:00:00"}]
```

### RAG 代码问答

`/chat` 会先切分代码，通过 OpenAI Embeddings 检索 Top-K 片段，再把检索结果交给 LLM 生成回答。响应同时返回引用片段，便于定位答案依据。

```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"数据库连接在哪里实现？\",\"top_k\":2,\"files\":[{\"file_path\":\"db.py\",\"content\":\"def connect_db(): return 'connected'\"}]}"
```

```json
{
  "answer": "数据库连接由 db.py 中的 connect_db 函数负责。",
  "retrieved_chunks": [
    {
      "text": "def connect_db(): return 'connected'",
      "score": 0.98,
      "chunk_id": "db.py:1-1",
      "file_path": "db.py",
      "start_line": 1,
      "end_line": 1,
      "symbol_name": "connect_db"
    }
  ]
}
```

`/search` 只返回检索证据；`/chat` 在相同检索链路上增加 Prompt 构建和 LLM 回答。测试使用假提供器隔离网络、费用和非确定输出，chunk、相似度排序、metadata 和响应组装仍走真实业务逻辑。

---

## 常见启动问题 / 排错

| 症状 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'codebase_agent'` | venv 未激活或依赖未装 | `venv\Scripts\activate` 然后 `pip install -r requirements.txt` |
| `ConfigError: GITHUB_TOKEN not set` | 缺 `.env` 或 Token 未填 | `copy .env.example .env` 然后填入真实 Token |
| 验证脚本输出乱码 | Windows 控制台编码不兼容 | 确保脚本显式使用 UTF-8 输出 |
| pytest 警告：`StarletteDeprecationWarning` | Starlette TestClient 与 httpx 的兼容提示 | 当前不影响功能，后续统一升级依赖时处理 |

---

## 当前架构

```
客户端 (curl / Swagger)
        │
        ▼
┌───────────────────┐
│   FastAPI API 层   │  ← main.py：路由、Pydantic 校验、依赖注入、HTTPException
└────────┬──────────┘
         │
┌────────▼──────────┐
│  GitHub 客户端     │  ← github_client.py：异步 HTTP，GitHub API 文件树拉取
│  业务层            │
└────────┬──────────┘
         │
┌────────▼──────────┐
│  SQLAlchemy + SQLite  │  ← database.py + models.py：ORM 模型、建表、写入、查询
└────────┬──────────┘
         │
┌────────▼──────────┐
│  pytest 测试层     │  ← tests/：覆盖 API/CLI/DB/GitHub 客户端
└───────────────────┘
```

---

## 数据库层

项目数据库层使用 **SQLAlchemy 2.x ORM**，通过 `DATABASE_URL` 支持 SQLite/PostgreSQL 切换，适合本地开发、容器化运行和后续迁移管理。

### 为什么迁移

| 对比维度 | sqlite3 手写 SQL | SQLAlchemy ORM |
|----------|-----------------|----------------|
| 建表 | `CREATE TABLE IF NOT EXISTS (...)` | `Base.metadata.create_all(engine)` |
| 写入 | `cur.execute("INSERT INTO ... VALUES (?,?,?)")` | `session.add(record)` + `commit()` |
| 查询 | `cur.fetchall()` → 手动 `dict(row)` | `select().order_by()` → ORM 对象 → `list[dict]` |
| 字段映射 | `row[0]`, `row[1]`，顺序依赖 | `record.repo_url`，IDE 补全 |
| 事务管理 | 手动 `conn.commit()` / 异常无回滚 | Session: try/commit/except rollback/close |
| 换数据库 | 重写 SQL 方言 | 只改连接串（`sqlite:///` → `postgresql://`） |

### ORM 模型

`AnalysisRecord` 对应 `analysis_record` 表，四个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 自增主键 |
| `repo_url` | str | 仓库地址，非空 |
| `count` | int | 文件数量，非空 |
| `create_at` | str | 分析时间（ISO 格式） |

### 当前设计

- **连接管理**：`build_database_url` 优先读 `DATABASE_URL` 环境变量，否则用本地 db_path 构造 `sqlite:///...`——为 Docker/PostgreSQL 做铺垫
- **建表**：`Base.metadata.create_all()`，适合本地 SQLite 快速迭代
- **Session**：每个数据库操作创建独立 Session，遵循 `try/commit/except rollback/close`
- **排序**：`ORDER BY create_at DESC, id DESC` 保证同秒记录的确定性排序

### 数据库层演进路线

```
sqlite3 手写 SQL
  │  原型阶段：验证分析历史持久化流程
  ▼
SQLAlchemy ORM（当前）
  │  工程化阶段：对象映射、Session 事务管理、跨数据库方言屏蔽
  ▼
DATABASE_URL 环境变量支持
  │  build_database_url 优先读环境变量，本地/Docker 自动切换
  ▼
Docker Compose 注入
  │  compose.yaml 设置 DATABASE_URL=postgresql://...
  ▼
PostgreSQL 切换
  │  改一行环境变量：sqlite:/// → postgresql://
  ▼
Alembic Migration
     增量 schema 变更，生产环境可控升级/回退
```

当前已通过 `DATABASE_URL` 环境变量和 `compose.yaml` 支持容器化启动；默认 SQLite 可快速运行，开启 `postgres` profile 后可切换 PostgreSQL。

---

## RAG Embeddings

系统已提供 `OpenAIEmbeddingProvider`、`SentenceTransformerEmbeddingProvider`、`InMemoryVectorStore`、`QdrantVectorStore` 和 `OpenAILLMProvider`。代码 chunk 和用户 query 转成向量后，按余弦相似度返回 Top-K 片段，再通过 `/chat` 构建受证据约束的 Prompt，并返回回答与 `retrieved_chunks`。

当前支持两种向量存储后端：
- **InMemoryVectorStore**（默认）：内存索引，每次请求重建，适合快速验证
- **QdrantVectorStore**：持久化向量索引，跨请求复用，支持 payload 过滤，适合生产环境

环境变量：

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_deepseek_or_openai_chat_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_CHAT_MODEL=deepseek-chat
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
GENERATION_EVALUATOR=heuristic
LLM_JUDGE_MODEL=
```

`OPENAI_*` 用于 Chat/Agent，可以接 DeepSeek 或 OpenAI 兼容服务；`EMBEDDING_*` 单独用于外部 Embedding，避免把向量请求发到不支持 Embedding 的 Chat 服务。`EMBEDDING_PROVIDER=openai` 适合生产或有稳定 OpenAI 兼容 Embeddings 服务的环境；`EMBEDDING_PROVIDER=local` 使用本地 sentence-transformers，适合离线或内网环境。测试默认使用假的 embedding/LLM 提供器，不请求真实外部 API；`InMemoryVectorStore` 的余弦相似度、Top-K 排序、metadata、Prompt 构建、LLM 失败和 `/chat` 响应链路已经有测试覆盖。

---

## RAG 检索质量评估

使用固定 20 条评估问题集，跨模块覆盖 FastAPI 路由、Pydantic、SQLAlchemy、代码切块、embedding、向量库、prompt、LLM 提供器和引用来源。评估集位于 `data/eval_set.json`。

### 运行评估

```powershell
# 需要 OPENAI_API_KEY 和 OPENAI_EMBEDDING_MODEL 环境变量
.\venv\Scripts\python.exe scripts\evaluate_rag.py
```

结构化结果会保存到 `data/eval_results.json`，每条失败样本都包含失败原因；单条坏样本不会中断整批评估。

### 当前基线结果

| 指标 | 值 |
|------|-----|
| 评估样本数 | 20 |
| 命中数 | 20 |
| Hit@1 | **0.800** |
| Hit@5 | **1.000** |
| MRR | **0.888** |
| Embedding 模型 | BAAI/bge-m3（通过 SiliconFlow 调用） |
| Chunk 总数 | 81 |

### Top-1 失败案例（Hit@5=1 但首位未命中）

以下案例虽命中但排名靠后，表明检索排序仍有改进空间：

| # | 问题 | 期望文件 | 实际排名 | RR |
|---|------|---------|---------|-----|
| 1 | “项目定义了哪些 FastAPI 路由？” | `backend/main.py` | 第4位 | 0.25 |
| 2 | “Embedding 提供器使用什么模型？” | `rag/embeddings.py` | 第2位 | 0.50 |
| 3 | “GitHub 客户端如何处理 API 错误？” | `backend/github_client.py` | 第2位 | 0.50 |
| 4 | “代码分析器如何提取符号？” | `codebase_agent/code_analyzer.py` | 第2位 | 0.50 |

**案例 1 分析**：问 FastAPI 路由时，`github_client.py` 排在 `main.py` 前面。可能原因是这些文件也包含网络/请求相关模式，与 query 的语义产生部分匹配。

**案例 2 分析**：问 embedding 模型时，`main.py` 排第一（因为其中引用了 `OpenAIEmbeddingProvider`），而 `embeddings.py`（定义本身）排第二。chunk 粒度和引用链影响了排序。

**案例 3 分析**：问错误处理时，`exceptions.py`（定义异常类）排在 `github_client.py`（使用异常类）前面。表明错误处理语义更接近定义侧而非使用侧。

**案例 4 分析**：问符号提取时，`chunker.py` 的代码切分语义排在 `codebase_agent/code_analyzer.py` 前面，说明相近模块之间仍存在语义混淆。

### 改进方向

- 增大 chunk_size 或引入父文档策略，减少同一主题分散到多个 chunk
- 引入重排（Cross-Encoder），在 Top-K 结果上重新排序
- 评估集扩展：增加更多难度梯度和边界场景（空结果、多义词、简称）
- 监控 chunk 质量：某些文件可能缺少对关键概念的显式提及

### 生成质量评估

- 检索质量使用 Hit@K / MRR 评估，定位召回是否命中目标文件。
- 生成质量接口 `/evaluate/generation` 输出忠实度、答案相关性、无证据支撑内容和缺失关键词。
- 默认 `GENERATION_EVALUATOR=heuristic`，使用离线规则评估，适合 CI、本地验收和回归检查。
- 演示或真实质量验收可设置 `GENERATION_EVALUATOR=llm_judge`，接口会调用线上模型做 LLM 评审；`LLM_JUDGE_MODEL` 为空时复用 `OPENAI_CHAT_MODEL`。
- API 返回 `evaluator` 字段，可直接看出本次使用的是 `heuristic` 还是 `llm_judge`。

---

## Qdrant 向量存储适配层

`QdrantVectorStore`（`codebase_agent/rag/qdrant_store.py`）实现了与 `InMemoryVectorStore` 相同的 `add_chunks` / `search` 接口，上层 API 无需修改即可切换。

### 架构

```
codebase_agent/rag/vector_store.py     # SearchResult 稳定边界
codebase_agent/rag/qdrant_store.py     # Qdrant 适配层（同一接口）
```

`QdrantVectorStore` 支持客户端注入，测试可使用假客户端隔离外部依赖，不依赖真实 Qdrant 服务。

### 启动 Qdrant

```bash
# Docker Compose（可选 profile，不影响默认 app 启动）
docker compose --profile qdrant up -d qdrant
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `QDRANT_URL` | Qdrant 服务地址 | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API Key（云端必填） | — |
| `QDRANT_COLLECTION` | Collection 名称 | `code_chunks` |
| `VECTOR_STORE_BACKEND` | 向量存储后端 | `memory`（可选 `qdrant`） |

### 迁移要点

- `SearchResult` 结构不变（text/score/metadata）
- `/search` 和 `/chat` 的 `retrieved_chunks` 字段不变
- `QdrantVectorStore` 的 `add_chunks` / `search` 签名与 `InMemoryVectorStore` 一致
- point id 基于 `chunk_id` 使用 uuid5 生成，重复 upsert 自动覆盖
- payload 保留 `text` / `chunk_id` / `file_path` / `start_line` / `end_line` / `symbol_name`

### API 自动切换

- `/search`、`/chat`、`/chat/stream` 已接入 `VECTOR_STORE_BACKEND`。
- 默认 `memory` 适合本地快速运行；设置 `VECTOR_STORE_BACKEND=qdrant` 后自动创建基于文件内容 hash 的 Qdrant collection，避免不同项目 chunk 混排。
- API 响应包含 `vector_backend`，可直接说明本次请求使用的是内存索引还是 Qdrant 持久索引。

---

## Agent 工具调用

codebase-agent 提供 `/agent/run` 端点，支持模型自主决定调用哪些工具探索代码库。API 会把本次请求传入的 `files` 写入隔离的临时项目目录，`list_files` / `get_file_content` 基于这份请求快照运行，避免误读服务器当前工作目录。

### 可用工具

| 工具 | 用途 | 输入 | 输出 | 失败行为 |
|------|------|------|------|----------|
| `list_files` | 列出项目中的文件 | `file_pattern`（默认 `*.py`） | 排序后的相对路径列表 | 目录不存在返回 `[]` |
| `get_file_content` | 读取指定文件内容 | `path`（相对路径） | UTF-8 文件内容 | 不存在→文件不存在错误；越权→权限拒绝 |
| `search_code` | 语义搜索代码 | `query` + `top_k` | 检索结果列表（含分数和元数据） | 空结果→`[]`；embedding 异常→Embedding 错误 |

### /agent/run API

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "数据库连接在哪实现？",
    "files": [{"file_path": "backend/database.py", "content": "def get_engine(): ..."}],
    "top_k": 3,
    "max_steps": 5,
    "history": [
      {"role": "user", "content": "项目有哪些文件"},
      {"role": "assistant", "content": "backend/database.py, backend/main.py, ..."}
    ]
  }'
```

### 响应字段

| 字段 | 说明 |
|------|------|
| `answer` | 模型生成的最终回答 |
| `tool_calls` | 每一步工具调用记录（工具名 + 参数 + 原因） |
| `events` | 每一步工具执行的脱敏审计事件（是否成功 / 错误类型 / 耗时 / 追踪 ID） |
| `errors` | 执行过程中的错误列表 |
| `status` | 执行状态：完成 / 部分完成 / 失败 |
| `trace_id` | 本次运行的唯一追踪 ID（12 位 hex） |
| `retrieved_chunks` | search_code 工具调用产出的检索证据（片段文本 / 分数 / 文件路径 / 行号 / 分数来源 / 重排器 / 父文档 ID） |
| `memory_used` | 是否使用了短期记忆（history 非空） |
| `memory_turns` | 实际保留的历史轮数 |

### 短期记忆

`/agent/run` 支持通过 `history` 字段传入多轮对话历史，实现上下文指代传递：

```
第一轮: "数据库连接在哪实现？"  → Agent 搜索并回答 backend/database.py
第二轮: "那它失败时怎么处理？"  → Agent 通过 history 知道"它"=数据库连接
```

- 滑动窗口：默认保留最近 10 轮，总字符数不超过 4000
- 空 content 自动过滤
- history 为空时 `memory_used=false`，不影响正常流程
- 非法 role（非 user/assistant/system）→ 422

### 执行器保障

- **白名单注册表**：只有登记的工具可被调用，未知工具名直接拒绝
- **参数校验**：必填参数缺失、类型错误、未知参数均转为结构化错误
- **请求快照隔离**：`/agent/run` 将请求内文件写入临时目录，工具只读这份快照
- **路径安全**：规范解析后比对项目根目录前缀，拒绝 `..` 越权
- **事件脱敏**：每次调用产出 `ToolEvent`（工具名 + 参数摘要 + 结果摘要 + 耗时 + trace_id），不记录完整文件内容或密钥
- **最大步数**：默认 3 步，可配置 1-10，超步数返回 partial + 降级答案

### 测试策略

- Agent 测试使用假模型提供器固定模型决策，隔离网络、费用和随机性
- 工具执行、参数校验、状态更新、失败处理 **全部真实运行**——不能 fake
- 检索链路（chunk → embedding → cosine → Top-K）真实计算，仅用假 embedding 提供器隔离外部 API

---

## 工程能力状态

当前项目按可运行、可测试、可解释和可复现标准验收：API 可运行、RAG 可检索、Agent 可调用工具、结果可追踪、评估可复现、Docker 可启动。

### 已完成能力

- **数据库**：SQLAlchemy 2.x 持久化，默认 SQLite，`DATABASE_URL` 可切换 PostgreSQL；`compose.yaml` 提供 `postgres` profile。
- **RAG 检索**：OpenAI Embeddings、向量检索、BM25 关键词检索、RRF 融合、父文档扩展、可选 Cross-Encoder 重排。
- **Qdrant**：`VECTOR_STORE_BACKEND=qdrant` 后 `/search`、`/chat`、`/chat/stream` 自动使用 Qdrant 持久向量库，collection 按文件内容 hash 隔离。
- **流式响应**：新增 `/chat/stream`，使用 SSE 输出 `chunk`、`done`、`error` 事件，前端可边生成边展示答案。
- **Agent**：`/agent/run` 支持工具路由、最大步数保护、错误事件、trace_id、请求快照隔离和 `session_id` 会话级持久记忆。
- **代码结构图**：`/code-graph` 基于 Python AST 和多语言轻量解析规则提取文件、函数、类、方法和 import 依赖，前端展示结构化图与明细。当前是静态结构分析，不是全语言运行时调用链分析。
- **评估**：检索评估覆盖 Hit@K/MRR；生成评估提供 `/evaluate/generation`，输出 faithfulness、answer_relevance、unsupported_claims、missing_keywords。
- **前端展示**：新增 React/Vite 操作台，真实调用 `/chat`、`/chat/stream`、`/agent/run`、`/code-graph`、`/evaluate/generation`。
- **流程图证据**：新增 README Mermaid 架构图、`docs/architecture.md` 和 `docs/agent-langgraph-flow.md`。
- **测试**：新增覆盖 Qdrant 后端切换、SSE 流式响应、Agent session 记忆和生成质量评估的回归测试。

### 可选增强项

这些不是当前验收阻塞项，只在具备运行环境或质量验收需要时开启：

- `RERANKER_ENABLED=true`：加载真实 Cross-Encoder Reranker，首次运行可能下载模型。
- `VECTOR_STORE_BACKEND=qdrant`：使用 Qdrant 持久向量库，需要先启动 `docker compose --profile qdrant up -d qdrant`。
- PostgreSQL：启动 `docker compose --profile postgres up -d postgres` 后，将 `DATABASE_URL` 设置为 `postgresql+psycopg://app_user:app_pass@localhost:5432/codebase_agent`。

### 设计取舍

这个项目不是只调 LLM API，而是把代码文件清洗切块、向量化、混合召回、上下文扩展、生成回答、Agent 工具调用、错误追踪和评估闭环串起来。对幻觉的约束不是依赖单一提示词，而是通过引用来源、检索评估、生成质量评估和失败事件把回答变成可检查结果。
