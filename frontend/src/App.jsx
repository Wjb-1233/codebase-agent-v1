import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  Boxes,
  Database,
  FileCode2,
  Gauge,
  GitBranch,
  Loader2,
  MessageSquareText,
  Network,
  Play,
  Radio,
  Search,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";

const DEFAULT_FILES = [
  {
    file_path: "backend/database.py",
    content: "from sqlalchemy import create_engine\n\nclass Database:\n    def connect(self, database_url):\n        return create_engine(database_url)\n",
  },
  {
    file_path: "rag/vector_store.py",
    content: "from rag.embeddings import OpenAIEmbeddingProvider\n\ndef search_code(query, chunks, embedding_provider):\n    return vector_store.search(query)\n",
  },
  {
    file_path: "agent/runner.py",
    content: "from rag.vector_store import search_code\n\nasync def run_agent(question, max_steps=3):\n    while step_count < max_steps:\n        dispatch_tool()\n",
  },
  {
    file_path: "frontend/App.jsx",
    content: "import { helper } from './utils'\n\nexport function App() {\n  return helper()\n}\n",
  },
  {
    file_path: "frontend/utils.ts",
    content: "export const helper = () => 'ok'\n",
  },
  {
    file_path: "service/App.java",
    content: "import java.util.List;\n\npublic class App {\n  public void run() {}\n}\n",
  },
  {
    file_path: "cmd/server.go",
    content: "package main\n\nimport \"fmt\"\n\nfunc main() {}\n",
  },
];

const API_BASE_KEY = "codebase-agent-api-base";

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function parseFiles(raw) {
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error("files 必须是数组");
  }
  for (const file of parsed) {
    if (!file.file_path || typeof file.content !== "string") {
      throw new Error("每个文件必须包含 file_path 和 content");
    }
  }
  return parsed;
}

async function requestJson(baseUrl, path, payload) {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

function parseSseBlock(block) {
  const eventLine = block.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
  if (!dataLine) return null;
  return {
    event: eventLine ? eventLine.replace("event:", "").trim() : "message",
    data: JSON.parse(dataLine.replace("data:", "").trim()),
  };
}

async function requestStream(baseUrl, payload, onEvent) {
  const response = await fetch(`${baseUrl}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const event = parseSseBlock(block);
      if (event) onEvent(event);
    }
  }

  if (buffer.trim()) {
    const event = parseSseBlock(buffer);
    if (event) onEvent(event);
  }
}

function App() {
  const [baseUrl, setBaseUrl] = useState(() => localStorage.getItem(API_BASE_KEY) || "http://127.0.0.1:8000");
  const [filesText, setFilesText] = useState(formatJson(DEFAULT_FILES));
  const [activeTab, setActiveTab] = useState("chat");
  const [githubUrl, setGithubUrl] = useState("");
  const [githubFetching, setGithubFetching] = useState(false);
  const [githubError, setGithubError] = useState("");

  function updateBaseUrl(value) {
    setBaseUrl(value);
    localStorage.setItem(API_BASE_KEY, value);
  }

  async function fetchGithubFiles() {
    if (!githubUrl.trim() || githubFetching) return;
    setGithubFetching(true);
    setGithubError("");
    try {
      const data = await requestJson(baseUrl, "/github/fetch", { repo_url: githubUrl.trim() });
      setFilesText(formatJson(data.files || []));
    } catch (err) {
      setGithubError(err.message);
    } finally {
      setGithubFetching(false);
    }
  }

  const files = useMemo(() => {
    try {
      return parseFiles(filesText);
    } catch {
      return [];
    }
  }, [filesText]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Bot size={22} /></div>
          <div>
            <h1>代码库智能分析控制台</h1>
            <p>RAG 检索、Agent 工具轨迹、生成评估一体化控制台</p>
          </div>
        </div>
        <label className="api-field">
          <span>后端地址</span>
          <input value={baseUrl} onChange={(event) => updateBaseUrl(event.target.value)} />
        </label>
      </header>

      <section className="overview-grid" aria-label="项目能力概览">
        <Capability icon={<Search />} title="混合 RAG" text="向量检索 + BM25 + RRF + 父文档扩展" />
        <Capability icon={<Database />} title="向量库可切换" text="内存与 Qdrant 后端可切换，响应返回向量后端类型" />
        <Capability icon={<Radio />} title="流式响应" text="问答过程可逐段返回，并在结束时返回完整来源" />
        <Capability icon={<GitBranch />} title="Agent 追踪" text="工具调用、执行事件、错误和追踪 ID 可查看" />
        <Capability icon={<Network />} title="代码结构图" text="静态解析多语言文件，提取符号和依赖关系" />
      </section>

      <section className="workbench">
        <aside className="side-panel">
          <div className="panel-title"><FileCode2 size={18} />请求文件快照</div>
          <div className="github-fetch">
            <input
              value={githubUrl}
              onChange={(event) => setGithubUrl(event.target.value)}
              placeholder="GitHub 仓库 URL（可选，拉取后填充快照）"
            />
            <button onClick={fetchGithubFiles} disabled={githubFetching || !githubUrl.trim()} type="button">
              {githubFetching ? "拉取中…" : "拉取文件"}
            </button>
          </div>
          {githubError && <div className="status-line error">{githubError}</div>}
          <textarea
            className="files-editor"
            value={filesText}
            onChange={(event) => setFilesText(event.target.value)}
            spellCheck="false"
            wrap="off"
          />
          <div className={files.length ? "status-line ok" : "status-line error"}>
            {files.length ? `${files.length} 个文件将作为本次请求快照` : "JSON 格式错误或文件为空"}
          </div>
        </aside>

        <section className="main-panel">
          <nav className="tabs" aria-label="功能切换">
            <TabButton active={activeTab === "chat"} onClick={() => setActiveTab("chat")} icon={<MessageSquareText />} label="RAG 问答" />
            <TabButton active={activeTab === "agent"} onClick={() => setActiveTab("agent")} icon={<Wrench />} label="Agent" />
            <TabButton active={activeTab === "graph"} onClick={() => setActiveTab("graph")} icon={<Network />} label="代码结构" />
            <TabButton active={activeTab === "eval"} onClick={() => setActiveTab("eval")} icon={<Gauge />} label="生成评估" />
          </nav>

          {activeTab === "chat" && <ChatPanel baseUrl={baseUrl} filesText={filesText} />}
          {activeTab === "agent" && <AgentPanel baseUrl={baseUrl} filesText={filesText} />}
          {activeTab === "graph" && <CodeGraphPanel baseUrl={baseUrl} filesText={filesText} />}
          {activeTab === "eval" && <EvaluationPanel baseUrl={baseUrl} />}
        </section>
      </section>

      <section className="architecture-strip">
        <img src="/assets/architecture-strip.svg" alt="Codebase Agent 架构流程" />
        <div>
          <h2>系统能力概览</h2>
          <p>从文件快照进入系统，先做 chunk 与检索，再进入回答生成或 Agent 工具循环；所有结果都有来源、事件、错误和评估证据。</p>
        </div>
      </section>
    </main>
  );
}

function Capability({ icon, title, text }) {
  return (
    <article className="capability">
      <div className="capability-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{text}</p>
    </article>
  );
}

function TabButton({ active, onClick, icon, label }) {
  return (
    <button className={active ? "tab active" : "tab"} onClick={onClick} type="button">
      {icon}
      <span>{label}</span>
    </button>
  );
}

function ChatPanel({ baseUrl, filesText }) {
  const [question, setQuestion] = useState("数据库连接在哪里实现？");
  const [topK, setTopK] = useState(3);
  const [sessionId, setSessionId] = useState("");
  const [mode, setMode] = useState("idle");
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState([]);
  const [backend, setBackend] = useState("");
  const [memoryScope, setMemoryScope] = useState("");
  const [error, setError] = useState("");

  async function runChat(streaming) {
    setMode(streaming ? "streaming" : "loading");
    setError("");
    setAnswer("");
    setChunks([]);
    setBackend("");
    try {
      const files = parseFiles(filesText);
      const payload = {
        question,
        top_k: Number(topK),
        files,
        ...(sessionId.trim() ? { session_id: sessionId.trim() } : {}),
      };
      if (!streaming) {
        const data = await requestJson(baseUrl, "/chat", payload);
        setAnswer(data.answer || "");
        setChunks(data.retrieved_chunks || []);
        setBackend(data.vector_backend || "memory");
        setMemoryScope(data.memory_scope || "");
      } else {
        await requestStream(baseUrl, payload, ({ event, data }) => {
          if (event === "chunk") {
            setAnswer((current) => current + (data.delta || ""));
          }
          if (event === "done") {
            setAnswer(data.answer || "");
            setChunks(data.retrieved_chunks || []);
            setBackend(data.vector_backend || "memory");
            setMemoryScope(data.memory_scope || "");
          }
          if (event === "error") {
            setError(`${formatStreamError(data.error_type)}: ${data.detail}`);
          }
        });
      }
      setMode("done");
    } catch (err) {
      setError(err.message);
      setMode("error");
    }
  }

  return (
    <div className="panel-body">
      <div className="form-grid two">
        <label>
          <span>问题</span>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} />
        </label>
        <label>
          <span>返回数量 Top K</span>
          <input type="number" min="1" max="20" value={topK} onChange={(event) => setTopK(event.target.value)} />
        </label>
        <label>
          <span>会话 ID（可选，留空为单轮）</span>
          <input value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="如 demo-1" />
        </label>
      </div>
      <div className="button-row">
        <button className="primary" onClick={() => runChat(false)} disabled={mode === "loading" || mode === "streaming"} type="button">
          {mode === "loading" ? <Loader2 className="spin" /> : <Play />}
          普通问答
        </button>
        <button className="secondary" onClick={() => runChat(true)} disabled={mode === "loading" || mode === "streaming"} type="button">
          {mode === "streaming" ? <Loader2 className="spin" /> : <Radio />}
          流式问答
        </button>
      </div>
      <ResultBlock error={error} answer={answer} backend={backend} memoryScope={memoryScope} />
      <SourceList chunks={chunks} />
    </div>
  );
}

function AgentPanel({ baseUrl, filesText }) {
  const [question, setQuestion] = useState("帮我找出数据库连接相关代码，并说明失败时怎么处理");
  const [sessionId, setSessionId] = useState("default-session");
  const [maxSteps, setMaxSteps] = useState(5);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function runAgent() {
    setStatus("loading");
    setError("");
    setResult(null);
    try {
      const files = parseFiles(filesText);
      const payload = { question, session_id: sessionId, max_steps: Number(maxSteps), top_k: 5, files };
      const data = await requestJson(baseUrl, "/agent/run", payload);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <div className="panel-body">
      <div className="form-grid three">
        <label>
          <span>Agent 问题</span>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} />
        </label>
        <label>
          <span>会话 ID</span>
          <input value={sessionId} onChange={(event) => setSessionId(event.target.value)} />
        </label>
        <label>
          <span>最大步数</span>
          <input type="number" min="1" max="10" value={maxSteps} onChange={(event) => setMaxSteps(event.target.value)} />
        </label>
      </div>
      <div className="button-row">
        <button className="primary" onClick={runAgent} disabled={status === "loading"} type="button">
          {status === "loading" ? <Loader2 className="spin" /> : <Sparkles />}
          运行 Agent
        </button>
      </div>
      {error && <ErrorNotice message={error} />}
      {!result && !error && <EmptyNotice text="运行后会展示回答、工具调用、执行事件、追踪 ID 和会话记忆状态。" />}
      {result && (
        <div className="agent-grid">
          <article className="answer-card">
            <div className="card-label"><ShieldCheck size={16} />状态：{formatRunStatus(result.status)}</div>
            <p>{result.answer || "暂无回答"}</p>
            <div className="meta-row">
              <span>追踪 ID: {result.trace_id}</span>
              <span>记忆: {formatMemoryScope(result.memory_scope)}/{result.memory_used ? "已使用" : "未使用"}</span>
              <span>轮次: {result.memory_turns}</span>
            </div>
          </article>
          <TraceList title="工具调用" items={result.tool_calls || []} empty="本次没有调用工具" />
          <TraceList title="执行事件" items={result.events || []} empty="暂无事件" />
        </div>
      )}
    </div>
  );
}

function CodeGraphPanel({ baseUrl, filesText }) {
  const [graph, setGraph] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function runCodeGraph() {
    setStatus("loading");
    setError("");
    setGraph(null);
    try {
      const files = parseFiles(filesText);
      const data = await requestJson(baseUrl, "/code-graph", { files });
      setGraph(data);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <div className="panel-body">
      <div className="button-row">
        <button className="primary" onClick={runCodeGraph} disabled={status === "loading"} type="button">
          {status === "loading" ? <Loader2 className="spin" /> : <Network />}
          生成代码结构图
        </button>
      </div>
      {error && <ErrorNotice message={error} />}
      {!graph && !error && <EmptyNotice text="基于当前请求文件快照做多语言静态结构分析：Python 用 AST，JS/TS、Java、Go 用轻量解析规则。" />}
      {graph && (
        <div className="graph-layout">
          <section className="score-grid">
            <ScoreCard title="文件数" value={graph.summary?.files ?? 0} />
            <ScoreCard title="支持语言文件" value={graph.summary?.supported_language_files ?? 0} />
            <ScoreCard title="符号数" value={graph.summary?.symbols ?? 0} />
            <ScoreCard title="导入数" value={graph.summary?.imports ?? 0} />
            <ScoreCard title="外部导入" value={graph.summary?.external_imports ?? 0} />
          </section>
          <CodeGraphSvg graph={graph} />
          <section className="graph-columns">
            <article className="trace-card">
              <div className="card-label"><FileCode2 size={16} />文件与符号</div>
              {graph.files.map((file) => (
                <div className="graph-file" key={file.file_path}>
                  <strong>{file.file_path}</strong>
                  <div className="meta-row">
                    <span>语言 {formatLanguage(file.language)}</span>
                    <span>{file.symbols.length} 个符号</span>
                    <span>{file.imports.length} 个导入</span>
                  </div>
                  {file.parse_error && <p className="parse-error">{file.parse_error}</p>}
                  {file.symbols.map((symbol) => (
                    <p key={symbol.id}>{symbol.type} · {symbol.label} · 第 {symbol.line} 行</p>
                  ))}
                </div>
              ))}
            </article>
            <TraceList title="依赖边" items={graph.edges || []} empty="没有识别到依赖边" />
          </section>
        </div>
      )}
    </div>
  );
}

function CodeGraphSvg({ graph }) {
  const fileNodes = (graph.nodes || []).filter((node) => node.type === "file").slice(0, 6);
  if (!fileNodes.length) return null;
  const width = 820;
  const height = Math.max(180, fileNodes.length * 74);
  const positions = new Map(
    fileNodes.map((node, index) => [
      node.id,
      { x: index % 2 === 0 ? 90 : 470, y: 44 + Math.floor(index / 2) * 88 },
    ])
  );
  const visibleEdges = (graph.edges || []).filter((edge) => positions.has(edge.source) && positions.has(edge.target));

  return (
    <article className="graph-canvas" aria-label="代码结构关系图">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <defs>
          <marker id="graph-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" />
          </marker>
        </defs>
        {visibleEdges.map((edge, index) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          return (
            <path
              key={`${edge.source}-${edge.target}-${index}`}
              className={`graph-edge ${edge.type}`}
              d={`M${source.x + 140} ${source.y + 22} C ${source.x + 210} ${source.y + 22}, ${target.x - 70} ${target.y + 22}, ${target.x} ${target.y + 22}`}
            />
          );
        })}
        {fileNodes.map((node) => {
          const position = positions.get(node.id);
          return (
            <g key={node.id} transform={`translate(${position.x} ${position.y})`}>
              <rect width="220" height="48" rx="8" />
              <text x="14" y="29">{node.label}</text>
            </g>
          );
        })}
      </svg>
    </article>
  );
}

function EvaluationPanel({ baseUrl }) {
  const [question, setQuestion] = useState("数据库连接在哪里实现？");
  const [answer, setAnswer] = useState("数据库连接在 get_engine 中实现，并通过 create_engine 创建连接。 ");
  const [contexts, setContexts] = useState("def get_engine(database_url):\n    return create_engine(database_url)");
  const [keywords, setKeywords] = useState("get_engine,create_engine,数据库");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function runEval() {
    setStatus("loading");
    setError("");
    setResult(null);
    try {
      const data = await requestJson(baseUrl, "/evaluate/generation", {
        question,
        answer,
        contexts: contexts.split("\n---\n").filter(Boolean),
        expected_keywords: keywords.split(",").map((item) => item.trim()).filter(Boolean),
      });
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <div className="panel-body">
      <div className="form-grid two eval-grid">
        <label>
          <span>问题</span>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} />
        </label>
        <label>
          <span>期望关键词，逗号分隔</span>
          <input value={keywords} onChange={(event) => setKeywords(event.target.value)} />
        </label>
        <label>
          <span>模型答案</span>
          <textarea value={answer} onChange={(event) => setAnswer(event.target.value)} />
        </label>
        <label>
          <span>检索上下文，多个上下文用 --- 分隔</span>
          <textarea value={contexts} onChange={(event) => setContexts(event.target.value)} />
        </label>
      </div>
      <div className="button-row">
        <button className="primary" onClick={runEval} disabled={status === "loading"} type="button">
          {status === "loading" ? <Loader2 className="spin" /> : <Gauge />}
          评估生成质量
        </button>
      </div>
      {error && <ErrorNotice message={error} />}
      {result && (
        <div className="score-grid">
          <ScoreCard title="忠实度" value={result.faithfulness} />
          <ScoreCard title="答案相关性" value={result.answer_relevance} />
          <ScoreCard title="是否通过" value={result.passed ? "通过" : "未通过"} />
          <ScoreCard title="评估器" value={formatEvaluator(result.evaluator)} />
          <article className="answer-card wide">
            <div className="card-label"><AlertTriangle size={16} />待检查项</div>
            <p>无证据支撑内容: {(result.unsupported_claims || []).join(", ") || "无"}</p>
            <p>缺失关键词: {(result.missing_keywords || []).join(", ") || "无"}</p>
          </article>
        </div>
      )}
    </div>
  );
}

function ResultBlock({ error, answer, backend, memoryScope }) {
  if (error) return <ErrorNotice message={error} />;
  if (!answer) return <EmptyNotice text="还没有回答。先提交一次普通问答或流式问答。" />;
  const memoryLabel = memoryScope === "session" ? "· 会话记忆" : "";
  return (
    <article className="answer-card">
      <div className="card-label"><Activity size={16} />回答 {backend ? `· 向量后端=${formatBackend(backend)}` : ""}{memoryLabel}</div>
      <p>{answer}</p>
    </article>
  );
}

function SourceList({ chunks }) {
  if (!chunks?.length) return null;
  return (
    <section className="source-list">
      <div className="panel-title"><Boxes size={18} />引用来源</div>
      {chunks.map((chunk) => (
        <article className="source-item" key={`${chunk.chunk_id}-${chunk.score}`}>
          <div className="source-head">
            <strong>{chunk.file_path}</strong>
            <span>分数 {Number(chunk.score).toFixed(3)}</span>
          </div>
          <p>{chunk.text}</p>
          <div className="meta-row">
            <span>行号 {chunk.start_line}-{chunk.end_line}</span>
            <span>{formatScoreSource(chunk.score_source || "vector")}</span>
            {chunk.parent_id && <span>父文档 {chunk.parent_id}</span>}
          </div>
        </article>
      ))}
    </section>
  );
}

function TraceList({ title, items, empty }) {
  return (
    <article className="trace-card">
      <div className="card-label"><Wrench size={16} />{title}</div>
      {!items.length && <p>{empty}</p>}
      {items.map((item, index) => (
        <pre key={`${title}-${index}`}>{formatJson(localizeTraceItem(item))}</pre>
      ))}
    </article>
  );
}

function ScoreCard({ title, value }) {
  return (
    <article className="score-card">
      <span>{title}</span>
      <strong>{typeof value === "number" ? value.toFixed(3) : value}</strong>
    </article>
  );
}

function formatBackend(value) {
  const map = {
    memory: "内存",
    qdrant: "Qdrant",
  };
  return map[value] || value;
}

function formatEvaluator(value) {
  const map = {
    heuristic: "离线规则",
    llm_judge: "线上模型评审",
  };
  return map[value] || value || "未返回";
}

function formatLanguage(value) {
  const map = {
    python: "Python",
    javascript: "JavaScript",
    typescript: "TypeScript",
    java: "Java",
    go: "Go",
  };
  return map[value] || value;
}

function formatScoreSource(value) {
  const map = {
    vector: "向量检索",
    keyword: "关键词检索",
    both: "混合检索",
  };
  return map[value] || value;
}

function formatRunStatus(value) {
  const map = {
    completed: "已完成",
    partial: "部分完成",
    failed: "失败",
    running: "运行中",
  };
  return map[value] || value;
}

function formatMemoryScope(value) {
  const map = {
    request: "单次请求",
    session: "会话",
  };
  return map[value] || value;
}

function formatStreamError(value) {
  const map = {
    config_error: "配置错误",
    llm_error: "模型调用错误",
  };
  return map[value] || value || "未知错误";
}

function localizeTraceItem(value) {
  if (Array.isArray(value)) {
    return value.map(localizeTraceItem);
  }
  if (!value || typeof value !== "object") {
    return localizeTraceValue(value);
  }
  const keyMap = {
    tool_name: "工具名称",
    arguments: "参数",
    reason: "原因",
    success: "是否成功",
    error_type: "错误类型",
    error_message: "错误说明",
    duration_ms: "耗时毫秒",
    trace_id: "追踪 ID",
    query: "查询语句",
    top_k: "返回数量",
    path: "文件路径",
    file_pattern: "文件匹配规则",
  };
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      keyMap[key] || key,
      localizeTraceItem(item),
    ])
  );
}

function localizeTraceValue(value) {
  const map = {
    true: "是",
    false: "否",
    search_code: "搜索代码",
    list_files: "列出文件",
    get_file_content: "读取文件内容",
    unknown_tool: "未知工具",
    invalid_argument: "参数无效",
    permission_denied: "拒绝访问",
    tool_not_configured: "工具未配置",
    FileNotFoundError: "文件不存在",
    repeated_tool_call_blocked: "重复工具调用已拦截",
    max_steps_exceeded: "达到最大执行步数",
    answer_generation_failed: "答案生成失败",
  };
  return map[String(value)] || value;
}

function ErrorNotice({ message }) {
  return (
    <div className="notice error" role="alert">
      <AlertTriangle size={18} />
      <span>{message}</span>
    </div>
  );
}

function EmptyNotice({ text }) {
  return (
    <div className="notice empty">
      <ShieldCheck size={18} />
      <span>{text}</span>
    </div>
  );
}

export default App;
