"""Agent API 端点测试：POST /agent/run。

覆盖：成功链路 / 测试假模型依赖覆盖 / 响应字段完整性 / 失败状态 / 无提供器降级。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from codebase_agent.backend.main import app, get_agent_memory_store, get_agent_model_provider, get_embedding_provider
from codebase_agent.agent.runner import AgentDecision
from tests.fakes import FakeAgentModelProvider


client = TestClient(app)


# ═══════════════════════ 测试用 Embedding 提供器 ═══════════════════════

class FakeEmbeddingProvider:
    """测试假实现：不碰 API key，不碰网络。"""

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


_FAKE_EMBED = FakeEmbeddingProvider()


def _setup_overrides(*, decisions, final_answer="基于工具执行结果的分析。"):
    """一键设置两个依赖覆盖：模型提供器和 embedding 提供器。"""
    app.dependency_overrides[get_embedding_provider] = lambda: _FAKE_EMBED
    app.dependency_overrides[get_agent_model_provider] = lambda: FakeAgentModelProvider(
        decisions=list(decisions),
        final_answer=final_answer,
    )


# ═══════════════════════ 1. 直接回答路径 ═══════════════════════

def test_agent_run_direct_answer():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="你好！有什么可以帮你的？")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "你好",
            "files": [{"file_path": "main.py", "content": "def main():\n    pass\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["answer"] == "你好！有什么可以帮你的？"
    assert data["tool_calls"] == []
    assert data["events"] == []
    assert data["errors"] == []
    assert "trace_id" in data


# ═══════════════════════ 2. 工具调用链路 ═══════════════════════

def test_agent_run_list_files():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.py"},
            ),
        ],
        final_answer="项目包含 1 个 Python 文件。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "有哪些文件",
            "files": [{"file_path": "main.py", "content": "print('hello')\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool_name"] == "list_files"
    assert len(data["events"]) == 1
    assert data["events"][0]["success"] is True


def test_agent_run_search_code():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="search_code",
                arguments={"query": "database", "top_k": 1},
            ),
        ],
        final_answer="数据库在 db.py 中。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "数据库在哪",
            "files": [
                {"file_path": "db.py", "content": "def connect_db():\n    return 'connected'\n"}
            ],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool_name"] == "search_code"
    assert len(data["events"]) == 1
    assert data["events"][0]["success"] is True


# ═══════════════════════ 3. 失败路径 → 有 errors，不 500 ═══════════════════════

def test_agent_run_path_traversal_in_errors():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="get_file_content",
                arguments={"path": "../.env"},
            ),
        ],
        final_answer="访问被拒绝。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "读 .env",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["errors"]) >= 1
    assert any("路径越权" in e for e in data["errors"])
    event = data["events"][0]
    assert event["error_type"] == "permission_denied"
    assert "路径越权" in event["error_message"]
    assert "../.env" not in str(event)


def test_agent_run_unknown_tool_in_errors():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="delete_everything",
                arguments={},
            ),
        ],
        final_answer="无法执行。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "删库",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["errors"]) >= 1
    assert any("未注册工具" in e for e in data["errors"])
    assert data["events"][0]["success"] is False
    assert data["events"][0]["error_type"] == "unknown_tool"
    assert "未知工具" in data["events"][0]["error_message"]


def test_agent_run_max_steps_exceeded():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.py"},
            ),
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.py"},
            ),
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.py"},
            ),
        ],
        final_answer="部分分析结果。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "分析",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
            "max_steps": 3,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "partial"
    assert any("重复工具调用" in e for e in data["errors"])


# ═══════════════════════ 4. 多步链路 ═══════════════════════

def test_agent_run_multi_step_chain():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.py"},
            ),
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.txt"},
            ),
            AgentDecision(
                is_direct_answer=True,
                content="分析完毕。",
            ),
        ],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "分析项目",
            "files": [
                {"file_path": "main.py", "content": "def main():\n    pass\n"},
                {"file_path": "utils.py", "content": "def helper():\n    return 42\n"},
            ],
            "max_steps": 5,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["tool_calls"]) == 2
    assert data["tool_calls"][0]["tool_name"] == "list_files"
    assert data["tool_calls"][1]["tool_name"] == "list_files"
    assert len(data["events"]) == 2
    assert data["events"][0]["success"] is True
    assert data["events"][1]["success"] is True


# ═══════════════════════ 5. 无模型提供器降级 ═══════════════════════

def test_agent_run_no_provider_returns_500():
    """未配置 API key 时返回 500：只覆盖 embedding，不覆盖 agent。"""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_embedding_provider] = lambda: _FAKE_EMBED
    app.dependency_overrides[get_agent_model_provider] = lambda: None

    resp = client.post(
        "/agent/run",
        json={"question": "测试", "files": []},
    )

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "未配置" in detail or "OPENAI_API_KEY" in detail


# ═══════════════════════ 6. 请求校验 ═══════════════════════

def test_agent_run_empty_question_rejected():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="...")],
    )

    resp = client.post(
        "/agent/run",
        json={"question": "", "files": []},
    )

    assert resp.status_code == 422


def test_agent_run_response_field_completeness():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="回答。")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "测试",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "tool_calls" in data
    assert "events" in data
    assert "errors" in data
    assert "status" in data
    assert "trace_id" in data
    assert "retrieved_chunks" in data
    assert "memory_used" in data
    assert "memory_turns" in data
    assert data["status"] in ("completed", "partial", "failed")
    assert isinstance(data["tool_calls"], list)
    assert isinstance(data["events"], list)
    assert isinstance(data["errors"], list)
    assert isinstance(data["retrieved_chunks"], list)
    assert isinstance(data["memory_used"], bool)
    assert isinstance(data["memory_turns"], int)


def test_agent_run_trace_id_unique():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="ok")],
    )

    body = {"question": "测试", "files": [{"file_path": "main.py", "content": "pass\n"}]}
    r1 = client.post("/agent/run", json=body).json()
    r2 = client.post("/agent/run", json=body).json()

    assert r1["trace_id"] != r2["trace_id"]


def test_agent_run_events_desensitized():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="get_file_content",
                arguments={"path": "config.py"},
            ),
        ],
        final_answer="读取完毕。",
    )

    long_content = "# config\n" + "x" * 200 + "\n"
    resp = client.post(
        "/agent/run",
        json={
            "question": "读配置",
            "files": [{"file_path": "config.py", "content": long_content}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    event = data["events"][0]
    event_str = str(event)
    assert "x" * 200 not in event_str


# ═══════════════════════ 7. 短期记忆 ═══════════════════════

def test_agent_run_with_history():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="我看到了之前的对话。")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "那它怎么处理失败",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
            "history": [
                {"role": "user", "content": "数据库在哪实现"},
                {"role": "assistant", "content": "在 database.py"},
            ],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["memory_used"] is True
    assert data["memory_turns"] == 2


def test_agent_run_without_history():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="无历史。")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "测试",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["memory_used"] is False
    assert data["memory_turns"] == 0


def test_agent_run_invalid_role_422():
    """history 中 role 非法 → Pydantic 校验 422。"""
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="...")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "测试",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
            "history": [
                {"role": "admin", "content": "不应该允许"},
            ],
        },
    )

    assert resp.status_code == 422


def test_agent_run_empty_history_role_422():
    """history 中 role 空字符串 → Pydantic 校验 422。"""
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="...")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "测试",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
            "history": [
                {"role": "", "content": "空 role"},
            ],
        },
    )

    assert resp.status_code == 422


# ═══════════════════════ 8. retrieved_chunks ═══════════════════════

def test_agent_run_retrieved_chunks_from_search_code():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="search_code",
                arguments={"query": "database", "top_k": 1},
            ),
        ],
        final_answer="数据库在 db.py。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "数据库在哪",
            "files": [
                {"file_path": "db.py", "content": "def connect_db():\n    return 'ok'\n"}
            ],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["retrieved_chunks"]) >= 1
    assert data["retrieved_chunks"][0]["file_path"] == "db.py"
    assert data["retrieved_chunks"][0]["score"] > 0


def test_agent_run_no_retrieved_chunks_without_search():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="list_files",
                arguments={"file_pattern": "*.py"},
            ),
        ],
        final_answer="有文件。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "有哪些文件",
            "files": [{"file_path": "main.py", "content": "pass\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieved_chunks"] == []


# ═══════════════════════ 9. 响应字段完整性（含新字段）══

def test_agent_run_get_file_content_uses_request_files():
    _setup_overrides(
        decisions=[
            AgentDecision(
                is_direct_answer=False,
                tool_name="get_file_content",
                arguments={"path": "uploaded_only.py"},
            ),
        ],
        final_answer="读取完成。",
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "读取上传文件",
            "files": [{"file_path": "uploaded_only.py", "content": "def only_in_request():\n    return 1\n"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["events"][0]["success"] is True
    assert data["errors"] == []


def test_agent_run_rejects_unsafe_uploaded_file_path():
    _setup_overrides(
        decisions=[AgentDecision(is_direct_answer=True, content="不会执行")],
    )

    resp = client.post(
        "/agent/run",
        json={
            "question": "测试路径",
            "files": [{"file_path": "../escape.py", "content": "pass\n"}],
        },
    )

    assert resp.status_code == 400


def test_agent_run_persists_session_memory(tmp_path):
    from codebase_agent.agent.memory_store import AgentMemoryStore

    memory_db = tmp_path / "agent_memory.db"
    app.dependency_overrides[get_agent_memory_store] = lambda: AgentMemoryStore(str(memory_db))
    _setup_overrides(decisions=[], final_answer="first answer")

    first = client.post(
        "/agent/run",
        json={
            "question": "Where is database code?",
            "session_id": "session-1",
            "files": [{"file_path": "database.py", "content": "def get_engine():\n    pass\n"}],
        },
    )
    assert first.status_code == 200
    assert first.json()["memory_scope"] == "session"

    _setup_overrides(decisions=[], final_answer="second answer")
    app.dependency_overrides[get_agent_memory_store] = lambda: AgentMemoryStore(str(memory_db))
    second = client.post(
        "/agent/run",
        json={
            "question": "What did we discuss before?",
            "session_id": "session-1",
            "files": [{"file_path": "database.py", "content": "def get_engine():\n    pass\n"}],
        },
    )

    data = second.json()
    assert second.status_code == 200
    assert data["memory_scope"] == "session"
    assert data["session_id"] == "session-1"
    assert data["memory_used"] is True
    assert data["memory_turns"] >= 2
