"""Agent 运行器单元测试。

覆盖：直接回答 / 工具调用成功 / 缺参数 / 越权 / 未知工具 / 最大步数 / 多步链路。

核心测试原则：
  测试假 LLM 控制模型决策（隔离网络/费用/随机性）
  工具执行、参数校验、状态更新、失败处理 全部真实运行
"""

from __future__ import annotations

import pytest

from codebase_agent.agent.runner import (
    AgentDecision,
    AgentRunResult,
    AgentState,
    AgentToolCall,
    run_agent,
)
from tests.fakes import FakeAgentModelProvider
from codebase_agent.agent.executor import _bootstrap_registry
from codebase_agent.rag.chunker import Chunk


# ═══════════════════════ 测试夹具 ═══════════════════════

class FakeEmbeddingProvider:
    """返回确定性向量的测试提供器：隔离外部 API，不伪造检索排序。"""

    def __init__(self, vectors_by_text: dict[str, list[float]] | None = None):
        self.vectors_by_text = vectors_by_text or {}
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        result = []
        for text in texts:
            if text in self.vectors_by_text:
                result.append(self.vectors_by_text[text])
            elif "database" in text.lower():
                result.append([1.0, 0.0, 0.0])
            elif "html" in text.lower():
                result.append([0.0, 1.0, 0.0])
            else:
                result.append([0.3, 0.3, 0.4])
        return result


@pytest.fixture
def fake_embedding():
    return FakeEmbeddingProvider()


@pytest.fixture
def project_dir(tmp_path):
    """模拟项目目录。"""
    (tmp_path / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "client.py").write_text("class Client:\n    pass\n", encoding="utf-8")
    return str(tmp_path)


def _make_chunk(chunk_id: str, text: str, file_path: str = "app.py") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        file_path=file_path,
        start_line=1,
        end_line=3,
        symbol_name="handler",
    )


# 确保工具注册表已初始化（执行器调度依赖它）
_bootstrap_registry()


# ═══════════════════════ 1. 直接回答路径 ═══════════════════════

class TestDirectAnswer:
    """模型说"不用调工具，我直接回答"。"""

    def test_direct_answer_returns_immediately(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(is_direct_answer=True, content="这是一个简单问题，直接回答。"),
            ],
            final_answer="兜底回答",
        )
        result = run_agent(
            question="你好",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
            max_steps=3,
        )

        assert result.status == "completed"
        assert result.answer == "这是一个简单问题，直接回答。"
        assert result.tool_calls == []
        assert result.events == []
        assert result.errors == []

    def test_direct_answer_has_trace_id(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[AgentDecision(is_direct_answer=True, content="回答。")],
        )
        result = run_agent(
            question="test",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )
        assert result.trace_id != ""
        assert len(result.trace_id) == 12


# ═══════════════════════ 2. list_files 工具调用 ═══════════════════════

class TestListFilesTool:
    def test_lists_files_in_project(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="list_files",
                    arguments={"file_pattern": "*.py"},
                ),
            ],
            final_answer="项目包含 3 个 Python 文件。",
        )
        result = run_agent(
            question="有哪些文件",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "list_files"
        assert result.tool_calls[0].arguments == {"file_pattern": "*.py"}

        assert len(result.events) == 1
        assert result.events[0].success is True
        assert result.events[0].tool_name == "list_files"
        assert result.events[0].duration_ms > 0
        assert result.events[0].trace_id == result.trace_id

        assert result.answer == "项目包含 3 个 Python 文件。"

    def test_lists_md_files(self, project_dir, fake_embedding):
        """file_pattern 过滤生效。"""
        import os as _os
        readme_path = project_dir + "/README.md"
        if not _os.path.exists(readme_path):
            open(readme_path, "w").close()

        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="list_files",
                    arguments={"file_pattern": "*.md"},
                ),
            ],
        )
        result = run_agent(
            question="列出 md 文件",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )
        assert result.status == "completed"
        assert len(result.events) == 1
        assert result.events[0].success is True


# ═══════════════════════ 3. get_file_content 工具调用 ═══════════════════════

class TestGetFileContentTool:
    def test_reads_file_successfully(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={"path": "main.py"},
                ),
            ],
            final_answer="main.py 包含一个 main 函数。",
        )
        result = run_agent(
            question="读 main.py",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_file_content"
        assert result.tool_calls[0].arguments == {"path": "main.py"}

        assert len(result.events) == 1
        assert result.events[0].success is True
        assert result.answer == "main.py 包含一个 main 函数。"


# ═══════════════════════ 4. search_code 工具调用 ═══════════════════════

class TestSearchCodeTool:
    def test_searches_code_successfully(self, project_dir, fake_embedding):
        chunks = [
            _make_chunk("chunk-db", "def connect_db():\n    pass\n", "db.py"),
            _make_chunk("chunk-ui", "def render_ui():\n    pass\n", "ui.py"),
        ]
        emb = FakeEmbeddingProvider({
            chunks[0].text: [1.0, 0.0, 0.0],
            chunks[1].text: [0.0, 1.0, 0.0],
            "database": [1.0, 0.0, 0.0],
        })
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="search_code",
                    arguments={"query": "database", "top_k": 1},
                ),
            ],
            final_answer="数据库连接在 db.py 中。",
        )
        result = run_agent(
            question="数据库在哪",
            project_root=project_dir,
            chunks=chunks,
            embedding_provider=emb,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search_code"

        assert len(result.events) == 1
        assert result.events[0].success is True


# ═══════════════════════ 5. 失败路径 ═══════════════════════

class TestFailurePaths:
    def test_missing_required_param(self, project_dir, fake_embedding):
        """模型返回 get_file_content 但缺 path → executor 拒绝。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={},  # 缺 path
                ),
            ],
            final_answer="无法读取文件。",
        )
        result = run_agent(
            question="读文件",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"  # 有降级答案
        assert len(result.errors) >= 1
        assert any("invalid_argument" in e or "缺少" in e for e in result.errors)
        assert not result.events[0].success

    def test_path_traversal_rejected(self, project_dir, fake_embedding):
        """模型请求 ../.env → executor 拒绝，事件脱敏。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={"path": "../.env"},
                ),
            ],
            final_answer="访问被拒绝。",
        )
        result = run_agent(
            question="读 .env",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.errors) >= 1
        assert any("路径越权" in e for e in result.errors)

        # 事件摘要不暴露越权路径
        event = result.events[0]
        input_str = str(event.input_summary)
        assert "../.env" not in input_str

    def test_unknown_tool(self, project_dir, fake_embedding):
        """模型请求不存在的工具 → executor 返回 unknown_tool。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="destroy_database",
                    arguments={},
                ),
            ],
            final_answer="无法执行该操作。",
        )
        result = run_agent(
            question="删除数据库",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.errors) >= 1
        assert any("未注册工具" in e for e in result.errors)
        assert not result.events[0].success

    def test_file_not_found_produces_error(self, project_dir, fake_embedding):
        """请求读取不存在的文件 → error 记录，不崩溃。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={"path": "nonexistent.py"},
                ),
            ],
            final_answer="文件不存在。",
        )
        result = run_agent(
            question="读不存在的文件",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.errors) >= 1
        assert any("文件不存在" in e for e in result.errors)

    def test_repeated_tool_call_is_blocked(self, project_dir, fake_embedding):
        """模型重复调用相同工具时，runner 提前拦截，防止无效循环。"""
        provider = FakeAgentModelProvider(
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
        result = run_agent(
            question="分析项目",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
            max_steps=3,
        )

        assert result.status == "partial"
        assert len(result.tool_calls) == 1
        assert len(result.errors) >= 1
        assert any("重复工具调用" in e for e in result.errors)
        # 即使触发循环保护，仍生成了降级答案
        assert result.answer != ""


# ═══════════════════════ 6. 多步链路 ═══════════════════════

class TestMultiStep:
    def test_list_then_read_then_answer(self, project_dir, fake_embedding):
        """模拟真实 Agent：先列文件 → 再读文件 → 模型回答。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="list_files",
                    arguments={"file_pattern": "*.py"},
                ),
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={"path": "main.py"},
                ),
                AgentDecision(
                    is_direct_answer=True,
                    content="main.py 只定义了一个空 main 函数，项目比较简单。",
                ),
            ],
        )
        result = run_agent(
            question="分析项目结构",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
            max_steps=5,
        )

        assert result.status == "completed"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "list_files"
        assert result.tool_calls[1].tool_name == "get_file_content"
        assert len(result.events) == 2
        assert result.events[0].success is True
        assert result.events[1].success is True
        assert result.answer == "main.py 只定义了一个空 main 函数，项目比较简单。"

    def test_first_fails_second_succeeds(self, project_dir, fake_embedding):
        """第一步失败，第二步成功 → 仍有答案。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={"path": "nonexistent.py"},  # 第一步失败
                ),
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="list_files",
                    arguments={"file_pattern": "*.py"},  # 第二步成功
                ),
            ],
            final_answer="项目有多个 Python 文件，但 nonexistent.py 不存在。",
        )
        result = run_agent(
            question="检查文件",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
            max_steps=5,
        )

        assert result.status == "completed"
        assert len(result.tool_calls) == 2
        assert not result.events[0].success  # 第一步失败
        assert result.events[1].success  # 第二步成功
        assert len(result.errors) >= 1
        assert "文件不存在" in result.errors[0]
        assert result.answer != ""

    def test_search_then_read_best_match(self, project_dir, fake_embedding):
        """先搜索找到目标文件，再读取它。"""
        chunks = [
            _make_chunk("chunk-db", "def connect_db():\n    pass\n", "db.py"),
        ]
        emb = FakeEmbeddingProvider({
            chunks[0].text: [1.0, 0.0, 0.0],
            "database": [1.0, 0.0, 0.0],
        })
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="search_code",
                    arguments={"query": "database", "top_k": 1},
                ),
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="get_file_content",
                    arguments={"path": "db.py"},
                ),
            ],
            final_answer="数据库实现在 db.py 的 connect_db 函数中。",
        )
        result = run_agent(
            question="数据库怎么连的",
            project_root=project_dir,
            chunks=chunks,
            embedding_provider=emb,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "search_code"
        assert result.tool_calls[1].tool_name == "get_file_content"


# ═══════════════════════ 7. 状态字段完整性 ═══════════════════════

class TestStateIntegrity:
    def test_result_contains_all_required_fields(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[AgentDecision(is_direct_answer=True, content="回答。")],
        )
        result = run_agent(
            question="test",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        # 验证所有字段都存在且类型正确
        assert isinstance(result.question, str)
        assert isinstance(result.answer, str)
        assert result.status in ("completed", "partial", "failed")
        assert isinstance(result.tool_calls, list)
        assert isinstance(result.events, list)
        assert isinstance(result.errors, list)
        assert isinstance(result.trace_id, str)

    def test_each_run_gets_unique_trace_id(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[AgentDecision(is_direct_answer=True, content="回答。")],
        )
        result1 = run_agent(
            question="q1", project_root=project_dir, chunks=[],
            embedding_provider=fake_embedding, model_provider=provider,
        )
        result2 = run_agent(
            question="q2", project_root=project_dir, chunks=[],
            embedding_provider=fake_embedding, model_provider=provider,
        )
        assert result1.trace_id != result2.trace_id


# ═══════════════════════ 8. 短期记忆 ═══════════════════════

class TestShortTermMemory:
    def test_memory_context_appears_in_state(self, project_dir, fake_embedding):
        """AgentState 应该携带构建好的 memory_context。"""
        from codebase_agent.agent.memory import ConversationTurn

        history = [
            ConversationTurn(role="user", content="数据库在哪"),
            ConversationTurn(role="assistant", content="在 database.py"),
        ]

        # 用 FakeAgentModelProvider 捕获状态。
        captured_states: list[AgentState] = []

        class StateCapturingProvider(FakeAgentModelProvider):
            def decide(self, state):
                captured_states.append(state)
                return AgentDecision(is_direct_answer=True, content="看到了历史。")

        provider = StateCapturingProvider()

        result = run_agent(
            question="那它怎么处理失败",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
            history=history,
        )

        assert result.memory_used is True
        assert result.memory_turns == 2
        assert "数据库在哪" in captured_states[0].memory_context
        assert "database.py" in captured_states[0].memory_context

    def test_no_history_results_in_memory_used_false(self, project_dir, fake_embedding):
        provider = FakeAgentModelProvider(
            decisions=[AgentDecision(is_direct_answer=True, content="无历史。")],
        )
        result = run_agent(
            question="test",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
            history=None,
        )

        assert result.memory_used is False
        assert result.memory_turns == 0


# ═══════════════════════ 9. retrieved_chunks ═══════════════════════

class TestRetrievedChunks:
    def test_search_code_produces_retrieved_chunks(self, project_dir):
        """search_code 工具调用后，result.retrieved_chunks 应包含检索结果。"""
        chunks = [
            _make_chunk("chunk-db", "def connect_db():\n    pass\n", "db.py"),
        ]
        emb = FakeEmbeddingProvider({
            chunks[0].text: [1.0, 0.0, 0.0],
            "database": [1.0, 0.0, 0.0],
        })
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="search_code",
                    arguments={"query": "database", "top_k": 1},
                ),
            ],
            final_answer="数据库在 db.py 中。",
        )
        result = run_agent(
            question="数据库在哪",
            project_root=project_dir,
            chunks=chunks,
            embedding_provider=emb,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0].metadata.get("file_path") == "db.py"
        assert result.retrieved_chunks[0].score > 0

    def test_list_files_does_not_produce_retrieved_chunks(self, project_dir, fake_embedding):
        """非 search_code 工具不产生 retrieved_chunks。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="list_files",
                    arguments={"file_pattern": "*.py"},
                ),
            ],
            final_answer="有 3 个文件。",
        )
        result = run_agent(
            question="有哪些文件",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.status == "completed"
        assert result.retrieved_chunks == []

    def test_search_code_defaults_top_k_when_model_omits_it(self, project_dir, fake_embedding):
        """真实模型漏传 top_k 时，search_code 使用默认值继续执行。"""
        provider = FakeAgentModelProvider(
            decisions=[
                AgentDecision(
                    is_direct_answer=False,
                    tool_name="search_code",
                    arguments={"query": "xyz"},
                ),
            ],
            final_answer="搜索完成。",
        )
        result = run_agent(
            question="搜索",
            project_root=project_dir,
            chunks=[],
            embedding_provider=fake_embedding,
            model_provider=provider,
        )

        assert result.retrieved_chunks == []
        assert result.errors == []
