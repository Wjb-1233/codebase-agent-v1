"""Agent 工具层单元测试。

覆盖：list_files / get_file_content / search_code / executor 成功 / executor 失败 / router / 安全 / 脱敏。
"""

from unittest.mock import MagicMock

import pytest

from codebase_agent.agent.executor import dispatch, get_registry, validate_args, check_path_safety
from codebase_agent.agent.models import ToolResult, ToolEvent
from codebase_agent.agent.router import route
from codebase_agent.agent.tools import get_file_content, list_files, search_code
from codebase_agent.exceptions import EmbeddingError
from codebase_agent.rag.chunker import Chunk


# ═══════════════════════ 共享 fixture ═══════════════════════

@pytest.fixture
def project_dir(tmp_path):
    """一个模拟项目目录。"""
    (tmp_path / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "client.py").write_text("class Client:\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    return str(tmp_path)


class FakeEmbeddingProvider:
    """返回确定性向量的 fake provider——隔离外部 API，不伪造检索排序。"""

    def __init__(self, vectors_by_text: dict[str, list[float]] | None = None):
        if vectors_by_text is not None:
            self.vectors_by_text = vectors_by_text
        else:
            self.vectors_by_text = {}
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        result = []
        for text in texts:
            lower = text.lower()
            if any(kw in lower for kw in ("database", "connect", "db", "sql")):
                result.append([1.0, 0.0, 0.0])
            elif any(kw in lower for kw in ("html", "render", "ui", "template")):
                result.append([0.0, 1.0, 0.0])
            else:
                result.append([0.3, 0.3, 0.4])  # default vector
        return result


@pytest.fixture
def fake_embedding_provider():
    return FakeEmbeddingProvider()


def _make_chunk(chunk_id: str, text: str, file_path: str = "app.py", symbol_name: str = "handler") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        file_path=file_path,
        start_line=1,
        end_line=3,
        symbol_name=symbol_name,
    )


# ═══════════════════════ 1. list_files ═══════════════════════

class TestListFiles:
    def test_returns_sorted_py_files(self, project_dir):
        result = list_files(project_dir)
        assert isinstance(result, list)
        assert len(result) >= 3
        assert result == sorted(result)  # 排序稳定
        assert "main.py" in result
        assert "backend/client.py" in result

    def test_filters_by_pattern(self, project_dir):
        result = list_files(project_dir, file_pattern="*.md")
        assert result == ["README.md"]

    def test_empty_for_nonexistent_dir(self):
        result = list_files("/nonexistent/path/xyz")
        assert result == []

    def test_empty_for_empty_dir(self, tmp_path):
        result = list_files(str(tmp_path))
        assert result == []


# ═══════════════════════ 2. get_file_content ═══════════════════════

class TestGetFileContent:
    def test_reads_existing_file(self, project_dir):
        content = get_file_content("main.py", project_dir)
        assert isinstance(content, str)
        assert "def main()" in content

    def test_raises_for_nonexistent_file(self, project_dir):
        with pytest.raises(FileNotFoundError, match="nonexistent.py"):
            get_file_content("nonexistent.py", project_dir)


# ═══════════════════════ 3. search_code（真实链路）══════════════════════

class TestSearchCode:
    def test_returns_relevant_chunk(self):  # 不需要 fake_embedding_provider fixture，手动创建更精确的向量
        chunks = [
            _make_chunk("chunk-db", "def connect_db():\n    return 'connected'\n", "db.py", "connect_db"),
            _make_chunk("chunk-ui", "def render_html():\n    return '<div/>'\n", "ui.py", "render_html"),
        ]
        provider = FakeEmbeddingProvider({
            chunks[0].text: [1.0, 0.0, 0.0],
            chunks[1].text: [0.0, 1.0, 0.0],
            "database connection": [1.0, 0.0, 0.0],
        })

        results = search_code(query="database connection", chunks=chunks, top_k=2, embedding_provider=provider)

        assert len(results) >= 1
        assert results[0].score > 0
        assert results[0].metadata["chunk_id"] == "chunk-db"
        assert results[0].metadata["file_path"] == "db.py"

    def test_returns_empty_for_no_chunks(self, fake_embedding_provider):
        results = search_code(query="test", chunks=[], top_k=5, embedding_provider=fake_embedding_provider)
        assert results == []

    def test_rejects_empty_query(self, fake_embedding_provider):
        with pytest.raises(ValueError, match="query 不能为空"):
            search_code(query="   ", chunks=[], top_k=5, embedding_provider=fake_embedding_provider)


# ═══════════════════════ 4. executor 成功路径 ═══════════════════════

class TestExecutorSuccess:
    def test_dispatch_list_files(self, project_dir):
        result, event = dispatch("list_files", {}, project_dir)

        assert result.success is True
        assert isinstance(result.output, list)
        assert len(result.output) >= 3
        assert result.output == sorted(result.output)

        assert event.success is True
        assert event.tool_name == "list_files"
        assert event.duration_ms > 0
        assert event.trace_id != ""
        assert event.output_summary is not None

    def test_dispatch_get_file_content(self, project_dir):
        result, event = dispatch("get_file_content", {"path": "main.py"}, project_dir)

        assert result.success is True
        assert "def main()" in str(result.output)

        assert event.success is True
        assert event.tool_name == "get_file_content"

    def test_dispatch_search_code(self, fake_embedding_provider):
        chunks = [
            _make_chunk("chunk-a", "def connect_db():\n    pass\n", "db.py"),
            _make_chunk("chunk-b", "def render_ui():\n    pass\n", "ui.py"),
        ]
        provider = FakeEmbeddingProvider({
            chunks[0].text: [1.0, 0.0, 0.0],
            chunks[1].text: [0.0, 1.0, 0.0],
            "database": [1.0, 0.0, 0.0],
        })

        result, event = dispatch(
            "search_code",
            {"query": "database", "top_k": 1},
            project_root="/fake/project",
            chunks=chunks,
            embedding_provider=provider,
        )

        assert result.success is True
        assert len(result.output) == 1
        assert result.output[0].metadata["chunk_id"] == "chunk-a"

        assert event.success is True
        assert event.tool_name == "search_code"


# ═══════════════════════ 5. executor 失败路径 ═══════════════════════

class TestExecutorFailure:
    def test_unknown_tool(self, project_dir):
        result, event = dispatch("nonexistent_tool", {}, project_dir)

        assert result.success is False
        assert result.error_type == "unknown_tool"

        assert event.success is False
        assert event.error_type == "unknown_tool"
        assert event.trace_id != ""

    def test_missing_required_param(self, project_dir):
        result, event = dispatch("get_file_content", {}, project_dir)

        assert result.success is False
        assert result.error_type == "invalid_argument"
        assert "缺少必填参数" in result.error_message

        assert event.success is False
        assert event.error_type == "invalid_argument"

    def test_wrong_param_type(self, project_dir):
        result, event = dispatch("search_code", {"query": 123}, project_dir)

        assert result.success is False
        assert result.error_type == "invalid_argument"
        assert "字符串" in result.error_message or "缺少必填参数" in result.error_message or "未知参数" in result.error_message

    def test_path_traversal_rejected(self, project_dir):
        result, event = dispatch("get_file_content", {"path": "../.env"}, project_dir)

        assert result.success is False
        assert result.error_type == "permission_denied"

        # 事件中不应包含完整越权路径
        input_str = str(event.input_summary)
        assert "../.env" not in input_str

    def test_deep_path_traversal_rejected(self, project_dir):
        """../../.. 多层越权也拒绝。"""
        result, event = dispatch("get_file_content", {"path": "../../../etc/passwd"}, project_dir)

        assert result.success is False
        assert result.error_type == "permission_denied"

    def test_circular_path_traversal_rejected(self, project_dir):
        """background/../../.env —— 等价于 ../.env，规范化后越权。"""
        result, event = dispatch("get_file_content", {"path": "background/../../.env"}, project_dir)

        assert result.success is False
        assert result.error_type == "permission_denied"

    def test_file_not_found(self, project_dir):
        result, event = dispatch("get_file_content", {"path": "nonexistent.py"}, project_dir)

        assert result.success is False
        assert result.error_type == "FileNotFoundError"

        assert event.success is False

    def test_embedding_provider_error(self, fake_embedding_provider):
        provider = MagicMock()
        provider.embed_texts.side_effect = EmbeddingError("simulated failure")

        chunks = [_make_chunk("chunk-a", "def foo(): pass\n", "a.py")]

        result, event = dispatch(
            "search_code",
            {"query": "foo", "top_k": 1},
            project_root="/fake/project",
            chunks=chunks,
            embedding_provider=provider,
        )

        assert result.success is False
        assert result.error_type is not None
        assert event.success is False

    def test_unknown_param_rejected(self, project_dir):
        """传入工具未声明的参数 → 校验失败。"""
        result, event = dispatch("list_files", {"bad_param": 42}, project_dir)

        assert result.success is False
        assert result.error_type == "invalid_argument"
        assert "未知参数" in result.error_message


# ═══════════════════════ 6. 路由测试 ═══════════════════════

class TestRouter:
    def test_route_list_files(self):
        cases = [
            "项目有哪些 Python 文件",
            "列出所有代码文件",
            "看看项目结构",
        ]
        for q in cases:
            r = route(q)
            assert r.tool_name == "list_files", f"Failed for: {q}"

    def test_route_get_file_content(self):
        cases = [
            "打开 backend/main.py",
            "读取 utils.py 的内容",
            "看看 client.py",
        ]
        for q in cases:
            r = route(q)
            assert r.tool_name == "get_file_content", f"Failed for: {q}"
            if "utils.py" in q:
                assert r.arguments["path"] == "utils.py"

    def test_route_search_code(self):
        cases = [
            "认证逻辑在哪",
            "数据库连接怎么实现的",
            "FastAPI 路由在哪里定义",
            "代码分析",
        ]
        for q in cases:
            r = route(q)
            assert r.tool_name == "search_code", f"Failed for: {q}"
            assert "top_k" in r.arguments

    def test_route_greeting_to_direct_answer(self):
        for q in ["你好", "hi", "早上好"]:
            r = route(q)
            assert r.tool_name == "direct_answer"
            assert r.is_direct_answer is True

    def test_route_empty_to_direct_answer(self):
        r = route("")
        assert r.tool_name == "direct_answer"
        assert r.is_direct_answer is True

    def test_route_unknown_to_direct_answer(self):
        """不匹配任何工具时兜底到 direct_answer。"""
        r = route("What is the capital of France")
        assert r.tool_name == "direct_answer"
        assert r.is_direct_answer is True


# ═══════════════════════ 7. 事件脱敏 ═══════════════════════

class TestEventSanitization:
    def test_event_does_not_leak_full_file_content(self, project_dir):
        """读取文件后，event.output_summary 应为摘要而非完整内容。"""
        # 写入一个超过 120 字符的文件
        long_file = project_dir + "/long_file.py"
        with open(long_file, "w", encoding="utf-8") as f:
            f.write("# " + "x" * 200 + "\ndef long_function():\n    pass\n")

        result, event = dispatch("get_file_content", {"path": "long_file.py"}, project_dir)

        assert result.success is True
        # output_summary 不应包含完整的长内容
        assert event.output_summary is not None
        assert "文本" in event.output_summary or len(str(event.output_summary)) < 200

    def test_event_summary_not_full_search_results(self, fake_embedding_provider):
        """search_code 事件摘要不应泄露完整 chunk 文本。"""
        chunks = [
            _make_chunk("chunk-x", "def very_specific_function_name():\n    return 'secret'\n", "secret.py"),
        ]
        provider = FakeEmbeddingProvider({
            chunks[0].text: [1.0, 0.0, 0.0],
            "very specific": [1.0, 0.0, 0.0],
        })

        result, event = dispatch(
            "search_code",
            {"query": "very specific", "top_k": 1},
            project_root="/fake",
            chunks=chunks,
            embedding_provider=provider,
        )

        assert result.success is True
        assert event.output_summary is not None
        # 不应包含完整函数体
        assert "secret" not in str(event.output_summary)

    def test_event_path_summary_strips_directories(self):
        """path 摘要只保留文件名，不保留完整目录路径。"""
        result, event = dispatch(
            "get_file_content",
            {"path": "nested/deep/main.py"},
            project_root="/fake/project",
        )

        assert event.input_summary is not None
        path_value = event.input_summary.get("path", "")
        assert "nested" not in str(path_value)  # 应该只剩 "main.py"
        assert "main.py" in str(path_value)
