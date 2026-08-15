from codebase_agent.code_graph import CodeGraphInputFile, build_code_graph


def test_build_code_graph_extracts_symbols_and_internal_imports():
    graph = build_code_graph(
        [
            CodeGraphInputFile(
                file_path="app/main.py",
                content=(
                    "from app.service import UserService\n"
                    "import os\n\n"
                    "async def health():\n"
                    "    return {'status': 'ok'}\n\n"
                    "class App:\n"
                    "    def run(self):\n"
                    "        return UserService()\n"
                ),
            ),
            CodeGraphInputFile(
                file_path="app/service.py",
                content="class UserService:\n    pass\n",
            ),
        ]
    )

    labels = {node.label for node in graph.nodes}
    assert "app/main.py" in labels
    assert "health" in labels
    assert "App" in labels
    assert "App.run" in labels
    assert "UserService" in labels

    internal_edges = [
        edge for edge in graph.edges
        if edge.type == "imports" and edge.target == "file:app/service.py"
    ]
    external_edges = [
        edge for edge in graph.edges
        if edge.type == "imports" and edge.target == "external:os"
    ]
    assert internal_edges
    assert external_edges
    assert graph.summary["files"] == 2
    assert graph.summary["python_files"] == 2
    assert graph.summary["internal_imports"] == 1


def test_build_code_graph_keeps_parse_errors_structured():
    graph = build_code_graph(
        [
            CodeGraphInputFile(
                file_path="broken.py",
                content="def broken(:\n    pass\n",
            )
        ]
    )

    assert graph.summary["parse_errors"] == 1
    assert graph.files[0].parse_error.startswith("SyntaxError:")
    assert any(node.id == "file:broken.py" for node in graph.nodes)


def test_build_code_graph_keeps_non_python_files_as_file_nodes():
    graph = build_code_graph(
        [
            CodeGraphInputFile(file_path="README.md", content="# Docs\n"),
            CodeGraphInputFile(file_path="config.json", content='{"debug": true}'),
        ]
    )

    assert graph.summary["files"] == 2
    assert graph.summary["python_files"] == 0
    assert {file.language for file in graph.files} == {"markdown", "json"}
    assert {node.id for node in graph.nodes} == {"file:README.md", "file:config.json"}


def test_build_code_graph_extracts_javascript_and_typescript_baseline():
    graph = build_code_graph(
        [
            CodeGraphInputFile(
                file_path="src/App.jsx",
                content=(
                    "import { helper } from './utils'\n"
                    "import React from 'react'\n\n"
                    "export function App() { return helper() }\n"
                    "const loadData = async () => helper()\n"
                    "class Dashboard {}\n"
                ),
            ),
            CodeGraphInputFile(
                file_path="src/utils.ts",
                content="export const helper = () => 'ok'\n",
            ),
        ]
    )

    labels = {node.label for node in graph.nodes}
    assert "App" in labels
    assert "loadData" in labels
    assert "Dashboard" in labels
    assert "helper" in labels
    assert graph.summary["javascript_files"] == 1
    assert graph.summary["typescript_files"] == 1
    assert graph.summary["internal_imports"] == 1
    assert graph.summary["external_imports"] == 1
    assert any(
        edge.type == "imports" and edge.source == "file:src/App.jsx" and edge.target == "file:src/utils.ts"
        for edge in graph.edges
    )


def test_build_code_graph_extracts_java_and_go_baseline():
    graph = build_code_graph(
        [
            CodeGraphInputFile(
                file_path="com/example/App.java",
                content=(
                    "package com.example;\n"
                    "import com.example.Service;\n"
                    "import java.util.List;\n\n"
                    "public class App {\n"
                    "  public void run() {}\n"
                    "}\n"
                ),
            ),
            CodeGraphInputFile(
                file_path="com/example/Service.java",
                content="public interface Service {}\n",
            ),
            CodeGraphInputFile(
                file_path="cmd/server.go",
                content=(
                    "package main\n\n"
                    "import (\n"
                    "  \"fmt\"\n"
                    "  \"net/http\"\n"
                    ")\n\n"
                    "func main() {}\n"
                    "func (s Server) Start() {}\n"
                ),
            ),
        ]
    )

    labels = {node.label for node in graph.nodes}
    assert {"App", "run", "Service", "main", "Start"}.issubset(labels)
    assert graph.summary["java_files"] == 2
    assert graph.summary["go_files"] == 1
    assert graph.summary["supported_language_files"] == 3
    assert graph.summary["internal_imports"] == 1
    assert any(
        edge.type == "imports" and edge.source == "file:com/example/App.java" and edge.target == "file:com/example/Service.java"
        for edge in graph.edges
    )
