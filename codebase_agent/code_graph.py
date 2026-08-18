"""从请求范围内的文件构建轻量代码结构图。

Python 文件使用 ``ast`` 解析。JavaScript/TypeScript、Java 和 Go 使用小型、
确定性的解析规则，这样控制台不用引入重型解析器依赖，也能展示有用的
多语言结构视图。这里做的是静态结构分析，不是运行时调用链分析。
"""

from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class CodeGraphNode:
    id: str
    label: str
    type: str
    file_path: str = ""
    line: int = 0


@dataclass(frozen=True)
class CodeGraphEdge:
    source: str
    target: str
    type: str
    label: str = ""


@dataclass(frozen=True)
class CodeGraphImport:
    file_path: str
    module: str
    line: int
    target_file: str = ""
    is_internal: bool = False


@dataclass(frozen=True)
class CodeGraphFile:
    file_path: str
    language: str
    symbols: list[CodeGraphNode]
    imports: list[CodeGraphImport]
    parse_error: str = ""


@dataclass(frozen=True)
class CodeGraph:
    nodes: list[CodeGraphNode]
    edges: list[CodeGraphEdge]
    files: list[CodeGraphFile]
    summary: dict[str, int]


@dataclass(frozen=True)
class CodeGraphInputFile:
    file_path: str
    content: str


def build_code_graph(files: list[CodeGraphInputFile]) -> CodeGraph:
    """解析文件并返回稳定的结构/依赖图。"""
    known_modules = _build_module_index(files)
    nodes_by_id: dict[str, CodeGraphNode] = {}
    edges_by_key: dict[tuple[str, str, str, str], CodeGraphEdge] = {}
    graph_files: list[CodeGraphFile] = []

    for file in sorted(files, key=lambda item: item.file_path):
        file_path = _normalize_path(file.file_path)
        file_node = CodeGraphNode(
            id=_file_node_id(file_path),
            label=file_path,
            type="file",
            file_path=file_path,
        )
        nodes_by_id[file_node.id] = file_node

        language = _guess_language(file_path)
        symbols, imports, parse_error = _analyze_file(file_path, file.content, language, known_modules)
        for symbol in symbols:
            nodes_by_id[symbol.id] = symbol
            _add_edge(edges_by_key, file_node.id, symbol.id, "contains", "contains")

        for import_item in imports:
            if import_item.is_internal and import_item.target_file:
                target_id = _file_node_id(import_item.target_file)
                nodes_by_id.setdefault(
                    target_id,
                    CodeGraphNode(
                        id=target_id,
                        label=import_item.target_file,
                        type="file",
                        file_path=import_item.target_file,
                    ),
                )
            else:
                target_id = _external_node_id(import_item.module)
                nodes_by_id.setdefault(
                    target_id,
                    CodeGraphNode(
                        id=target_id,
                        label=import_item.module,
                        type="external",
                    ),
                )
            _add_edge(edges_by_key, file_node.id, target_id, "imports", import_item.module)

        graph_files.append(
            CodeGraphFile(
                file_path=file_path,
                language=language,
                symbols=symbols,
                imports=imports,
                parse_error=parse_error,
            )
        )

    nodes = sorted(nodes_by_id.values(), key=lambda item: (item.type, item.id))
    edges = sorted(edges_by_key.values(), key=lambda item: (item.source, item.type, item.target, item.label))
    summary = {
        "files": len(files),
        "python_files": sum(1 for item in graph_files if item.language == "python"),
        "javascript_files": sum(1 for item in graph_files if item.language == "javascript"),
        "typescript_files": sum(1 for item in graph_files if item.language == "typescript"),
        "java_files": sum(1 for item in graph_files if item.language == "java"),
        "go_files": sum(1 for item in graph_files if item.language == "go"),
        "supported_language_files": sum(1 for item in graph_files if _is_supported_language(item.language)),
        "symbols": sum(1 for item in nodes if item.type in {"function", "async_function", "class", "method"}),
        "imports": sum(1 for item in edges if item.type == "imports"),
        "internal_imports": sum(
            1
            for item in graph_files
            for import_item in item.imports
            if import_item.is_internal
        ),
        "external_imports": sum(
            1
            for item in graph_files
            for import_item in item.imports
            if not import_item.is_internal
        ),
        "parse_errors": sum(1 for item in graph_files if item.parse_error),
    }
    return CodeGraph(nodes=nodes, edges=edges, files=graph_files, summary=summary)


def _analyze_file(
    file_path: str,
    content: str,
    language: str,
    known_modules: dict[str, str],
) -> tuple[list[CodeGraphNode], list[CodeGraphImport], str]:
    if language == "python":
        return _analyze_python_file(file_path, content, known_modules)
    if language in {"javascript", "typescript"}:
        return _analyze_javascript_like_file(file_path, content, known_modules)
    if language == "java":
        return _analyze_java_file(file_path, content, known_modules)
    if language == "go":
        return _analyze_go_file(file_path, content, known_modules)
    return [], [], ""


def _analyze_python_file(
    file_path: str,
    content: str,
    known_modules: dict[str, str],
) -> tuple[list[CodeGraphNode], list[CodeGraphImport], str]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return [], [], f"SyntaxError: {exc.msg} at line {exc.lineno or 0}"

    symbols = _extract_symbols(file_path, tree)
    imports = _extract_imports(file_path, tree, known_modules)
    return symbols, imports, ""


def _analyze_javascript_like_file(
    file_path: str,
    content: str,
    known_modules: dict[str, str],
) -> tuple[list[CodeGraphNode], list[CodeGraphImport], str]:
    symbols: list[CodeGraphNode] = []
    imports: list[CodeGraphImport] = []
    for line_number, line in _iter_lines(content):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        import_module = _match_javascript_import(stripped)
        if import_module:
            imports.append(_build_import(file_path, import_module, line_number, known_modules))

        class_match = re.search(r"\bclass\s+([A-Za-z_$][\w$]*)", stripped)
        if class_match:
            symbols.append(_symbol_node(file_path, class_match.group(1), "class", line_number))

        function_match = re.search(
            r"\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
            stripped,
        )
        if function_match:
            node_type = "async_function" if "async" in stripped[: function_match.start(1)] else "function"
            symbols.append(_symbol_node(file_path, function_match.group(1), node_type, line_number))
            continue

        arrow_match = re.search(
            r"\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
            stripped,
        )
        if arrow_match:
            node_type = "async_function" if "async" in stripped[: arrow_match.end()] else "function"
            symbols.append(_symbol_node(file_path, arrow_match.group(1), node_type, line_number))

    return _dedupe_nodes(symbols), _dedupe_imports(imports), ""


def _analyze_java_file(
    file_path: str,
    content: str,
    known_modules: dict[str, str],
) -> tuple[list[CodeGraphNode], list[CodeGraphImport], str]:
    symbols: list[CodeGraphNode] = []
    imports: list[CodeGraphImport] = []
    for line_number, line in _iter_lines(content):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        import_match = re.match(r"import\s+(?:static\s+)?([\w.]+)\s*;", stripped)
        if import_match:
            imports.append(_build_import(file_path, import_match.group(1), line_number, known_modules))

        type_match = re.search(r"\b(class|interface|enum)\s+([A-Za-z_]\w*)", stripped)
        if type_match:
            symbols.append(_symbol_node(file_path, type_match.group(2), "class", line_number))

        method_match = re.search(
            r"\b(?:public|private|protected|static|final|synchronized|abstract|native|\s)+"
            r"[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{",
            stripped,
        )
        if method_match and method_match.group(1) not in {"if", "for", "while", "switch", "catch"}:
            symbols.append(_symbol_node(file_path, method_match.group(1), "method", line_number))

    return _dedupe_nodes(symbols), _dedupe_imports(imports), ""


def _analyze_go_file(
    file_path: str,
    content: str,
    known_modules: dict[str, str],
) -> tuple[list[CodeGraphNode], list[CodeGraphImport], str]:
    symbols: list[CodeGraphNode] = []
    imports: list[CodeGraphImport] = []
    in_import_block = False
    for line_number, line in _iter_lines(content):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        if stripped.startswith("import ("):
            in_import_block = True
            continue
        if in_import_block and stripped == ")":
            in_import_block = False
            continue
        if in_import_block:
            module = _match_go_import_value(stripped)
            if module:
                imports.append(_build_import(file_path, module, line_number, known_modules))
            continue

        single_import = re.match(r'import\s+(?:[\w.]+\s+)?["`]([^"`]+)["`]', stripped)
        if single_import:
            imports.append(_build_import(file_path, single_import.group(1), line_number, known_modules))

        method_match = re.match(r"func\s+\([^)]+\)\s+([A-Za-z_]\w*)\s*\(", stripped)
        if method_match:
            symbols.append(_symbol_node(file_path, method_match.group(1), "method", line_number))
            continue

        function_match = re.match(r"func\s+([A-Za-z_]\w*)\s*\(", stripped)
        if function_match:
            symbols.append(_symbol_node(file_path, function_match.group(1), "function", line_number))

    return _dedupe_nodes(symbols), _dedupe_imports(imports), ""


def _extract_symbols(file_path: str, tree: ast.AST) -> list[CodeGraphNode]:
    symbols: list[CodeGraphNode] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef):
            class_node = CodeGraphNode(
                id=_symbol_node_id(file_path, node.name),
                label=node.name,
                type="class",
                file_path=file_path,
                line=node.lineno,
            )
            symbols.append(class_node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = f"{node.name}.{child.name}"
                    symbols.append(
                        CodeGraphNode(
                            id=_symbol_node_id(file_path, method_name),
                            label=method_name,
                            type="method",
                            file_path=file_path,
                            line=child.lineno,
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                CodeGraphNode(
                    id=_symbol_node_id(file_path, node.name),
                    label=node.name,
                    type="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                    file_path=file_path,
                    line=node.lineno,
                )
            )
    return symbols


def _extract_imports(
    file_path: str,
    tree: ast.AST,
    known_modules: dict[str, str],
) -> list[CodeGraphImport]:
    imports: list[CodeGraphImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(_build_import(file_path, alias.name, node.lineno, known_modules))
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from_module(file_path, node.level, node.module or "")
            if module:
                imports.append(_build_import(file_path, module, node.lineno, known_modules))
    return imports


def _build_import(
    file_path: str,
    module: str,
    line: int,
    known_modules: dict[str, str],
) -> CodeGraphImport:
    target_file = _resolve_import_target(file_path, module, known_modules)
    return CodeGraphImport(
        file_path=file_path,
        module=module,
        line=line,
        target_file=target_file,
        is_internal=bool(target_file),
    )


def _build_module_index(files: list[CodeGraphInputFile]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for file in files:
        file_path = _normalize_path(file.file_path)
        if not _is_supported_language(_guess_language(file_path)):
            continue
        path_without_suffix = _strip_known_suffix(file_path)
        slash_module = path_without_suffix
        dotted_module = path_without_suffix.replace("/", ".")
        modules[slash_module] = file_path
        modules[dotted_module] = file_path
        if dotted_module.endswith(".__init__"):
            modules[dotted_module.removesuffix(".__init__")] = file_path
        if slash_module.endswith("/index"):
            modules[slash_module.removesuffix("/index")] = file_path
        if dotted_module.endswith(".index"):
            modules[dotted_module.removesuffix(".index")] = file_path
    return modules


def _resolve_import_target(file_path: str, module: str, known_modules: dict[str, str]) -> str:
    if module.startswith("."):
        parent = PurePosixPath(file_path).parent.as_posix()
        candidate = posixpath.normpath(posixpath.join(parent, module))
        return known_modules.get(candidate, "") or known_modules.get(candidate.replace("/", "."), "")
    return _resolve_internal_module(module, known_modules)


def _resolve_internal_module(module: str, known_modules: dict[str, str]) -> str:
    current = module
    while current:
        if current in known_modules:
            return known_modules[current]
        if "." not in current:
            break
        current = current.rsplit(".", 1)[0]
    return ""


def _resolve_from_module(file_path: str, level: int, module: str) -> str:
    if level <= 0:
        return module

    current_module = file_path[:-3].replace("/", ".") if file_path.endswith(".py") else file_path.replace("/", ".")
    package_parts = current_module.split(".")[:-1]
    keep = max(len(package_parts) - level + 1, 0)
    base_parts = package_parts[:keep]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(part for part in base_parts if part)


def _normalize_path(file_path: str) -> str:
    return PurePosixPath(file_path.replace("\\", "/")).as_posix().lstrip("/")


def _guess_language(file_path: str) -> str:
    suffix = PurePosixPath(file_path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".java": "java",
        ".go": "go",
    }.get(suffix, "text")


def _is_supported_language(language: str) -> bool:
    return language in {"python", "javascript", "typescript", "java", "go"}


def _strip_known_suffix(file_path: str) -> str:
    suffix = PurePosixPath(file_path).suffix
    return file_path[: -len(suffix)] if suffix else file_path


def _iter_lines(content: str):
    yield from enumerate(content.splitlines(), start=1)


def _match_javascript_import(line: str) -> str:
    from_match = re.match(r"import\s+.+?\s+from\s+['\"]([^'\"]+)['\"]", line)
    if from_match:
        return from_match.group(1)
    side_effect_match = re.match(r"import\s+['\"]([^'\"]+)['\"]", line)
    if side_effect_match:
        return side_effect_match.group(1)
    require_match = re.search(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", line)
    if require_match:
        return require_match.group(1)
    return ""


def _match_go_import_value(line: str) -> str:
    match = re.match(r'(?:[\w.]+\s+)?["`]([^"`]+)["`]', line)
    return match.group(1) if match else ""


def _symbol_node(file_path: str, symbol_name: str, node_type: str, line: int) -> CodeGraphNode:
    return CodeGraphNode(
        id=_symbol_node_id(file_path, symbol_name),
        label=symbol_name,
        type=node_type,
        file_path=file_path,
        line=line,
    )


def _dedupe_nodes(nodes: list[CodeGraphNode]) -> list[CodeGraphNode]:
    return list({node.id: node for node in nodes}.values())


def _dedupe_imports(imports: list[CodeGraphImport]) -> list[CodeGraphImport]:
    return list({(item.file_path, item.module, item.line): item for item in imports}.values())


def _file_node_id(file_path: str) -> str:
    return f"file:{file_path}"


def _symbol_node_id(file_path: str, symbol_name: str) -> str:
    return f"symbol:{file_path}:{symbol_name}"


def _external_node_id(module: str) -> str:
    return f"external:{module}"


def _add_edge(
    edges: dict[tuple[str, str, str, str], CodeGraphEdge],
    source: str,
    target: str,
    edge_type: str,
    label: str,
) -> None:
    edges[(source, target, edge_type, label)] = CodeGraphEdge(
        source=source,
        target=target,
        type=edge_type,
        label=label,
    )
