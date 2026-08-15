"""code_analyzer.py — 用正则提取 Python 文件的 import/函数/类。

用法: python -m codebase_agent.code_analyzer <file_path>
输出: 结构化 dict {imports, functions, classes}
"""

import re
import sys
from pathlib import Path
from typing import Any
import json

def read_file(path: str) -> str:
    """读取文件内容，文件不存在时抛出 FileNotFoundError。"""
    return Path(path).read_text(encoding="utf-8")


def extract_imports(code: str) -> list[dict[str, Any]]:
    """从代码中提取所有 import 和 from ... import 语句。

    返回格式:
        [
            {"line": 3, "type": "import", "module": "os"},
            {"line": 5, "type": "from", "module": "collections", "name": "Counter"},
        ]

    关键: 按行遍历，先判断是否 from 行，再判断是否 import 行，
    避免 import pattern 误吃 from 行里的 import。
    """
    result = []
    for line_num, line in enumerate(code.splitlines(), start=1):
        line = line.lstrip()
        if not line:
            continue
        from_match = re.match(r"from\s+(\S+)\s+import\s+(.+)", line)
        if from_match:
            module, names = from_match.groups()
            name_list = [name.strip() for name in names.split(",")]
            for name in name_list:
                result.append({"line": line_num, "type": "from", "module": module, "name": name})
            continue
        import_match = re.match(r"import\s+(.+)", line)
        if import_match:
            modules = import_match.group(1).split(",")
            for module in modules:
                result.append({"line": line_num, "type": "import", "module": module.strip()})
    return result


def extract_functions(code: str) -> list[dict[str, Any]]:
    """提取所有函数定义。

    返回格式: [{"line": 10, "name": "main"}, ...]

    正则思路: def 后面至少一个空格，然后函数名（字母数字下划线），然后左括号。
    注意: 前面可能有缩进空格，所以不要用 ^ 锚定。
    """
    result = []
    for m in re.finditer(r"def\s+(\w+)\s*\(", code):
        func_name = m.group(1)
        line_num = code[:m.start()].count('\n') + 1
        result.append({"line": line_num, "name": func_name})
    return result

def extract_classes(code: str) -> list[dict[str, Any]]:
    """提取所有类定义。

    返回格式: [{"line": 8, "name": "MyClass"}, ...]

    正则思路: class 后面至少一个空格，然后类名，然后冒号或左括号。
    """
    result = []
    for m in re.finditer(r"class\s+(\w+)\s*[:(]", code):
        class_name = m.group(1)
        line_num = code[:m.start()].count('\n') + 1
        result.append({"line": line_num, "name": class_name})
    return result


def analyze(file_path: str) -> dict[str, Any]:
    """主入口：读文件 → 提取 → 返回结构化结果。"""
    code = read_file(file_path)
    imports = extract_imports(code)
    functions = extract_functions(code)
    classes = extract_classes(code)
    return {"imports": imports, "functions": functions, "classes": classes}


def main() -> None:
    """CLI 入口：接收文件路径参数，打印统计和结构化结果。"""
    if len(sys.argv) < 2:
        print("用法: python -m codebase_agent.code_analyzer <file_path>")
        sys.exit(1)
    try:
        result = analyze(sys.argv[1])
    except FileNotFoundError:
        print("错误: 文件不存在")
        sys.exit(1)
    print(f"文件路径: {sys.argv[1]}")
    print(f"函数数: {len(result['functions'])}")
    print(f"类数: {len(result['classes'])}")
    print(f"导入数: {len(result['imports'])}")
    print("结构化结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
