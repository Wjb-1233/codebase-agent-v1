"""test_code_analyzer.py — 测试 code_analyzer 的正则提取和结构化输出。

覆盖场景：正常文件 / 空文件 / 无导入有函数 / 文件不存在
"""

import pytest

from codebase_agent.code_analyzer import analyze, extract_imports, extract_functions, extract_classes


# ============================================================
# 场景1: 正常 Python 文件（有 import、def、class）
# ============================================================

def test_analyze_normal_file(tmp_path):
    """正常 Python 文件 → 正确提取 imports、functions、classes"""
    file_path = tmp_path / "sample.py"
    file_path.write_text("import os\nfrom collections import Counter\ndef foo(): pass\ndef bar(): pass\nclass MyClass: pass\n", encoding="utf-8")
    result = analyze(str(file_path))
    assert len(result["imports"]) >= 2
    assert len(result["functions"]) >= 2
    assert len(result["classes"]) >= 1
    assert any(imp["type"] == "import" for imp in result["imports"])
    assert any(imp["type"] == "from" for imp in result["imports"])


# ============================================================
# 场景1 补充: 逐函数单元测试（可选，加固理解）
# ============================================================

def test_extract_imports_from_code():
    """直接测 extract_imports：from 和 import 都能识别"""
    code = """import os
from pathlib import Path
import sys, json
"""
    result = extract_imports(code)
    assert len(result) == 4
    assert result[0]["type"] == "import"
    assert result[0]["module"] == "os"
    assert result[1]["type"] == "from"
    assert result[1]["module"] == "pathlib"


def test_extract_functions_from_code():
    """直接测 extract_functions：能拿到函数名和行号"""
    code = """def foo():
    pass

def bar(x):
    return x + 1
"""
    result = extract_functions(code)
    assert len(result) == 2
    assert result[0]["name"] == "foo"
    assert result[1]["name"] == "bar"
    assert result[1]["line"] > result[0]["line"]

def test_extract_classes_from_code():
    """直接测 extract_classes：能拿到类名"""
    code = """class MyClass:
    pass

class AnotherClass(BaseClass):
    pass
"""
    result = extract_classes(code)
    assert len(result) == 2
    assert result[0]["name"] == "MyClass"
    assert result[1]["name"] == "AnotherClass"

# ============================================================
# 场景2: 空文件
# ============================================================

def test_analyze_empty_file(tmp_path):
    """空文件 → 三个列表都为空，不报错"""
    file_path = tmp_path / "empty.py"
    file_path.write_text("", encoding="utf-8")
    result = analyze(str(file_path))
    assert result["imports"] == []
    assert result["functions"] == []
    assert result["classes"] == []


# ============================================================
# 场景3: 无导入但只有函数的文件
# ============================================================

def test_analyze_no_imports_only_functions(tmp_path):
    """文件只有函数没有 import → imports 为空，functions 不为空"""
    file_path = tmp_path / "functions_only.py"
    file_path.write_text("def hello(): pass\ndef world(): pass\n", encoding="utf-8")
    result = analyze(str(file_path))
    assert result["imports"] == []
    assert len(result["functions"]) >= 1
    assert result["classes"] == []


# ============================================================
# 场景4: 文件不存在
# ============================================================

def test_analyze_file_not_found():
    """文件路径不存在 → 抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError) as exc_info:
        analyze("non_existent_file.py")


# ============================================================
# 选做: 边界场景 — 文件只有类没有函数
# ============================================================

def test_analyze_only_classes(tmp_path):
    """文件只有类没有函数 → classes 不为空，functions 为空"""
    file_path = tmp_path / "classes_only.py"
    file_path.write_text("class MyClass: pass\nclass AnotherClass: pass\n", encoding="utf-8")
    result = analyze(str(file_path))
    assert result["functions"] == []
    assert len(result["classes"]) >= 1
