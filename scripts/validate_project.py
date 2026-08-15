"""本地项目自检脚本。

用法:
    venv/Scripts/python.exe scripts/validate_project.py
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def check_step(name: str, passed: bool, detail: str = "") -> bool:
    """打印检查结果，并返回是否通过。"""
    status = "通过" if passed else "失败"
    print(f"  [{status}] {name}")
    if detail:
        print(f"       {detail}")
    return passed


def main() -> int:
    print("=== validate_project.py ===\n")
    all_ok = True

    check_files = [
        "README.md",
        "Dockerfile",
        ".dockerignore",
        "compose.yaml",
        ".env.example",
        "codebase_agent/backend/main.py",
        "codebase_agent/backend/database.py",
        "requirements.txt",
        ".gitignore",
    ]
    missing_list = []
    for rel_path in check_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            missing_list.append(rel_path)

    passed1 = len(missing_list) == 0
    detail1 = f"缺少文件: {', '.join(missing_list)}" if missing_list else "必需文件齐全"
    if not check_step("必需文件", passed1, detail1):
        all_ok = False

    sensitive_found = []
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        tracked_files = res.stdout.splitlines()
        for f in tracked_files:
            if f == ".env":
                sensitive_found.append(".env")
            elif f.endswith((".db", ".sqlite")):
                sensitive_found.append(f)
            elif "__pycache__" in f:
                sensitive_found.append(f)
    except Exception as e:
        sensitive_found.append(f"git 命令执行失败: {str(e)}")

    passed2 = len(sensitive_found) == 0
    detail2 = (
        f"已跟踪的敏感文件: {', '.join(sensitive_found)}"
        if sensitive_found
        else "未发现已跟踪的敏感文件"
    )
    if not check_step("Git 卫生", passed2, detail2):
        all_ok = False

    try:
        pip_res = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        passed3 = (pip_res.returncode == 0)
        detail3 = pip_res.stderr.strip() or pip_res.stdout.strip() or "未发现依赖冲突"
        if detail3 == "No broken requirements found.":
            detail3 = "未发现依赖冲突"
    except Exception as e:
        passed3 = False
        detail3 = f"pip check 执行失败: {str(e)}"

    if not check_step("Python 依赖", passed3, detail3):
        all_ok = False

    print("\n[下一步] 常用命令:")
    print("   (1) 运行测试: venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp .test-tmp/pytest")
    print("   (2) 启动 API: venv/Scripts/python.exe -m uvicorn codebase_agent.backend.main:app --reload")

    if all_ok:
        print("\n[全部通过] 自检通过，项目可以启动。")
        return 0
    else:
        print("\n[自检失败] 请先修复上面的问题。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
