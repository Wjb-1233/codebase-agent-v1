"""文本文件统计命令行工具。

用法:
  python scripts/file_stats.py data/sample.txt
  python scripts/file_stats.py data/sample.txt --top 20
  python scripts/file_stats.py data/sample.txt --no-words
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="分析文本文件的基础统计信息（行数、单词数、字符数、高频词）"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="要分析的文件路径",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="显示前 N 个高频词（默认 10）",
    )
    parser.add_argument(
        "--no-words",
        action="store_true",
        help="不统计单词（只看行数、字符数、文件大小）",
    )
    return parser.parse_args()


def check_file(filepath: str) -> Path:
    """检查文件是否存在，返回 Path 对象"""
    path = Path(filepath)

    path = path.expanduser().resolve()

    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)

    if not path.is_file():
        print(f"❌ 不是普通文件: {path}")
        sys.exit(1)

    return path


def is_binary_file(path: Path, sample_size: int = 1024) -> bool:
    """快速判断是否为二进制文件：读取前 1024 字节，找空字符"""
    with open(path, "rb") as f:
        chunk = f.read(sample_size)
    return b"\x00" in chunk


def count_lines(path: Path) -> int:
    """统计文件行数"""
    with open(path, encoding="utf-8", errors="replace") as f:
        return len([line for line in f])


def count_words(text: str) -> int:
    """统计单词数（按空白字符分割，过滤空字符串）"""
    words = text.split()
    return len(words)


def count_chars(text: str) -> int:
    """统计字符数（含空格和换行）"""
    return len(text)


def get_file_size(path: Path) -> str:
    """获取文件大小，自动选择合适的单位"""
    size_bytes = path.stat().st_size

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_top_words(text: str, top_n: int = 10) -> list[tuple[str, int]]:
    """统计高频词，返回前 N 个结果"""
    words = text.lower().split()

    alpha_words = [w.strip(".,!?;:()[]{}\"'") for w in words]
    alpha_words = [w for w in alpha_words if w.isalpha()]

    counter = Counter(alpha_words)
    return counter.most_common(top_n)


def detect_file_type(path: Path) -> str:
    """推测文件类型"""
    suffix = path.suffix.lower()

    type_map = {
        ".py": "Python 脚本",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".html": "HTML",
        ".css": "CSS",
        ".json": "JSON 数据",
        ".yaml": "YAML 配置",
        ".yml": "YAML 配置",
        ".md": "Markdown 文档",
        ".txt": "纯文本",
        ".csv": "CSV 表格",
        ".xml": "XML",
        ".toml": "TOML 配置",
        ".ini": "INI 配置",
        ".cfg": "配置文件",
        ".log": "日志文件",
        ".sh": "Shell 脚本",
        ".sql": "SQL 脚本",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".rs": "Rust",
        ".go": "Go",
    }

    return type_map.get(suffix, f"未知类型 ({suffix or '无后缀'})")


def output_basic_stats(path: Path, text: str) -> None:
    """输出基本统计信息"""
    print("=" * 50)
    print(f"📄 文件: {path.name}")
    print(f"📁 路径: {path}")
    print(f"🏷️  类型: {detect_file_type(path)}")
    print(f"💾 大小: {get_file_size(path)}")
    print("-" * 50)

    lines = count_lines(path)
    words = count_words(text)
    chars = count_chars(text)

    print(f"📏 行数:   {lines:,}")
    print(f"🔤 单词数: {words:,}")
    print(f"🔡 字符数: {chars:,}")
    print(f"📊 平均每行单词: {words / max(lines, 1):.1f}")


def output_top_words(path: Path, top_n: int) -> None:
    """输出高频词"""
    print("-" * 50)
    print(f"🏆 前 {top_n} 个高频词:")

    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    top_words = get_top_words(text, top_n)

    if not top_words:
        print("  （没有找到纯字母单词）")
        return

    for rank, (word, count) in enumerate(top_words, start=1):
        bar = "█" * min(count, 40)
        print(f"  {rank:>2}. {word:<20} {count:>5}  {bar}")


def main() -> None:
    args = parse_args()
    path = check_file(args.filepath)

    try:
        if is_binary_file(path):
            print("=" * 50)
            print(f"📄 文件: {path.name}")
            print(f"📁 路径: {path}")
            print(f"💾 大小: {get_file_size(path)}")
            print("-" * 50)
            print("⚠️  检测为二进制文件，无法统计行数/单词/字符。")
            return

        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        if not text.strip():
            print("=" * 50)
            print(f"📄 文件: {path.name}")
            print(f"📁 路径: {path}")
            print(f"💾 大小: {get_file_size(path)}")
            print("-" * 50)
            print("📭 空文件，没有内容可以分析。")
            return

        output_basic_stats(path, text)

        if not args.no_words:
            output_top_words(path, args.top)

    except PermissionError:
        print(f"❌ 没有权限读取文件: {path}")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"❌ 文件编码无法识别（可能不是文本文件）: {path}")
        sys.exit(1)
    except Exception as e:
        # 捕获其他所有异常，打印具体错误信息
        print(f"❌ 读取文件时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
