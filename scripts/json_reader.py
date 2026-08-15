import argparse
import json
from collections import Counter


def load_repo_summary(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "repo_url": data.get("repo_url", ""),
        "total_files": data.get("total_files", 0),
        "files": data.get("files", []),
    }


def main():
    parser = argparse.ArgumentParser(description="读取仓库文件列表 JSON 并展示文件类型分布")
    parser.add_argument("json_file", help="JSON 文件路径")
    args = parser.parse_args()
    data = load_repo_summary(args.json_file)
    suffixes = [f["type"] for f in data["files"]]
    suffix_counts = Counter(suffixes)
    total_files = data.get("total_files", 0)

    print(f"仓库地址: {data.get('repo_url', 'N/A')}")
    print(f"文件总数: {total_files}")
    print("文件类型分布:")
    print(f"  {'类型':<20} {'数量':<10} 占比")
    print("  " + "-"*30)
    for suffix, count in suffix_counts.items():
        percentage = count / total_files * 100 if total_files else 0
        print(f"  {suffix:<20}: {count} {percentage:.1f}%")
    print("  " + "-"*30)


if __name__ == "__main__":
    main()
