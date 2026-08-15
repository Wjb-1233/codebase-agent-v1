"""HTTP API 连通性测试工具。"""

import argparse
import logging
import sys
from urllib.parse import urlparse

import requests


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

STATUS_MAP = {
    200: "成功",
    301: "永久重定向",
    302: "临时重定向",
    400: "请求错误",
    401: "未授权",
    403: "禁止访问",
    404: "未找到",
    500: "服务器内部错误",
    502: "网关错误",
    503: "服务不可用",
}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP 请求测试工具")
    parser.add_argument("url", type=str, help="要测试的 API 地址")
    parser.add_argument(
        "--method",
        choices=["GET", "POST"],
        default="GET",
        help="HTTP 请求方法（默认 GET）",
    )
    return parser.parse_args()


def validate_url(url: str) -> bool:
    result = urlparse(url)
    if result.scheme not in ["http", "https"]:
        logger.error("URL 必须以 http:// 或 https:// 开头")
        return False
    return True


def main() -> int:
    args = parse_args()
    if not validate_url(args.url):
        return 1

    try:
        if args.method == "GET":
            response = requests.get(args.url, timeout=10)
        else:
            response = requests.post(args.url, timeout=10)
        response.raise_for_status()

        status_text = STATUS_MAP.get(response.status_code, "未知")
        print(f"URL: {args.url}")
        print(f"请求方法: {args.method}")
        print(f"[成功] 状态码: {response.status_code} {status_text}")
        print(f"耗时: {response.elapsed.total_seconds():.2f}s")
        print(f"内容类型: {response.headers.get('Content-Type', '未知')}")
        print()
        print("响应体（前 500 字符）:")
        print(response.text[:500])
        return 0
    except requests.exceptions.ConnectionError:
        print("连接错误: 无法连接到服务器")
    except requests.exceptions.Timeout:
        print("请求超时: 服务器没有响应")
    except requests.exceptions.HTTPError:
        print(f"HTTP 错误: 请求失败，状态码 {response.status_code}")
    except requests.exceptions.RequestException as exc:
        print(f"请求异常: {exc}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
