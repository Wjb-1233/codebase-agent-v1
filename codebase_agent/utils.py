from functools import wraps
from logging import getLogger
from time import perf_counter
from types import TracebackType
from typing import Any, Callable, TypeVar, cast
import time
from contextlib import contextmanager
F = TypeVar("F", bound=Callable[..., Any])
logger = getLogger(__name__)


def cache_result(func: F) -> F:
    cache: dict[tuple[Any, frozenset[tuple[str, Any]]], Any] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return cast(F, wrapper)


class timed_operation:
    def __init__(self, name: str):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "timed_operation":
        self.start_time = time.time()
        logger.info("开始时间: %s", self.name)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool:
        end_time = time.time()
        self.elapsed = end_time - (self.start_time or end_time)
        if exc_type:
            logger.error("操作失败: %s, 耗时: %.4f秒, 错误: %s", self.name, self.elapsed, exc_val)
        else:
            logger.info("操作完成: %s, 耗时: %.4f秒", self.name, self.elapsed)
        return False

@contextmanager
def timed_block(name: str):
    # 进入 with 块之前执行（开始计时）
    start = time.time()
    logger.info("开始计时: %s", name)
    try:
        yield  # 这里会暂停，去执行 with里面的代码
    except Exception as e:
        logger.error("计时块执行出错: %s", e)
        raise
    finally:
        # 退出 with 块后一定执行（结束计时）
        cost = time.time() - start
        logger.info("计时结束，耗时: %.4f秒", cost, extra={"operation": name})
