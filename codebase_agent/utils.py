from logging import getLogger
import time
from types import TracebackType
from contextlib import contextmanager

logger = getLogger(__name__)


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
