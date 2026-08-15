import logging

import pytest

from codebase_agent.utils import cache_result, timed_operation


call_count = 0


@cache_result
def compute(a, b):
    global call_count
    call_count += 1
    return a + b


def test_cache_hit():
    global call_count
    call_count = 0

    compute(1, 2)
    compute(1, 2)

    assert call_count == 1


def test_cache_miss():
    global call_count
    call_count = 0

    compute(7, 8)
    compute(5, 6)

    assert call_count == 2


@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (10, 20, 30),
    (-5, 5, 0),
    (0, 0, 0),
])
def test_compute_param(a, b, expected):
    assert compute(a, b) == expected


def test_timed_operation_records_elapsed_time():
    with timed_operation("unit-test") as op:
        result = 1 + 1

    assert result == 2
    assert op.elapsed is not None
    assert op.elapsed >= 0


def test_timed_operation_logs_operation_name(caplog):
    with caplog.at_level(logging.INFO):
        with timed_operation("unit-test"):
            pass

    assert "unit-test" in caplog.text


def test_timed_operation_records_elapsed_when_exception_raised():
    op = None

    with pytest.raises(ValueError):
        with timed_operation("failure") as op:
            raise ValueError("boom")

    assert op is not None
    assert op.elapsed is not None
    assert op.elapsed >= 0
