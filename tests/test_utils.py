import logging

import pytest

from codebase_agent.utils import timed_operation


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
