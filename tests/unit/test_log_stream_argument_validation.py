import logging

import pytest

from korshmer.utils.log_stream import LogStreamHandler

NAN = float("nan")


def record(message="hello"):
    return logging.LogRecord("t", logging.INFO, __file__, 1, message, (), None)


@pytest.mark.parametrize(
    "capacity", [0, -1, 2.5, "5", None, True, NAN, float("inf")]
)
def test_capacity_must_be_a_positive_whole_number(capacity):
    with pytest.raises(ValueError, match="capacity"):
        LogStreamHandler(capacity)


@pytest.mark.parametrize("since", ["x", None, 1.5, True, -1, NAN])
def test_read_since_must_be_a_non_negative_whole_number(since):
    with pytest.raises(ValueError, match="since"):
        LogStreamHandler().read(since=since)


@pytest.mark.parametrize("limit", ["x", 1.5, True, -1, NAN])
def test_read_limit_must_be_a_non_negative_whole_number(limit):
    with pytest.raises(ValueError, match="limit"):
        LogStreamHandler().read(limit=limit)


def test_read_limit_none_and_zero_return_every_entry():
    handler = LogStreamHandler()
    for i in range(3):
        handler.emit(record(str(i)))
    assert len(handler.read(limit=0)[0]) == 3
    assert len(handler.read(limit=None)[0]) == 3
    assert len(handler.read(limit=2)[0]) == 2


def test_valid_arguments_still_work():
    handler = LogStreamHandler(2)
    for i in range(3):
        handler.emit(record(str(i)))
    entries, last, dropped = handler.read(since=0)
    assert [e["message"] for e in entries] == ["1", "2"]
    assert last == 3 and dropped is True


@pytest.mark.parametrize("size", [0, -1, 2.5, "5", None, True, NAN])
def test_tracer_size_must_be_a_positive_whole_number(size):
    from korshmer.utils.tracing import Tracer

    with pytest.raises(ValueError, match="max_spans"):
        Tracer(max_spans=size)
