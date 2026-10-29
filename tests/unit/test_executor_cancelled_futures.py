import concurrent.futures
import logging
import threading

import pytest

from korshmer.utils import executor as ex


@pytest.fixture(autouse=True)
def _isolated_executor_state(monkeypatch):
    monkeypatch.setattr(ex, "_executor_shutdown", False)
    monkeypatch.setattr(ex, "_executors", {})
    monkeypatch.setattr(ex, "_active_futures", {})
    yield
    for executor in list(ex._executors.values()):
        executor.shutdown(wait=True)


def _cancelled_future():
    future = concurrent.futures.Future()
    assert future.cancel() is True
    return future


def _failed_future():
    future = concurrent.futures.Future()
    future.set_exception(ValueError("boom"))
    return future


def test_cancelled_future_is_removed_without_an_error_log(caplog):
    with ex._executor_lock:
        ex._active_futures["test.cancelled"] = [_cancelled_future()]

    with caplog.at_level(logging.ERROR):
        cleaned = ex.cleanup_completed_futures()

    assert cleaned == 1
    assert "test.cancelled" not in ex._active_futures
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_failed_future_is_still_logged_once(caplog):
    with ex._executor_lock:
        ex._active_futures["test.failed"] = [_failed_future()]

    with caplog.at_level(logging.ERROR):
        cleaned = ex.cleanup_completed_futures()

    assert cleaned == 1
    errors = [r.getMessage() for r in caplog.records]
    assert errors == ["Exception in completed future: boom"]


def test_running_future_is_kept_and_not_counted():
    gate = threading.Event()
    future = ex.run_async(gate.wait, 5)
    try:
        assert ex.cleanup_completed_futures() == 0
        assert any(future in futures for futures in ex._active_futures.values())
    finally:
        gate.set()
        future.result(timeout=5)


def test_pending_task_cancelled_by_the_caller_is_cleaned_quietly(caplog):
    gate = threading.Event()
    workers = ex.compute_worker_count("io")
    blockers = [ex.run_async(gate.wait, 5) for _ in range(workers)]
    pending = ex.run_async(lambda: 1)
    try:
        assert pending.cancel() is True
    finally:
        gate.set()
        for blocker in blockers:
            blocker.result(timeout=5)

    with caplog.at_level(logging.ERROR):
        ex.cleanup_completed_futures()

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
