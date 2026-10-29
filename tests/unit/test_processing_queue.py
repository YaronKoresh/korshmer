from __future__ import annotations

import threading
import time

import pytest

from korshmer.processing.queue import (
    ProcessingQueue,
    TaskPriority,
    TaskStatus,
)


@pytest.fixture
def queue():
    q = ProcessingQueue(max_workers=1)
    yield q
    q.shutdown()


def test_many_tiny_tasks_finish_quickly(queue):
    start = time.monotonic()
    tasks = [queue.submit(lambda: None) for _ in range(100)]
    for task in tasks:
        task.future.result(timeout=10)
    assert time.monotonic() - start < 3.0


def test_submit_returns_a_future_right_away(queue):
    task = queue.submit(lambda: 7)
    assert task.future is not None
    assert task.future.result(timeout=5) == 7
    assert task.status == TaskStatus.DONE


def test_failed_task_future_holds_the_exception(queue):
    def boom():
        raise ValueError("boom")

    task = queue.submit(boom)
    with pytest.raises(ValueError):
        task.future.result(timeout=5)
    assert task.status == TaskStatus.FAILED
    assert queue.stats()["failed_count"] == 1


def test_drain_waits_for_pending_tasks():
    q = ProcessingQueue(max_workers=2)
    try:
        tasks = [q.submit(lambda: time.sleep(0.1)) for _ in range(6)]
        q.drain(timeout=10)
        assert all(t.status == TaskStatus.DONE for t in tasks)
    finally:
        q.shutdown()


def test_finished_tasks_are_not_kept(queue):
    tasks = [queue.submit(lambda: None) for _ in range(50)]
    queue.drain(timeout=10)
    assert all(t.future.done() for t in tasks)
    assert queue._tasks_dict == {}


def test_cancel_pending_task_cancels_its_future(queue):
    gate = threading.Event()
    blocker = queue.submit(gate.wait)
    waiting = queue.submit(lambda: 1)
    try:
        assert queue.cancel(waiting.task_id) is True
        assert waiting.status == TaskStatus.CANCELLED
        assert waiting.future.cancelled()
    finally:
        gate.set()
    blocker.future.result(timeout=5)
    queue.drain(timeout=5)
    assert queue.stats()["pending_count"] == 0
    assert queue._tasks_dict == {}


def test_cancel_running_or_unknown_task_returns_false(queue):
    started = threading.Event()
    gate = threading.Event()

    def work():
        started.set()
        gate.wait(5)

    task = queue.submit(work)
    assert started.wait(5)
    try:
        assert queue.cancel(task.task_id) is False
        assert queue.cancel("missing") is False
    finally:
        gate.set()
    task.future.result(timeout=5)


def test_priority_order_is_kept(queue):
    gate = threading.Event()
    order = []
    blocker = queue.submit(gate.wait)
    queue.submit(order.append, "batch", priority=TaskPriority.BATCH)
    queue.submit(order.append, "realtime", priority=TaskPriority.REALTIME)
    queue.submit(order.append, "interactive")
    gate.set()
    blocker.future.result(timeout=5)
    queue.drain(timeout=5)
    assert order == ["realtime", "interactive", "batch"]


def test_shutdown_cancels_pending_tasks():
    q = ProcessingQueue(max_workers=1)
    gate = threading.Event()
    first = q.submit(gate.wait)
    later = [q.submit(lambda: 1) for _ in range(3)]
    threading.Timer(0.3, gate.set).start()
    q.shutdown()
    assert first.future.done()
    assert all(t.future.cancelled() for t in later)
    with pytest.raises(RuntimeError):
        q.submit(lambda: 1)
