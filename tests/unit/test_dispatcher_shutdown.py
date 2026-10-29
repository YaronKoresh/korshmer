from __future__ import annotations

from korshmer.processing.queue import ProcessingQueue


def test_dispatcher_thread_exits_after_shutdown():
    q = ProcessingQueue()
    assert q._dispatcher_thread is not None
    assert q._dispatcher_thread.is_alive()

    q.shutdown()

    q._dispatcher_thread.join(timeout=5)
    assert not q._dispatcher_thread.is_alive(), (
        "Dispatcher thread must exit after shutdown"
    )
