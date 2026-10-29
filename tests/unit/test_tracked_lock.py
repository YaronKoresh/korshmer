import logging
import threading
import time

from korshmer.utils.tracked_lock import TrackedLock


def test_long_wait_logs_warning_with_names(caplog):
    lock = TrackedLock("test-lock", max_wait=0.05)
    held = threading.Event()
    release = threading.Event()

    def holder():
        with lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=holder, name="holder-thread")
    t.start()
    held.wait(5)

    def waiter():
        with lock:
            pass

    w = threading.Thread(target=waiter, name="waiter-thread")
    with caplog.at_level(logging.WARNING, logger="korshmer.utils.tracked_lock"):
        w.start()
        time.sleep(0.3)
        release.set()
        w.join(5)
        t.join(5)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "test-lock" in msgs[0]
    assert "waiter-thread" in msgs[0]
    assert "holder-thread" in msgs[0]


def test_reentrant_acquire_does_not_warn(caplog):
    lock = TrackedLock("re", max_wait=0.05, reentrant=True)
    with caplog.at_level(logging.WARNING):
        with lock:
            with lock:
                assert lock.locked()
    assert not caplog.records
    assert not lock.locked()
    assert lock.holder is None


def test_nonblocking_and_timeout():
    lock = TrackedLock("x", max_wait=0.05)
    assert lock.acquire()
    result = []
    t = threading.Thread(
        target=lambda: result.extend(
            [lock.acquire(False), lock.acquire(True, 0.1)]
        )
    )
    t.start()
    t.join(5)
    assert result == [False, False]
    lock.release()
