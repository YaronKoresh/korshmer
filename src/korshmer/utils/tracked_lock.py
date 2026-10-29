import logging
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_MAX_WAIT_SECONDS = 5.0


class TrackedLock:
    def __init__(
        self,
        name: str,
        max_wait: float = DEFAULT_MAX_WAIT_SECONDS,
        reentrant: bool = False,
    ):
        self.name = name
        self.max_wait = max_wait
        self._lock = threading.RLock() if reentrant else threading.Lock()
        self._holder: str | None = None
        self._depth = 0

    @property
    def holder(self) -> str | None:
        return self._holder

    def _mark_acquired(self) -> None:
        self._holder = threading.current_thread().name
        self._depth += 1

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(False):
            self._mark_acquired()
            return True
        if not blocking:
            return False

        start = time.monotonic()
        deadline = None if timeout < 0 else start + timeout
        warned = False
        while True:
            wait = self.max_wait
            if not warned and wait > 0:
                step = wait
            else:
                step = -1
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                step = remaining if step < 0 else min(step, remaining)
            if self._lock.acquire(True, step):
                self._mark_acquired()
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if not warned:
                warned = True
                logger.warning(
                    "Possible deadlock: thread %r waited %.1fs for lock %r held by thread %r",
                    threading.current_thread().name,
                    time.monotonic() - start,
                    self.name,
                    self._holder,
                )

    def release(self) -> None:
        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            self._holder = None
        try:
            self._lock.release()
        except RuntimeError:
            self._depth = 0
            raise

    def locked(self) -> bool:
        if self._depth > 0:
            return True
        if self._lock.acquire(False):
            self._lock.release()
            return False
        return True

    def __enter__(self) -> "TrackedLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()
