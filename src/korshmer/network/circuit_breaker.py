from __future__ import annotations

import math
import numbers
import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RECOVERY_SECONDS = 30.0


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(
        self,
        name: str = "service",
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_seconds: float = DEFAULT_RECOVERY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if (
            isinstance(failure_threshold, bool)
            or not isinstance(failure_threshold, numbers.Integral)
            or failure_threshold < 1
        ):
            raise ValueError(
                "failure_threshold must be a whole number of at least 1, "
                f"got {failure_threshold!r}"
            )
        if (
            isinstance(recovery_seconds, bool)
            or not isinstance(recovery_seconds, numbers.Real)
            or not math.isfinite(recovery_seconds)
            or recovery_seconds < 0
        ):
            raise ValueError(
                "recovery_seconds must be a finite number that is not "
                f"negative, got {recovery_seconds!r}"
            )
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probing = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._current_state()

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def _current_state(self) -> str:
        if (
            self._state == OPEN
            and self._clock() - self._opened_at >= self.recovery_seconds
        ):
            self._state = HALF_OPEN
            self._probing = False
        return self._state

    def allow(self) -> bool:
        with self._lock:
            state = self._current_state()
            if state == CLOSED:
                return True
            if state == HALF_OPEN and not self._probing:
                self._probing = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = CLOSED
            self._failures = 0
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            state = self._current_state()
            self._probing = False
            if state == HALF_OPEN:
                self._trip()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self._clock()

    def reset(self) -> None:
        self.record_success()

    def call(self, func: Callable[[], T]) -> T:
        if not self.allow():
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            result = func()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result
