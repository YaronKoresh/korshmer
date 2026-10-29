from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

MAX_DELAY = 30.0


DEFAULT_MAX_BYTES = 6 * 1024 * 1024


class DelayBuffer:
    def __init__(
        self,
        delay: float = 0.0,
        max_bytes: int = DEFAULT_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._blocks: deque[list] = deque()
        self._bytes = 0
        self._delay = 0.0
        self.set_delay(delay)

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def set_delay(self, seconds: float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError("delay must be a number of seconds")
        if not math.isfinite(seconds) or seconds < 0 or seconds > MAX_DELAY:
            raise ValueError(
                f"delay must be between 0 and {MAX_DELAY:g} seconds"
            )
        with self._lock:
            self._delay = float(seconds)

    def push(self, data: bytes) -> list[bytes]:
        now = self._clock()
        with self._lock:
            if data:
                self._blocks.append([now, bytes(data)])
                self._bytes += len(data)
            due: list[bytes] = []
            while self._blocks:
                arrival, block = self._blocks[0]
                if (
                    now - arrival >= self._delay
                    or self._bytes > self._max_bytes
                ):
                    self._blocks.popleft()
                    self._bytes -= len(block)
                    due.append(block)
                else:
                    break
            return due

    def silence_last(self, seconds: float) -> int:
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("seconds must be a non-negative number")
        now = self._clock()
        return self.silence_between(now - seconds, now)

    def silence_between(self, start: float, end: float) -> int:
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError("invalid time window")
        count = 0
        with self._lock:
            for item in self._blocks:
                if start <= item[0] <= end:
                    item[1] = bytes(len(item[1]))
                    count += 1
        return count

    def clear(self) -> int:
        with self._lock:
            n = len(self._blocks)
            self._blocks.clear()
            self._bytes = 0
            return n
