from __future__ import annotations

import functools
import math
import numbers
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from korshmer.utils.tracing import tracer

SAMPLE_WINDOW = 512

F = TypeVar("F", bound=Callable[..., Any])


class _Stat:
    __slots__ = ("count", "errors", "total", "minimum", "maximum", "recent")

    def __init__(self, window: int):
        self.count = 0
        self.errors = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = 0.0
        self.recent: deque[float] = deque(maxlen=window)

    def add(self, seconds: float, failed: bool) -> None:
        self.count += 1
        if failed:
            self.errors += 1
        self.total += seconds
        self.minimum = min(self.minimum, seconds)
        self.maximum = max(self.maximum, seconds)
        self.recent.append(seconds)

    def as_dict(self) -> dict[str, float | int]:
        ordered = sorted(self.recent)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        return {
            "count": self.count,
            "errors": self.errors,
            "total_ms": self.total * 1000,
            "mean_ms": self.total / self.count * 1000,
            "min_ms": self.minimum * 1000,
            "max_ms": self.maximum * 1000,
            "p95_ms": ordered[p95_index] * 1000,
            "last_ms": self.recent[-1] * 1000,
        }


class PerfMetrics:
    def __init__(self, window: int = SAMPLE_WINDOW):
        if (
            isinstance(window, bool)
            or not isinstance(window, numbers.Integral)
            or window < 1
        ):
            raise ValueError(
                f"window must be a whole number of at least 1, got {window!r}"
            )
        self._window = window
        self._lock = threading.Lock()
        self._stats: dict[str, _Stat] = {}

    def record(self, name: str, seconds: float, failed: bool = False) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty text, got {name!r}")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, numbers.Real)
            or not math.isfinite(seconds)
        ):
            raise ValueError(
                f"seconds must be a finite number, got {seconds!r}"
            )
        with self._lock:
            stat = self._stats.get(name)
            if stat is None:
                stat = self._stats[name] = _Stat(self._window)
            stat.add(max(seconds, 0.0), failed)

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        failed = False
        try:
            with tracer.span(name):
                yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(name, time.perf_counter() - started, failed)

    def measure(self, name: str) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.timed(name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {name: stat.as_dict() for name, stat in self._stats.items()}

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()

    def report(self) -> str:
        snap = self.snapshot()
        if not snap:
            return "No performance metrics recorded."
        lines = []
        for name in sorted(snap):
            s = snap[name]
            lines.append(
                f"{name}: {s['count']} calls, {s['errors']} errors, "
                f"mean {s['mean_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms, "
                f"max {s['max_ms']:.1f} ms"
            )
        return "\n".join(lines)


metrics = PerfMetrics()

timed = metrics.timed
measure = metrics.measure
