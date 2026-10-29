from __future__ import annotations

import contextvars
import functools
import itertools
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

MAX_SPANS = 2048
MAX_ERROR_LENGTH = 200

STATUS_OK = "ok"
STATUS_ERROR = "error"

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    thread: str
    started_at: float
    started_perf: float = field(repr=False, default=0.0)
    seq: int = field(repr=False, default=0)
    duration_ms: float | None = None
    status: str = STATUS_OK
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "thread": self.thread,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": dict(self.attributes),
        }


_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "korshmer_current_span", default=None
)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Tracer:
    def __init__(self, max_spans: int = MAX_SPANS):
        if (
            isinstance(max_spans, bool)
            or not isinstance(max_spans, int)
            or max_spans < 1
        ):
            raise ValueError(
                f"max_spans must be a whole number of at least 1, got {max_spans!r}"
            )
        self._lock = threading.Lock()
        self._sequence = itertools.count()
        self._finished: deque[Span] = deque(maxlen=max_spans)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = _current.get()
        span = Span(
            trace_id=parent.trace_id if parent else _new_id(),
            span_id=_new_id(),
            parent_id=parent.span_id if parent else None,
            name=name,
            thread=threading.current_thread().name,
            started_at=time.time(),
            started_perf=time.perf_counter(),
            seq=next(self._sequence),
            attributes=dict(attributes),
        )
        token = _current.set(span)
        try:
            yield span
        except BaseException as e:
            span.status = STATUS_ERROR
            span.error = f"{type(e).__name__}: {e}"[:MAX_ERROR_LENGTH]
            raise
        finally:
            _current.reset(token)
            span.duration_ms = (time.perf_counter() - span.started_perf) * 1000
            with self._lock:
                self._finished.append(span)

    def trace(self, name: str) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.span(name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._finished)

    def snapshot(self, limit: int = 50) -> list[dict[str, Any]]:
        by_trace: dict[str, list[Span]] = {}
        for span in self.spans():
            by_trace.setdefault(span.trace_id, []).append(span)
        traces = []
        order: dict[str, tuple[float, int]] = {}
        for trace_id, spans in by_trace.items():
            spans.sort(key=lambda s: (s.started_at, s.seq))
            order[trace_id] = (spans[0].started_at, spans[0].seq)
            root = next((s for s in spans if s.parent_id is None), spans[0])
            end = max(
                s.started_at + (s.duration_ms or 0.0) / 1000 for s in spans
            )
            traces.append(
                {
                    "trace_id": trace_id,
                    "name": root.name,
                    "started_at": spans[0].started_at,
                    "duration_ms": (end - spans[0].started_at) * 1000,
                    "status": STATUS_ERROR
                    if any(s.status == STATUS_ERROR for s in spans)
                    else STATUS_OK,
                    "span_count": len(spans),
                    "threads": sorted({s.thread for s in spans}),
                    "spans": [s.as_dict() for s in spans],
                }
            )
        traces.sort(key=lambda t: order[t["trace_id"]], reverse=True)
        return traces[: max(0, limit)]

    def reset(self) -> None:
        with self._lock:
            self._finished.clear()


def current_span() -> Span | None:
    return _current.get()


def current_ids() -> tuple[str, str] | None:
    span = _current.get()
    if span is None:
        return None
    return span.trace_id, span.span_id


def correlation_id() -> str | None:
    span = _current.get()
    return span.trace_id if span else None


def propagate(fn: Callable[..., Any]) -> Callable[..., Any]:
    context = contextvars.copy_context()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return context.run(fn, *args, **kwargs)

    return wrapper


tracer = Tracer()

span = tracer.span
trace = tracer.trace
