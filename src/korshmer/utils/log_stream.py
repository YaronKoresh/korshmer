from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from korshmer.utils.log_filter import _matches, parse_level, record_components
from korshmer.utils.tracing import correlation_id

DEFAULT_CAPACITY = 1000
MAX_MESSAGE_LENGTH = 4000
MAX_EXCEPTION_LENGTH = 8000


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _bad_count(value) -> bool:
    return isinstance(value, bool) or not isinstance(value, int)


class LogStreamHandler(logging.Handler):
    def __init__(
        self, capacity: int = DEFAULT_CAPACITY, level: int = logging.INFO
    ):
        if _bad_count(capacity) or capacity < 1:
            raise ValueError(
                f"capacity must be a whole number of at least 1, got {capacity!r}"
            )
        super().__init__(level)
        self._entries: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._cond = threading.Condition()
        self._seq = 0
        self._closed = False

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._seq

    @property
    def closed(self) -> bool:
        return self._closed

    def emit(self, record: logging.LogRecord) -> None:
        try:
            try:
                message = record.getMessage()
            except Exception:
                message = str(record.msg)
            entry: dict[str, Any] = {
                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(record.created)
                ),
                "level": record.levelname,
                "levelno": record.levelno,
                "logger": record.name,
                "components": record_components(record),
                "file": record.filename,
                "line": record.lineno,
                "thread": record.threadName,
                "message": _clip(message, MAX_MESSAGE_LENGTH),
            }
            cid = correlation_id()
            if cid:
                entry["correlation_id"] = cid
            if record.exc_info:
                entry["exception"] = _clip(
                    logging.Formatter().formatException(record.exc_info),
                    MAX_EXCEPTION_LENGTH,
                )
            with self._cond:
                if self._closed:
                    return
                self._seq += 1
                entry["seq"] = self._seq
                self._entries.append(entry)
                self._cond.notify_all()
        except Exception:
            self.handleError(record)

    def read(
        self,
        since: int = 0,
        level: int | None = None,
        component: str | None = None,
        limit: int = DEFAULT_CAPACITY,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        if _bad_count(since) or since < 0:
            raise ValueError(
                f"since must be a whole number that is not negative, got {since!r}"
            )
        if limit is not None and (_bad_count(limit) or limit < 0):
            raise ValueError(
                f"limit must be a whole number that is not negative, got {limit!r}"
            )
        with self._cond:
            entries = list(self._entries)
            last = self._seq
        oldest = entries[0]["seq"] if entries else last + 1
        dropped = since + 1 < oldest and last > since
        selected = []
        for entry in entries:
            if entry["seq"] <= since:
                continue
            if level is not None and entry["levelno"] < level:
                continue
            if component and not _matches(entry["components"], (component,)):
                continue
            selected.append(entry)
        if limit and len(selected) > limit:
            selected = selected[:limit]
            last = selected[-1]["seq"]
        return selected, last, dropped

    def wait(self, since: int, timeout: float) -> bool:
        with self._cond:
            if self._seq > since or self._closed:
                return self._seq > since
            self._cond.wait(timeout)
            return self._seq > since

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        super().close()


def parse_stream_level(value: str | None) -> tuple[int | None, bool]:
    level, invalid = parse_level(value)
    return level, invalid is not None


def install_log_stream(
    logger: logging.Logger | None = None,
    capacity: int = DEFAULT_CAPACITY,
) -> LogStreamHandler:
    handler = LogStreamHandler(capacity)
    (logger or logging.getLogger()).addHandler(handler)
    return handler


def remove_log_stream(
    handler: LogStreamHandler, logger: logging.Logger | None = None
) -> None:
    (logger or logging.getLogger()).removeHandler(handler)
    handler.close()
