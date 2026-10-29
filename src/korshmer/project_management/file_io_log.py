from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from korshmer.utils.perf_metrics import metrics

LOGGER_NAME = "korshmer.file_io"

logger = logging.getLogger(LOGGER_NAME)


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


@contextmanager
def logged_file_op(
    operation: str, path, report_size: bool = True
) -> Iterator[None]:
    path = Path(path)
    logger.debug("%s started: %s", operation, path)
    started = time.perf_counter()
    try:
        yield
    except Exception as e:
        metrics.record(
            f"file_io.{operation}", time.perf_counter() - started, failed=True
        )
        logger.error(
            "%s failed: %s (%s: %s)", operation, path, type(e).__name__, e
        )
        raise
    elapsed = time.perf_counter() - started
    metrics.record(f"file_io.{operation}", elapsed)
    elapsed_ms = elapsed * 1000
    size = _file_size(path) if report_size else None
    if size is None:
        logger.info("%s done: %s (%.1f ms)", operation, path, elapsed_ms)
    else:
        logger.info(
            "%s done: %s (%d bytes, %.1f ms)",
            operation,
            path,
            size,
            elapsed_ms,
        )
