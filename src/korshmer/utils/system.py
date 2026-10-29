import asyncio
import contextlib
import logging
import os
import sys
import threading

_stderr_lock = threading.Lock()
_stderr_suppression_count = 0


@contextlib.contextmanager
def suppress_stderr():
    global _stderr_suppression_count
    should_suppress = False
    with _stderr_lock:
        _stderr_suppression_count += 1
        should_suppress = _stderr_suppression_count == 1
    if not should_suppress:
        try:
            yield
        finally:
            with _stderr_lock:
                _stderr_suppression_count -= 1
        return
    old_stderr = None
    devnull = None
    try:
        if sys.platform == "win32":
            null_file = "NUL"
        else:
            null_file = "/dev/null"
        try:
            stderr_fd = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fd)
            devnull = open(null_file, "w")
            os.dup2(devnull.fileno(), stderr_fd)
        except (AttributeError, OSError) as e:
            logging.debug(f"Could not suppress stderr: {e}")
        yield
    finally:
        with _stderr_lock:
            _stderr_suppression_count -= 1
        if old_stderr is not None:
            try:
                stderr_fd = sys.stderr.fileno()
                os.dup2(old_stderr, stderr_fd)
            except OSError:
                pass
            try:
                os.close(old_stderr)
            except OSError:
                pass
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass


def setup_event_loop():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
