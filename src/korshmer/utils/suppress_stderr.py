import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stderr():
    old_stderr = sys.stderr
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old_stderr
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass
