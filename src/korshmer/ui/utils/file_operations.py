import contextlib
import logging
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

from korshmer.utils.secure_random import token_hex

RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def sanitize_filename(filename):
    if not filename:
        return "unnamed"

    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    sanitized = sanitized.rstrip(" .")
    if not sanitized:
        return "unnamed"
    if sys.platform == "win32":
        base_name = os.path.splitext(sanitized)[0].upper()
        if base_name in RESERVED_WINDOWS_NAMES:
            sanitized = f"file_{sanitized}"
    return sanitized[:255].rstrip(" .")


def validate_path(path, must_exist=False, check_writable=False):
    try:
        path = Path(path).resolve()

        if sys.platform == "win32" and len(str(path)) > 260:
            logging.warning(f"Path length exceeds Windows MAX_PATH: {path}")

        if must_exist and not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if check_writable and path.exists():
            if not os.access(path, os.W_OK):
                raise PermissionError(f"Path is not writable: {path}")

        return path
    except Exception as e:
        logging.error(f"Path validation failed for {path}: {e}")
        raise


def ensure_directory(path, fallback_to_temp=False):
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (OSError, PermissionError) as e:
        error_msg = f"Failed to create directory {path}: {e}"
        if logging.getLogger().hasHandlers():
            logging.error(error_msg)
        else:
            print(f"Warning: {error_msg}", file=sys.stderr)

        if fallback_to_temp:
            unique_suffix = token_hex(4)
            temp_path = (
                Path(tempfile.gettempdir()) / f"{path.name}_{unique_suffix}"
            )
            warning_msg = f"Falling back to temp directory: {temp_path}"
            if logging.getLogger().hasHandlers():
                logging.warning(warning_msg)
            else:
                print(f"Warning: {warning_msg}", file=sys.stderr)
            temp_path.mkdir(parents=True, exist_ok=True)
            return temp_path
        raise


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
            except OSError:
                pass


def cleanup_temp_files(data_dir, temp_dir):
    files_to_clean = ["temp_preview.wav", "temp_spectrum.png"]
    search_dirs = [data_dir, temp_dir]
    for d in search_dirs:
        for f in files_to_clean:
            try:
                p = Path(d) / f
                if p.exists():
                    p.unlink()
                    logging.debug(f"Cleaned up temp file: {p}")
            except Exception as e:
                logging.debug(f"Could not clean up {p}: {e}")
