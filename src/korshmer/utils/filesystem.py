import logging
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

from korshmer.constants import RESERVED_WINDOWS_NAMES
from korshmer.exceptions import InvalidFileError
from korshmer.utils.audit_log import OUTCOME_DENIED, audit

_PATH_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f]")

MIDI_EXTENSIONS = (".mid", ".midi")


def is_midi_path(path) -> bool:
    if not isinstance(path, (str, Path)):
        return False
    return os.fspath(path).lower().endswith(MIDI_EXTENSIONS)


def sanitize_filename(filename: str) -> str:
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


def sanitize_path(path: str | Path) -> str | Path:
    path_str = os.fspath(path) if isinstance(path, Path) else str(path)
    if "\x00" in path_str:
        audit(
            "path.sanitize", OUTCOME_DENIED, reason="null_byte", path=path_str
        )
        raise InvalidFileError(f"Path contains a null byte: {path_str!r}")
    if _PATH_CONTROL_CHARS_RE.search(path_str):
        audit(
            "path.sanitize",
            OUTCOME_DENIED,
            reason="control_characters",
            path=path_str,
        )
        raise InvalidFileError(
            f"Path contains control characters: {path_str!r}"
        )
    return path


def is_safe_path(path: str | Path) -> bool:
    try:
        sanitize_path(path)
    except (InvalidFileError, TypeError):
        return False
    return True


def validate_path(
    path: str | Path,
    must_exist: bool = False,
    check_writable: bool = False,
) -> Path:
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


def ensure_directory(path: str | Path, fallback_to_temp: bool = False) -> Path:
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
            unique_suffix = str(uuid.uuid4())[:8]
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
