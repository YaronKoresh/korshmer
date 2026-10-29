from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

from korshmer.utils.log_rotation import CompressingRotatingFileHandler

AUDIT_LOGGER_NAME = "korshmer.audit"
AUDIT_LOG_FILENAME = "audit.log"

OUTCOME_ALLOWED = "allowed"
OUTCOME_DENIED = "denied"
OUTCOME_FAILED = "failed"

MAX_VALUE_LENGTH = 200

_SECRET_KEY_RE = re.compile(
    r"token|password|passwd|secret|credential|api_?key|stream_?key|auth",
    re.IGNORECASE,
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_EVENT_RE = re.compile(r"[^a-z0-9_.]")

logger = logging.getLogger(AUDIT_LOGGER_NAME)

_FILE_HANDLER_ATTR = "_korshmer_audit_file"


def _clean_value(value) -> str:
    text = str(value)
    truncated = len(text) > MAX_VALUE_LENGTH
    if truncated:
        text = text[:MAX_VALUE_LENGTH]
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = _CONTROL_CHARS_RE.sub(lambda m: f"\\x{ord(m.group()):02x}", text)
    return text + "..." if truncated else text


def format_event(event: str, outcome: str, **details) -> str:
    parts = [
        f"event={_EVENT_RE.sub('_', str(event).lower())}",
        f"outcome={_EVENT_RE.sub('_', str(outcome).lower())}",
    ]
    for key in sorted(details):
        value = details[key]
        if value is None:
            continue
        if _SECRET_KEY_RE.search(key):
            shown = "<redacted>"
        else:
            shown = _clean_value(value)
        parts.append(f'{key}="{shown}"')
    return " ".join(parts)


def audit(event: str, outcome: str = OUTCOME_ALLOWED, **details) -> None:
    try:
        line = format_event(event, outcome, **details)
        level = logging.INFO if outcome == OUTCOME_ALLOWED else logging.WARNING
        logger.log(level, line)
    except Exception:
        logging.getLogger(__name__).debug("audit record failed", exc_info=True)


def setup_audit_file_logging(log_dir: str | Path) -> bool:
    log_dir = Path(log_dir)
    current = getattr(logger, _FILE_HANDLER_ATTR, None)
    if current is not None:
        if (
            Path(current.baseFilename)
            == (log_dir / AUDIT_LOG_FILENAME).absolute()
        ):
            return True
        close_audit_file_logging()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = CompressingRotatingFileHandler(
            log_dir / AUDIT_LOG_FILENAME,
            maxBytes=5 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
    except OSError as e:
        logging.getLogger(__name__).warning(
            "Could not create audit log file: %s", e
        )
        return False
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    setattr(logger, _FILE_HANDLER_ATTR, handler)
    return True


def close_audit_file_logging() -> None:
    handler = getattr(logger, _FILE_HANDLER_ATTR, None)
    if handler is None:
        return
    logger.removeHandler(handler)
    handler.close()
    setattr(logger, _FILE_HANDLER_ATTR, None)
