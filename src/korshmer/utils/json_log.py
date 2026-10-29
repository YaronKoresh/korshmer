import json
import logging
import os

from korshmer.utils.tracing import correlation_id

LOG_FORMAT_ENV = "KORSHMER_LOG_FORMAT"
TEXT_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(correlation)s%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName", "correlation"}


class TextLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(TEXT_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record):
        cid = correlation_id()
        record.correlation = f"[cid={cid}] " if cid else ""
        return super().format(record)


class JsonLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(datefmt=DATE_FORMAT)

    def format(self, record):
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "file": record.filename,
            "line": record.lineno,
            "thread": record.threadName,
            "message": message,
        }
        cid = correlation_id()
        if cid:
            payload["correlation_id"] = cid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_") or key in payload:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def resolve_log_format(environ=None):
    env = os.environ if environ is None else environ
    value = (env.get(LOG_FORMAT_ENV) or "").strip().lower()
    if value in ("", "text"):
        return "text", None
    if value == "json":
        return "json", None
    return "text", value


def build_formatter(environ=None):
    name, _ = resolve_log_format(environ)
    if name == "json":
        return JsonLogFormatter()
    return TextLogFormatter()


def warn_if_invalid_log_format(environ=None):
    _, invalid = resolve_log_format(environ)
    if invalid is not None:
        logging.warning(f"Ignoring invalid {LOG_FORMAT_ENV} value: {invalid!r}")
