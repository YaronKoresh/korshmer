import logging
import logging.handlers
import os
import platform
import sys

from korshmer import __version__
from korshmer.constants import (
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TEMP_DIR,
)
from korshmer.utils.audit_log import setup_audit_file_logging
from korshmer.utils.filesystem import ensure_directory
from korshmer.utils.json_log import build_formatter, warn_if_invalid_log_format
from korshmer.utils.log_filter import configure_log_filters
from korshmer.utils.log_rotation import CompressingRotatingFileHandler

LOG_LEVELS_ENV = "KORSHMER_LOG_LEVELS"


def parse_log_levels(spec):
    levels = {}
    errors = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, level_name = part.partition("=")
        name = name.strip()
        level_name = level_name.strip().upper()
        level = logging.getLevelName(level_name) if level_name else None
        if not sep or not name or not isinstance(level, int):
            errors.append(part)
            continue
        levels[name] = level
    return levels, errors


def apply_log_levels(levels):
    for name, level in levels.items():
        target = (
            logging.getLogger() if name == "root" else logging.getLogger(name)
        )
        target.setLevel(level)


def configure_module_log_levels(environ=None):
    env = os.environ if environ is None else environ
    levels, errors = parse_log_levels(env.get(LOG_LEVELS_ENV, ""))
    apply_log_levels(levels)
    for part in errors:
        logging.warning(f"Ignoring invalid {LOG_LEVELS_ENV} entry: {part!r}")
    for name, level in levels.items():
        logging.info(f"Log level for {name}: {logging.getLevelName(level)}")
    return levels


def setup_logging():
    try:
        logger = logging.getLogger()
        if logger.hasHandlers():
            return
        logger.setLevel(logging.DEBUG)
        formatter = build_formatter()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logging.info("=" * 60)
        logging.info("KORSHMER Application Starting")
        logging.info(f"Version: {__version__}")
        logging.info(f"Python: {sys.version}")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Base Directory: {BASE_DIR}")
        logging.info("=" * 60)
    except Exception as e:
        print(f"Warning: Logging setup failed: {e}", file=sys.stderr)


def ensure_directories():
    failures = []
    for d in [DATA_DIR, TEMP_DIR, MODELS_DIR, LOGS_DIR]:
        try:
            ensure_directory(d)
        except Exception as e:
            failures.append((d, e))
    for d, e in failures:
        if logging.getLogger().hasHandlers():
            logging.error(f"Could not create directory {d}: {e}")
        else:
            print(
                f"Error: Could not create directory {d}: {e}", file=sys.stderr
            )


def setup_file_logging():
    try:
        log_file = LOGS_DIR / "korshmer.log"
        logger = logging.getLogger()
        file_handler = CompressingRotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = build_formatter()
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info("File logging enabled")
    except Exception as e:
        logging.warning(f"Could not create log file: {e}")


def initialize_logging():
    setup_logging()
    ensure_directories()
    setup_file_logging()
    warn_if_invalid_log_format()
    configure_module_log_levels()
    configure_log_filters()
    setup_audit_file_logging(LOGS_DIR)
