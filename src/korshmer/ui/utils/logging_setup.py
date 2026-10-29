import logging
import logging.handlers
import platform
import sys
from pathlib import Path

from korshmer import __version__
from korshmer.utils.json_log import build_formatter
from korshmer.utils.log_rotation import CompressingRotatingFileHandler

from .file_operations import ensure_directory


def setup_logging(base_dir):
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
        logging.info(f"Base Directory: {base_dir}")
        logging.info("=" * 60)
    except Exception as e:
        print(f"Warning: Logging setup failed: {e}", file=sys.stderr)


def setup_file_logging(logs_dir):
    try:
        log_file = Path(logs_dir) / "korshmer.log"
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


def ensure_directories(data_dir, temp_dir, models_dir, logs_dir):
    try:
        for d in [data_dir, temp_dir, models_dir, logs_dir]:
            ensure_directory(d)
    except Exception as e:
        if logging.getLogger().hasHandlers():
            logging.error(f"Could not create directories: {e}")
        else:
            print(f"Error: Could not create directories: {e}", file=sys.stderr)
