import gzip
import logging

from korshmer.utils import audit_log
from korshmer.utils.log_rotation import CompressingRotatingFileHandler


def _logger(name, handler):
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_rotated_files_are_gzipped(tmp_path):
    handler = CompressingRotatingFileHandler(
        tmp_path / "app.log", maxBytes=300, backupCount=3, encoding="utf-8"
    )
    logger = _logger("test_log_rotation_a", handler)
    for i in range(60):
        logger.info("message number %d", i)
    handler.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "app.log" in names
    assert "app.log.1.gz" in names
    assert not any(n.endswith(".1") or n.endswith(".tmp") for n in names)
    with gzip.open(tmp_path / "app.log.1.gz", "rt", encoding="utf-8") as f:
        assert "message number" in f.read()


def test_backup_count_is_respected(tmp_path):
    handler = CompressingRotatingFileHandler(
        tmp_path / "app.log", maxBytes=200, backupCount=2, encoding="utf-8"
    )
    logger = _logger("test_log_rotation_b", handler)
    for i in range(100):
        logger.info("message number %d", i)
    handler.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["app.log", "app.log.1.gz", "app.log.2.gz"]


def test_oldest_backup_keeps_older_lines_in_order(tmp_path):
    handler = CompressingRotatingFileHandler(
        tmp_path / "app.log", maxBytes=200, backupCount=5, encoding="utf-8"
    )
    logger = _logger("test_log_rotation_c", handler)
    for i in range(40):
        logger.info("line %03d", i)
    handler.close()
    lines = []
    for n in (5, 4, 3, 2, 1):
        path = tmp_path / f"app.log.{n}.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                lines += f.read().splitlines()
    lines += (tmp_path / "app.log").read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines)
    assert lines[-1] == "line 039"


def test_no_rotation_when_below_limit(tmp_path):
    handler = CompressingRotatingFileHandler(
        tmp_path / "app.log", maxBytes=100000, backupCount=3, encoding="utf-8"
    )
    logger = _logger("test_log_rotation_d", handler)
    logger.info("hello")
    handler.close()
    assert [p.name for p in tmp_path.iterdir()] == ["app.log"]


def test_audit_log_uses_compressing_handler(tmp_path):
    audit_log.close_audit_file_logging()
    try:
        assert audit_log.setup_audit_file_logging(tmp_path)
        handler = getattr(audit_log.logger, audit_log._FILE_HANDLER_ATTR)
        assert isinstance(handler, CompressingRotatingFileHandler)
    finally:
        audit_log.close_audit_file_logging()
