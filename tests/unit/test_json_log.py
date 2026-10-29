import json
import logging

import pytest

from korshmer.utils import json_log, logging_config


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in saved_handlers:
        root.removeHandler(h)
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def _record(msg="hello %s", args=("world",), **kw):
    return logging.LogRecord(
        "korshmer.demo",
        logging.WARNING,
        "/x/demo.py",
        12,
        msg,
        args,
        None,
        **kw,
    )


def test_json_formatter_emits_one_object_per_line():
    out = json_log.JsonLogFormatter().format(_record("line one\nline two", ()))
    assert "\n" not in out
    data = json.loads(out)
    assert data["message"] == "line one\nline two"
    assert data["level"] == "WARNING"
    assert data["logger"] == "korshmer.demo"
    assert data["file"] == "demo.py"
    assert data["line"] == 12
    assert set(data) >= {"time", "thread"}


def test_json_formatter_formats_arguments():
    data = json.loads(json_log.JsonLogFormatter().format(_record()))
    assert data["message"] == "hello world"


def test_json_formatter_includes_exception_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    data = json.loads(json_log.JsonLogFormatter().format(rec))
    assert "ValueError: boom" in data["exception"]


def test_json_formatter_keeps_extra_fields_and_stringifies_unknown_types():
    rec = _record()
    rec.track_id = 7
    rec.path = object()
    data = json.loads(json_log.JsonLogFormatter().format(rec))
    assert data["track_id"] == 7
    assert data["path"].startswith("<object")


def test_json_formatter_survives_bad_format_arguments():
    data = json.loads(
        json_log.JsonLogFormatter().format(_record("value %d", ("text",)))
    )
    assert data["message"] == "value %d"


def test_resolve_log_format_values():
    assert json_log.resolve_log_format({}) == ("text", None)
    assert json_log.resolve_log_format({"KORSHMER_LOG_FORMAT": " JSON "}) == (
        "json",
        None,
    )
    assert json_log.resolve_log_format({"KORSHMER_LOG_FORMAT": "text"}) == (
        "text",
        None,
    )
    assert json_log.resolve_log_format({"KORSHMER_LOG_FORMAT": "xml"}) == (
        "text",
        "xml",
    )


def test_build_formatter_default_is_plain_text():
    fmt = json_log.build_formatter({})
    assert not isinstance(fmt, json_log.JsonLogFormatter)
    assert "hello world" in fmt.format(_record())
    assert "[demo.py:12]" in fmt.format(_record())


def test_build_formatter_json_from_env():
    fmt = json_log.build_formatter({"KORSHMER_LOG_FORMAT": "json"})
    assert isinstance(fmt, json_log.JsonLogFormatter)


def test_invalid_format_value_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        json_log.warn_if_invalid_log_format({"KORSHMER_LOG_FORMAT": "xml"})
    assert "KORSHMER_LOG_FORMAT" in caplog.text


def test_file_handler_writes_json_lines(monkeypatch, tmp_path, clean_root):
    monkeypatch.setenv("KORSHMER_LOG_FORMAT", "json")
    monkeypatch.setattr(logging_config, "LOGS_DIR", tmp_path)
    clean_root.setLevel(logging.DEBUG)
    logging_config.setup_file_logging()
    logging.getLogger("korshmer.demo").info("saved %d tracks", 3)
    for h in clean_root.handlers:
        h.flush()
    lines = (tmp_path / "korshmer.log").read_text("utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert any(p["message"] == "saved 3 tracks" for p in parsed)


def test_console_handler_uses_json_when_requested(
    monkeypatch, capsys, clean_root
):
    monkeypatch.setenv("KORSHMER_LOG_FORMAT", "json")
    for h in list(clean_root.handlers):
        clean_root.removeHandler(h)
    logging_config.setup_logging()
    logging.getLogger("korshmer.demo").warning("careful")
    out = capsys.readouterr().out.strip().splitlines()
    assert out
    assert all(isinstance(json.loads(line), dict) for line in out)
    assert json.loads(out[-1])["message"] == "careful"


def _log_from_worker_thread(fmt, in_span):
    import io
    import threading

    from korshmer.utils.tracing import current_ids, propagate, span

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(fmt)
    logger = logging.getLogger("korshmer.test.correlation")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    ids = None
    try:
        if in_span:
            with span("export"):
                ids = current_ids()
                worker = threading.Thread(
                    target=propagate(lambda: logger.info("worker line"))
                )
                worker.start()
                worker.join()
        else:
            logger.info("worker line")
    finally:
        logger.removeHandler(handler)
    return buf.getvalue(), ids


def test_json_log_carries_correlation_id_across_threads():
    fmt = json_log.build_formatter({"KORSHMER_LOG_FORMAT": "json"})
    line, ids = _log_from_worker_thread(fmt, True)
    assert json.loads(line)["correlation_id"] == ids[0]


def test_text_log_carries_correlation_id_across_threads():
    fmt = json_log.build_formatter({})
    line, ids = _log_from_worker_thread(fmt, True)
    assert f"[cid={ids[0]}] worker line" in line


def test_logs_without_span_have_no_correlation_id():
    json_line, _ = _log_from_worker_thread(
        json_log.build_formatter({"KORSHMER_LOG_FORMAT": "json"}), False
    )
    assert "correlation_id" not in json.loads(json_line)
    text_line, _ = _log_from_worker_thread(json_log.build_formatter({}), False)
    assert "cid=" not in text_line
    assert text_line.rstrip().endswith("- worker line")
