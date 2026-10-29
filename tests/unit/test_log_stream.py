import json
import logging
import threading
import urllib.error
import urllib.request

import pytest

from korshmer.network.health_server import HealthServer
from korshmer.utils.log_stream import (
    LogStreamHandler,
    install_log_stream,
    parse_stream_level,
    remove_log_stream,
)


def make_record(msg="hello", level=logging.INFO, name="korshmer.audio.engine"):
    return logging.LogRecord(
        name, level, "/app/src/korshmer/audio/engine.py", 7, msg, (), None
    )


def _get_json(server, path):
    url = f"http://{server.host}:{server.port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def stream():
    handler = LogStreamHandler(capacity=5)
    yield handler
    handler.close()


@pytest.fixture
def served(stream):
    server = HealthServer(port=0, log_stream=stream)
    assert server.start()
    yield server, stream
    server.stop()


class TestHandler:
    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            LogStreamHandler(capacity=0)

    def test_entries_get_increasing_sequence_numbers(self, stream):
        stream.emit(make_record("a"))
        stream.emit(make_record("b"))
        entries, last, dropped = stream.read()
        assert [e["seq"] for e in entries] == [1, 2]
        assert [e["message"] for e in entries] == ["a", "b"]
        assert last == 2
        assert dropped is False

    def test_entry_fields(self, stream):
        stream.emit(make_record("hi", logging.WARNING))
        entry = stream.read()[0][0]
        assert entry["level"] == "WARNING"
        assert entry["logger"] == "korshmer.audio.engine"
        assert entry["file"] == "engine.py"
        assert entry["line"] == 7
        json.dumps(entry)

    def test_read_since_returns_only_newer(self, stream):
        for i in range(3):
            stream.emit(make_record(str(i)))
        entries, last, _ = stream.read(since=2)
        assert [e["message"] for e in entries] == ["2"]
        assert last == 3

    def test_oldest_entries_are_dropped_and_flagged(self, stream):
        for i in range(8):
            stream.emit(make_record(str(i)))
        entries, last, dropped = stream.read(since=0)
        assert [e["seq"] for e in entries] == [4, 5, 6, 7, 8]
        assert dropped is True
        assert stream.read(since=3)[2] is False
        assert stream.read(since=8)[2] is False

    def test_level_filter(self, stream):
        stream.emit(make_record("low", logging.INFO))
        stream.emit(make_record("high", logging.ERROR))
        entries, last, _ = stream.read(level=logging.WARNING)
        assert [e["message"] for e in entries] == ["high"]
        assert last == 2

    def test_component_filter_matches_logger_and_module(self, stream):
        stream.emit(make_record("a", name="korshmer.network.x"))
        stream.emit(make_record("b", name="other"))
        by_logger = stream.read(component="korshmer.network")[0]
        by_module = stream.read(component="korshmer.audio.engine")[0]
        assert [e["message"] for e in by_logger] == ["a"]
        assert len(by_module) == 2

    def test_limit_keeps_cursor_at_last_returned(self, stream):
        for i in range(4):
            stream.emit(make_record(str(i)))
        entries, last, _ = stream.read(limit=2)
        assert [e["seq"] for e in entries] == [1, 2]
        assert last == 2

    def test_level_names_are_parsed(self):
        assert parse_stream_level("warning") == (logging.WARNING, False)
        assert parse_stream_level("") == (None, False)
        assert parse_stream_level("loud") == (None, True)

    def test_exception_text_is_attached(self, stream):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            record = make_record("failed", logging.ERROR)
            record.exc_info = sys.exc_info()
        stream.emit(record)
        assert "RuntimeError: boom" in stream.read()[0][0]["exception"]

    def test_long_message_is_clipped(self, stream):
        stream.emit(make_record("x" * 10000))
        assert len(stream.read()[0][0]["message"]) < 5000

    def test_wait_wakes_on_new_entry(self, stream):
        timer = threading.Timer(0.1, stream.emit, args=(make_record(),))
        timer.start()
        assert stream.wait(0, 3) is True
        timer.join()

    def test_wait_times_out_without_entries(self, stream):
        assert stream.wait(0, 0.05) is False

    def test_closed_handler_ignores_records(self, stream):
        stream.close()
        stream.emit(make_record())
        assert stream.read()[0] == []
        assert stream.closed

    def test_install_and_remove(self):
        logger = logging.Logger("stream_test")
        handler = install_log_stream(logger)
        logger.warning("seen")
        assert handler in logger.handlers
        assert handler.read()[0][0]["message"] == "seen"
        remove_log_stream(handler, logger)
        assert handler not in logger.handlers
        assert handler.closed


class TestLogsEndpoint:
    def test_returns_logs_and_cursor(self, served):
        server, stream = served
        stream.emit(make_record("one"))
        stream.emit(make_record("two"))
        status, body = _get_json(server, "/health/logs")
        assert status == 200
        assert body["last_seq"] == 2
        assert [e["message"] for e in body["logs"]] == ["one", "two"]
        status, body = _get_json(server, "/health/logs?since=1")
        assert [e["message"] for e in body["logs"]] == ["two"]

    def test_level_component_and_limit_parameters(self, served):
        server, stream = served
        stream.emit(make_record("info", logging.INFO))
        stream.emit(make_record("err", logging.ERROR, name="korshmer.network"))
        stream.emit(make_record("err2", logging.ERROR, name="korshmer.network"))
        _, body = _get_json(server, "/health/logs?level=error&limit=1")
        assert [e["message"] for e in body["logs"]] == ["err"]
        _, body = _get_json(server, "/health/logs?component=korshmer.network")
        assert [e["message"] for e in body["logs"]] == ["err", "err2"]

    @pytest.mark.parametrize(
        "query", ["since=abc", "since=-1", "limit=0", "level=loud"]
    )
    def test_bad_parameters_return_400(self, served, query):
        server, _ = served
        status, body = _get_json(server, f"/health/logs?{query}")
        assert status == 400
        assert body["status"] == "error"

    def test_off_without_stream(self):
        server = HealthServer(port=0)
        assert server.start()
        try:
            status, _ = _get_json(server, "/health/logs")
            assert status == 404
        finally:
            server.stop()

    def test_other_endpoints_still_work_with_query_string(self, served):
        server, _ = served
        status, body = _get_json(server, "/health?x=1")
        assert status == 200
        assert body["status"] == "ok"


class TestSseEndpoint:
    def _open(self, server, path="/health/logs/stream", headers=None):
        url = f"http://{server.host}:{server.port}{path}"
        request = urllib.request.Request(url, headers=headers or {})
        return urllib.request.urlopen(request, timeout=5)

    def _read_event(self, resp):
        seq = data = None
        while True:
            line = resp.readline().decode("utf-8").rstrip("\n")
            if line.startswith("id: "):
                seq = int(line[4:])
            elif line.startswith("data: "):
                data = json.loads(line[6:])
            elif line == "" and data is not None:
                return seq, data

    def test_pushes_new_lines_as_events(self, served):
        server, stream = served
        stream.emit(make_record("before"))
        with self._open(server) as resp:
            assert resp.headers["Content-Type"] == "text/event-stream"
            seq, data = self._read_event(resp)
            assert (seq, data["message"]) == (1, "before")
            threading.Timer(
                0.1, stream.emit, args=(make_record("live"),)
            ).start()
            seq, data = self._read_event(resp)
            assert (seq, data["message"]) == (2, "live")

    def test_resumes_from_last_event_id(self, served):
        server, stream = served
        for name in ("a", "b", "c"):
            stream.emit(make_record(name))
        with self._open(server, headers={"Last-Event-ID": "2"}) as resp:
            seq, data = self._read_event(resp)
            assert (seq, data["message"]) == (3, "c")

    def test_level_filter_applies_to_events(self, served):
        server, stream = served
        stream.emit(make_record("skip", logging.INFO))
        stream.emit(make_record("keep", logging.ERROR))
        with self._open(server, "/health/logs/stream?level=ERROR") as resp:
            _, data = self._read_event(resp)
            assert data["message"] == "keep"

    def test_bad_parameter_returns_400(self, served):
        server, _ = served
        with pytest.raises(urllib.error.HTTPError) as err:
            self._open(server, "/health/logs/stream?since=x")
        assert err.value.code == 400

    def test_stop_ends_open_streams(self, stream):
        server = HealthServer(port=0, log_stream=stream)
        assert server.start()
        resp = self._open(server)
        stopper = threading.Timer(0.2, server.stop)
        stopper.start()
        assert resp.read() == b""
        stopper.join()
        resp.close()
