import asyncio
import json
import logging

import pytest

from korshmer.exceptions import InvalidFileError
from korshmer.network.health_server import HealthServer
from korshmer.network.obs_client import OP_HELLO, OP_IDENTIFIED, ObsClient
from korshmer.network.twitch_client import TwitchClient
from korshmer.plugins import PluginRegistry
from korshmer.project_management.project_loader import is_safe_archive_member
from korshmer.project_management.validators.project_validator import (
    resolve_inside,
)
from korshmer.utils import audit_log
from korshmer.utils.filesystem import sanitize_path


@pytest.fixture
def records(caplog):
    caplog.set_level(logging.INFO, logger=audit_log.AUDIT_LOGGER_NAME)

    def lines():
        return [
            r.getMessage()
            for r in caplog.records
            if r.name == audit_log.AUDIT_LOGGER_NAME
        ]

    return lines


def test_format_event_sorts_fields_and_skips_none():
    line = audit_log.format_event("Path.Check", "denied", b="2", a="1", c=None)
    assert line == 'event=path.check outcome=denied a="1" b="2"'


def test_format_event_redacts_secret_fields():
    line = audit_log.format_event(
        "login", "allowed", token="abc123", password="pw", api_key="k", nick="n"
    )
    assert "abc123" not in line
    assert "pw" not in line.replace("password", "")
    assert line.count("<redacted>") == 3
    assert 'nick="n"' in line


def test_format_event_escapes_control_characters_and_truncates():
    line = audit_log.format_event("x", "denied", path="a\nb\x00c" + "z" * 500)
    assert "\n" not in line
    assert "\\x0a" in line and "\\x00" in line
    assert line.endswith('..."')
    assert len(line) < 400


def test_format_event_normalizes_event_name():
    assert audit_log.format_event("A b/c", "OK").startswith("event=a_b_c ")


def test_audit_levels(caplog):
    caplog.set_level(logging.INFO, logger=audit_log.AUDIT_LOGGER_NAME)
    audit_log.audit("a")
    audit_log.audit("b", audit_log.OUTCOME_DENIED)
    levels = [
        r.levelno
        for r in caplog.records
        if r.name == audit_log.AUDIT_LOGGER_NAME
    ]
    assert levels == [logging.INFO, logging.WARNING]


def test_audit_never_raises(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(audit_log, "format_event", broken)
    audit_log.audit("x")


def test_file_handler_writes_records_and_is_added_once(tmp_path):
    try:
        assert audit_log.setup_audit_file_logging(tmp_path / "logs")
        assert audit_log.setup_audit_file_logging(tmp_path / "logs")
        assert len(audit_log.logger.handlers) == 1
        audit_log.audit("file.test", audit_log.OUTCOME_DENIED, reason="r")
        for h in audit_log.logger.handlers:
            h.flush()
        text = (tmp_path / "logs" / audit_log.AUDIT_LOG_FILENAME).read_text(
            encoding="utf-8"
        )
        assert 'event=file.test outcome=denied reason="r"' in text
        assert "WARNING" in text
    finally:
        audit_log.close_audit_file_logging()
    assert audit_log.logger.handlers == []


def test_file_handler_switches_to_a_new_directory(tmp_path):
    try:
        assert audit_log.setup_audit_file_logging(tmp_path / "first")
        assert audit_log.setup_audit_file_logging(tmp_path / "second")
        assert len(audit_log.logger.handlers) == 1
        audit_log.audit("file.switch", audit_log.OUTCOME_DENIED)
        for h in audit_log.logger.handlers:
            h.flush()
        second = tmp_path / "second" / audit_log.AUDIT_LOG_FILENAME
        assert "event=file.switch" in second.read_text(encoding="utf-8")
    finally:
        audit_log.close_audit_file_logging()


def test_file_handler_failure_returns_false(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert audit_log.setup_audit_file_logging(blocker / "logs") is False
    assert audit_log.logger.handlers == []


def test_resolve_inside_records_denied_path(tmp_path, records):
    base = tmp_path / "base"
    base.mkdir()
    assert resolve_inside(base, base / "ok.txt") is not None
    assert records() == []
    assert resolve_inside(base, tmp_path / "other.txt") is None
    assert resolve_inside(base, "bad\x00name") is None
    out = records()
    assert len(out) == 2
    assert "event=path.resolve_inside outcome=denied" in out[0]
    assert 'reason="outside_base"' in out[0]
    assert 'reason="null_byte"' in out[1]


def test_sanitize_path_records_denied_path(records):
    with pytest.raises(InvalidFileError):
        sanitize_path("a\x00b")
    with pytest.raises(InvalidFileError):
        sanitize_path("a\tb")
    sanitize_path("fine.txt")
    out = records()
    assert len(out) == 2
    assert 'reason="null_byte"' in out[0]
    assert 'reason="control_characters"' in out[1]


def test_unsafe_archive_member_is_recorded(tmp_path, records):
    assert is_safe_archive_member("assets/a.wav", tmp_path)
    assert records() == []
    assert not is_safe_archive_member("../evil.txt", tmp_path)
    assert not is_safe_archive_member("/abs.txt", tmp_path)
    out = records()
    assert len(out) == 2
    assert all("event=archive.member outcome=denied" in x for x in out)
    assert 'member="../evil.txt"' in out[0]


class FakeSocket:
    def settimeout(self, t):
        pass

    def sendall(self, data):
        pass

    def close(self):
        pass

    def shutdown(self, how):
        pass


def test_twitch_invalid_token_is_recorded_without_the_token(records):
    with pytest.raises(ValueError):
        TwitchClient("chan", "bad token!", "nick")
    out = records()
    assert len(out) == 1
    assert "event=twitch.credentials outcome=denied" in out[0]
    assert "bad token" not in out[0]


def test_twitch_login_outcomes_are_recorded(records):
    client = TwitchClient("chan", "tok", "nick", connect=lambda t: FakeSocket())
    client.handle_line(":tmi.twitch.tv 001 nick :Welcome")
    client.handle_line(":tmi.twitch.tv NOTICE * :Login authentication failed")
    out = records()
    assert len(out) == 2
    assert "event=twitch.login outcome=allowed" in out[0]
    assert 'channel="chan"' in out[0]
    assert "outcome=denied" in out[1]
    assert 'reason="authentication_failed"' in out[1]
    assert not any("tok" in x.replace("twitch", "") for x in out)


def test_twitch_connect_failure_is_recorded(records):
    def connect(timeout):
        raise OSError("down")

    client = TwitchClient("chan", "tok", "nick", connect=connect)
    assert client.start() is False
    out = records()
    assert len(out) == 1
    assert "event=twitch.connect outcome=failed" in out[0]
    assert 'error="OSError"' in out[0]


class FakeWs:
    def __init__(self, replies):
        self.replies = [json.dumps(r) for r in replies]
        self.sent = []

    async def recv(self):
        return self.replies.pop(0)

    async def send(self, data):
        self.sent.append(data)


def run_handshake(client, replies):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(client._handshake(FakeWs(replies)))
    finally:
        loop.close()


def test_obs_login_outcomes_are_recorded(records):
    hello = {
        "op": OP_HELLO,
        "d": {"authentication": {"salt": "s", "challenge": "c"}},
    }
    client = ObsClient(password="secretpw")
    run_handshake(client, [hello, {"op": OP_IDENTIFIED}])
    with pytest.raises(PermissionError):
        run_handshake(client, [hello, {"op": 0}])
    with pytest.raises(PermissionError):
        run_handshake(ObsClient(), [hello])
    out = records()
    assert len(out) == 3
    assert "event=obs.login outcome=allowed" in out[0]
    assert 'reason="identification_refused"' in out[1]
    assert 'reason="password_missing"' in out[2]
    assert not any("secretpw" in x for x in out)


def test_health_server_start_and_stop_are_recorded(records):
    server = HealthServer(port=0)
    assert server.start()
    server.stop()
    out = records()
    assert len(out) == 2
    assert "event=health_server.start outcome=allowed" in out[0]
    assert "event=health_server.stop outcome=allowed" in out[1]


def test_health_server_bind_failure_is_recorded(records):
    first = HealthServer(port=0)
    assert first.start()
    try:
        second = HealthServer(port=first.port)
        assert second.start() is False
    finally:
        first.stop()
    assert any(
        "event=health_server.start outcome=failed" in x for x in records()
    )


def test_plugin_directory_loading_is_recorded(tmp_path, records):
    (tmp_path / "good.py").write_text("X = 1\n")
    (tmp_path / "broken.py").write_text("raise RuntimeError('no')\n")
    PluginRegistry.reset()
    try:
        PluginRegistry().discover_directory(str(tmp_path))
    finally:
        PluginRegistry.reset()
    out = records()
    assert len(out) == 2
    good = [x for x in out if "good.py" in x]
    bad = [x for x in out if "broken.py" in x]
    assert "event=plugin.load outcome=allowed" in good[0]
    assert "event=plugin.load outcome=failed" in bad[0]
    assert 'error="RuntimeError"' in bad[0]
