import re

import pytest

from korshmer.utils import audit_log

FIELD = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _fields(line):
    return dict(FIELD.findall(line))


def test_a_quote_in_a_value_cannot_add_a_field():
    line = audit_log.format_event(
        "obs.login", "denied", reason='x" outcome="allowed'
    )
    assert line.startswith("event=obs.login outcome=denied ")
    assert _fields(line) == {"reason": 'x\\" outcome=\\"allowed'}


def test_a_trailing_backslash_cannot_swallow_the_closing_quote():
    line = audit_log.format_event("x", "denied", path="a\\", other="b")
    assert _fields(line) == {"path": "a\\\\", "other": "b"}


def test_backslashes_and_quotes_are_escaped_once():
    line = audit_log.format_event("x", "allowed", path='a\\"b')
    assert 'path="a\\\\\\"b"' in line


def test_control_characters_are_still_shown_as_hex_escapes():
    line = audit_log.format_event("x", "denied", path="a\nb\x00c")
    assert "\\x0a" in line and "\\x00" in line
    assert "\n" not in line


@pytest.mark.parametrize("tail", ["", '"', "\\", '\\"'])
def test_long_values_are_cut_without_breaking_the_quoting(tail):
    line = audit_log.format_event(
        "x", "denied", path="z" * 199 + tail + "q" * 50, other="b"
    )
    assert _fields(line)["other"] == "b"
    assert "..." in _fields(line)["path"]


def test_secret_fields_stay_redacted():
    line = audit_log.format_event("x", "denied", token='a" b')
    assert 'token="<redacted>"' in line


def test_plain_values_are_unchanged():
    line = audit_log.format_event("x", "allowed", channel="room-1", n=3)
    assert line == 'event=x outcome=allowed channel="room-1" n="3"'
