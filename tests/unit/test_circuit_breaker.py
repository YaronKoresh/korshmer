import urllib.error

import pytest

from korshmer.ai.copilot import MusicalContext, MusicCopilot
from korshmer.network.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitOpenError,
)
from korshmer.network.twitch_client import TwitchClient


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(threshold=3, recovery=10.0):
    clock = Clock()
    return CircuitBreaker("t", threshold, recovery, clock), clock


def boom():
    raise OSError("down")


def test_starts_closed_and_passes_result():
    b, _ = make()
    assert b.state == CLOSED
    assert b.call(lambda: 5) == 5


def test_opens_after_threshold_failures():
    b, _ = make(threshold=3)
    for _ in range(3):
        with pytest.raises(OSError):
            b.call(boom)
    assert b.state == OPEN
    with pytest.raises(CircuitOpenError):
        b.call(lambda: 1)


def test_success_resets_failure_count():
    b, _ = make(threshold=3)
    for _ in range(2):
        with pytest.raises(OSError):
            b.call(boom)
    b.call(lambda: 1)
    assert b.failures == 0
    for _ in range(2):
        with pytest.raises(OSError):
            b.call(boom)
    assert b.state == CLOSED


def test_half_open_after_recovery_allows_single_probe():
    b, clock = make(threshold=1, recovery=10.0)
    with pytest.raises(OSError):
        b.call(boom)
    clock.now = 9.9
    assert b.state == OPEN
    clock.now = 10.0
    assert b.state == HALF_OPEN
    assert b.allow() is True
    assert b.allow() is False


def test_probe_success_closes():
    b, clock = make(threshold=1)
    with pytest.raises(OSError):
        b.call(boom)
    clock.now = 11
    assert b.call(lambda: 1) == 1
    assert b.state == CLOSED


def test_probe_failure_reopens():
    b, clock = make(threshold=2)
    for _ in range(2):
        with pytest.raises(OSError):
            b.call(boom)
    clock.now = 11
    with pytest.raises(OSError):
        b.call(boom)
    assert b.state == OPEN
    with pytest.raises(CircuitOpenError):
        b.call(lambda: 1)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_seconds=-1)


def test_copilot_stops_calling_after_repeated_failures(monkeypatch):
    calls = []

    def fake_urlopen(*a, **k):
        calls.append(1)
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    copilot = MusicCopilot(provider="openai", api_key="k")
    ctx = MusicalContext()
    for _ in range(6):
        assert copilot.suggest(ctx, "x") == ""
    assert len(calls) == 3
    assert copilot._breakers["openai"].state == OPEN
    assert copilot._breakers["anthropic"].state == CLOSED


def test_copilot_recovers_after_success(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"response": "ok"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    copilot = MusicCopilot(provider="ollama")
    assert copilot.suggest(MusicalContext(), "x") == "ok"
    assert copilot._breakers["ollama"].failures == 0


def test_twitch_start_is_blocked_while_open():
    attempts = []

    def connect(timeout):
        attempts.append(1)
        raise OSError("no route")

    breaker = CircuitBreaker("twitch", 2, 60.0)
    messages = []
    client = TwitchClient(
        "chan",
        "tok",
        "nick",
        on_status=messages.append,
        connect=connect,
        breaker=breaker,
    )
    assert client.start() is False
    assert client.start() is False
    assert client.start() is False
    assert len(attempts) == 2
    assert "paused" in messages[-1]
