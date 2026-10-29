import threading

import pytest

from korshmer.network.twitch_client import (
    ChatMessage,
    RateLimiter,
    TwitchClient,
    find_virtual_output,
    normalize_channel,
    parse_irc_line,
    sanitize_chat_text,
    settings_from_env,
)


class FakeSocket:
    def __init__(self, script):
        self.sent = []
        self._script = list(script)
        self._cond = threading.Condition()
        self.closed = False

    def settimeout(self, t):
        pass

    def sendall(self, data):
        self.sent.append(data.decode())

    def feed(self, data):
        with self._cond:
            self._script.append(data)
            self._cond.notify_all()

    def recv(self, n):

        with self._cond:
            if not self._script and not self.closed:
                self._cond.wait(0.05)
            if self.closed:
                return b""
            if not self._script:
                raise TimeoutError()
            return self._script.pop(0)

    def close(self):
        with self._cond:
            self.closed = True
            self._cond.notify_all()


def make(sock, **kw):
    return TwitchClient(
        "#Chan", "oauth:abc123", "Me", connect=lambda t: sock, timeout=1.0, **kw
    )


def test_parse_privmsg_with_tags():
    m = parse_irc_line(
        "@badges=;id=1 :bob!bob@bob.tmi.twitch.tv PRIVMSG #chan :hi there"
    )
    assert m.command == "PRIVMSG" and m.nick == "bob"
    assert m.params == ("#chan", "hi there") and m.tags["id"] == "1"


def test_incoming_message_reaches_callback():
    got = []
    sock = FakeSocket([b":tmi 001 me :Welcome\r\n"])
    c = make(sock, on_message=got.append)
    assert c.start()
    sock.feed(b":bob!bob@x PRIVMSG #chan :hello\r\n")
    for _ in range(50):
        if got:
            break
        threading.Event().wait(0.05)
    c.stop()
    assert got == [ChatMessage("bob", "hello", "chan")]
    assert sock.sent[:3] == [
        "PASS oauth:abc123\r\n",
        "NICK me\r\n",
        "JOIN #chan\r\n",
    ]


def test_ping_answered_and_send_message():
    sock = FakeSocket([b":tmi 001 me :Welcome\r\n", b"PING :tmi.twitch.tv\r\n"])
    c = make(sock)
    assert c.start()
    for _ in range(50):
        if any(s.startswith("PONG") for s in sock.sent):
            break
        threading.Event().wait(0.05)
    assert "PONG :tmi.twitch.tv\r\n" in sock.sent
    assert c.send_message("hi\x00 all")
    assert "PRIVMSG #chan :hi all\r\n" in sock.sent
    c.stop()


def test_failed_login_reports_and_does_not_raise():
    status = []
    sock = FakeSocket([b":tmi NOTICE * :Login authentication failed\r\n"])
    c = make(sock, on_status=status.append)
    assert c.start() is False
    assert any("login failed" in s for s in status)
    assert all("abc123" not in s for s in status)
    c.stop()


def test_unreachable_server_returns_false():
    status = []

    def boom(t):
        raise OSError("down")

    c = TwitchClient("chan", "tok", "me", connect=boom, on_status=status.append)
    assert c.start() is False
    assert status and "could not be reached" in status[0]


def test_rate_limit():
    now = [0.0]
    r = RateLimiter(2, 10, clock=lambda: now[0])
    assert r.allow() and r.allow() and not r.allow()
    now[0] = 10.0
    assert r.allow()


def test_sanitize_and_validation():
    assert sanitize_chat_text("<b>x\x07</b>") == "&lt;b&gt;x&lt;/b&gt;"
    assert len(sanitize_chat_text("a" * 1000)) == 500
    assert normalize_channel("#Chan_1") == "chan_1"
    with pytest.raises(ValueError):
        normalize_channel("bad name\r\nJOIN #x")
    with pytest.raises(ValueError):
        TwitchClient("chan", "", "me")


def test_find_virtual_output():
    devices = [
        {"name": "Mic", "max_output_channels": 0},
        {"name": "Speakers", "max_output_channels": 2},
        {"name": "CABLE Input (VB-Audio)", "max_output_channels": 2},
    ]
    assert find_virtual_output(devices) == 2
    assert find_virtual_output(devices, "speak") == 1
    assert find_virtual_output(devices[:2]) is None


def test_settings_from_env():
    s = settings_from_env({"KORSHMER_TWITCH_CHANNEL": "c"})
    assert s == {"channel": "c", "nick": "", "token": ""}
