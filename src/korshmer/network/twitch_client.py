from __future__ import annotations

import html
import logging
import math
import os
import re
import socket
import ssl
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from korshmer.network.circuit_breaker import CircuitBreaker
from korshmer.utils.audit_log import OUTCOME_DENIED, OUTCOME_FAILED, audit
from korshmer.utils.credential_store import get_secret

_logger = logging.getLogger(__name__)

HOST = "irc.chat.twitch.tv"
PORT = 6697


RATE_LIMIT_MESSAGES = 20
RATE_LIMIT_SECONDS = 30.0


MAX_MESSAGE_LENGTH = 500

DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 300.0

_CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{1,25}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


VIRTUAL_DEVICE_KEYWORDS = (
    "cable input",
    "vb-audio",
    "voicemeeter",
    "blackhole",
    "loopback",
    "virtual",
    "monitor of",
)


@dataclass(frozen=True)
class ChatMessage:
    user: str
    text: str
    channel: str = ""


@dataclass(frozen=True)
class IrcMessage:
    command: str
    params: tuple[str, ...] = ()
    prefix: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def nick(self) -> str:
        return self.prefix.split("!", 1)[0]


def parse_irc_line(line: str) -> IrcMessage | None:
    line = line.strip("\r\n")
    if not line.strip():
        return None
    tags: dict = {}
    prefix = ""
    if line.startswith("@"):
        raw_tags, _, line = line.partition(" ")
        for item in raw_tags[1:].split(";"):
            key, _, value = item.partition("=")
            tags[key] = value
    line = line.lstrip(" ")
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
    trailing = None
    if " :" in line:
        line, _, trailing = line.partition(" :")
    elif line.startswith(":"):
        trailing, line = line[1:], ""
    parts = line.split()
    if not parts:
        return None
    params = parts[1:]
    if trailing is not None:
        params.append(trailing)
    return IrcMessage(parts[0].upper(), tuple(params), prefix, tags)


def sanitize_chat_text(text: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
    text = _CONTROL_RE.sub("", str(text))[:limit]
    return html.escape(text, quote=True)


def valid_channel(name: str) -> bool:
    return bool(_CHANNEL_RE.match(name or ""))


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def normalize_channel(name: str) -> str:
    if name is not None and not isinstance(name, str):
        raise ValueError("invalid Twitch channel name")
    cleaned = (name or "").strip().lstrip("#")
    if not valid_channel(cleaned):
        raise ValueError("invalid Twitch channel name")
    return cleaned.lower()


def settings_from_env(environ=None) -> dict:
    env = os.environ if environ is None else environ
    token = env.get("KORSHMER_TWITCH_TOKEN", "")
    if not token and environ is None:
        token = get_secret("TWITCH_TOKEN")
    return {
        "channel": env.get("KORSHMER_TWITCH_CHANNEL", ""),
        "nick": env.get("KORSHMER_TWITCH_NICK", ""),
        "token": token,
    }


def find_virtual_output(devices: Iterable[dict], preferred: str | None = None):
    outputs = [
        (i, str(d.get("name", "")).lower())
        for i, d in enumerate(devices)
        if d.get("max_output_channels", 0) > 0
    ]
    if preferred:
        wanted = preferred.lower()
        for index, name in outputs:
            if wanted in name:
                return index
    for keyword in VIRTUAL_DEVICE_KEYWORDS:
        for index, name in outputs:
            if keyword in name:
                return index
    return None


class RateLimiter:
    def __init__(
        self,
        count=RATE_LIMIT_MESSAGES,
        seconds=RATE_LIMIT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"rate limit count must be a whole number of at least 1, got {count!r}"
            )
        if not _is_number(seconds) or seconds <= 0:
            raise ValueError(
                f"rate limit seconds must be a number above 0, got {seconds!r}"
            )
        self.count = count
        self.seconds = seconds
        self._clock = clock
        self._times: deque = deque()

    def allow(self) -> bool:
        now = self._clock()
        while self._times and now - self._times[0] >= self.seconds:
            self._times.popleft()
        if len(self._times) >= self.count:
            return False
        self._times.append(now)
        return True


def _default_connect(timeout: float):
    raw = socket.create_connection((HOST, PORT), timeout=timeout)
    context = ssl.create_default_context()
    return context.wrap_socket(raw, server_hostname=HOST)


class TwitchClient:
    def __init__(
        self,
        channel: str,
        token: str,
        nick: str,
        on_message: Callable[[ChatMessage], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        connect: Callable[[float], object] | None = None,
        rate_limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
    ):
        if not _is_number(timeout) or not 0 < timeout <= MAX_TIMEOUT:
            raise ValueError(
                "Twitch timeout must be a number of seconds above 0 and at "
                f"most {MAX_TIMEOUT}, got {timeout!r}"
            )
        self.channel = normalize_channel(channel)
        self.nick = (nick or "").strip().lower()
        if not valid_channel(self.nick):
            raise ValueError("invalid Twitch user name")
        token = (token or "").strip()
        if token.lower().startswith("oauth:"):
            token = token[6:]
        if not token or not re.match(r"^[A-Za-z0-9_\-]+$", token):
            audit(
                "twitch.credentials",
                OUTCOME_DENIED,
                reason="invalid_token_format",
                channel=self.channel,
            )
            raise ValueError("invalid Twitch access token")
        self._token = token
        self._on_message = on_message
        self._on_status = on_status
        self.timeout = timeout
        self._connect = connect or _default_connect
        self._limiter = rate_limiter or RateLimiter()
        self._breaker = breaker or CircuitBreaker("twitch")
        self._sock = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._connected = False
        self._stopping = False
        self._send_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    def _status(self, text: str) -> None:
        _logger.info(text)
        if self._on_status is not None:
            try:
                self._on_status(text)
            except Exception:
                _logger.debug("status callback failed", exc_info=True)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return self._connected
        self._ready.clear()
        self._connected = False
        self._stopping = False
        if not self._breaker.allow():
            self._status("Twitch connection paused after repeated failures.")
            return False
        try:
            self._sock = self._connect(self.timeout)
            self._sock.settimeout(self.timeout)
            self._write("PASS oauth:" + self._token)
            self._write("NICK " + self.nick)
            self._write("JOIN #" + self.channel)
        except Exception as e:
            self._close()
            self._breaker.record_failure()
            audit(
                "twitch.connect",
                OUTCOME_FAILED,
                channel=self.channel,
                error=type(e).__name__,
            )
            self._status(f"Twitch could not be reached ({type(e).__name__}).")
            return False
        self._thread = threading.Thread(
            target=self._run, name="korshmer-twitch", daemon=True
        )
        self._thread.start()
        self._ready.wait(self.timeout + 1.0)
        if self._connected:
            self._breaker.record_success()
        else:
            self._breaker.record_failure()
        return self._connected

    def stop(self) -> None:
        self._stopping = True
        self._close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.timeout + 1.0)
        self._thread = None
        self._connected = False

    def send_message(self, text: str) -> bool:
        text = _CONTROL_RE.sub("", str(text)).strip()[:MAX_MESSAGE_LENGTH]
        if not text or not self._connected:
            return False
        if not self._limiter.allow():
            self._status("Twitch chat rate limit reached.")
            return False
        try:
            self._write(f"PRIVMSG #{self.channel} :{text}")
        except Exception:
            return False
        return True

    def _write(self, line: str) -> None:
        data = (line + "\r\n").encode("utf-8")
        with self._send_lock:
            self._sock.sendall(data)

    def _close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    def _run(self) -> None:
        buffer = b""
        try:
            while not self._stopping:
                sock = self._sock
                if sock is None:
                    break
                try:
                    chunk = sock.recv(4096)
                except TimeoutError:
                    if not self._connected and not self._stopping:
                        self._status("Twitch did not answer the login.")
                        break
                    continue
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    self.handle_line(raw.decode("utf-8", "replace"))
                if len(buffer) > 65536:
                    break
        except Exception as e:
            if not self._stopping:
                _logger.debug("Twitch connection ended: %s", type(e).__name__)
        finally:
            was_connected = self._connected
            self._connected = False
            self._close()
            self._ready.set()
            if was_connected and not self._stopping:
                self._status("Twitch connection lost.")

    def handle_line(self, line: str) -> None:
        msg = parse_irc_line(line)
        if msg is None:
            return
        if msg.command == "PING":
            try:
                self._write("PONG :" + (msg.params[0] if msg.params else ""))
            except Exception:
                pass
        elif msg.command == "001":
            self._connected = True
            self._ready.set()
            audit("twitch.login", channel=self.channel, nick=self.nick)
            self._status(f"Twitch connected to #{self.channel}.")
        elif msg.command == "NOTICE":
            text = msg.params[-1] if msg.params else ""
            if (
                "authentication failed" in text.lower()
                or "improperly formatted auth" in text.lower()
            ):
                audit(
                    "twitch.login",
                    OUTCOME_DENIED,
                    reason="authentication_failed",
                    channel=self.channel,
                    nick=self.nick,
                )
                self._status("Twitch login failed: check the access token.")
                self._stopping = True
                self._ready.set()
                self._close()
            else:
                self._status("Twitch: " + sanitize_chat_text(text, 200))
        elif msg.command == "PRIVMSG" and len(msg.params) >= 2:
            if self._on_message is not None:
                chat = ChatMessage(
                    user=sanitize_chat_text(msg.nick, 50),
                    text=sanitize_chat_text(msg.params[-1]),
                    channel=self.channel,
                )
                try:
                    self._on_message(chat)
                except Exception:
                    _logger.debug("message callback failed", exc_info=True)
