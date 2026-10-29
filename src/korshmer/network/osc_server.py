from __future__ import annotations

import logging
import math
import re
import socket
import struct
import threading
from collections.abc import Callable

from korshmer.utils.tempo import validate_bpm

_logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PACKET_SIZE = 8192
_MAX_BUNDLE_DEPTH = 4
MAX_MESSAGES_PER_PACKET = 32

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")


def _read_string(data: bytes, pos: int) -> tuple[str, int]:
    end = data.find(b"\0", pos)
    if end < 0:
        raise ValueError("Unterminated OSC string")
    text = data[pos:end].decode("utf-8")
    return text, (end + 4) & ~3


def _unpack(fmt: str, data: bytes, pos: int) -> tuple[object, int]:
    size = struct.calcsize(fmt)
    if pos + size > len(data):
        raise ValueError("Truncated OSC argument")
    return struct.unpack_from(fmt, data, pos)[0], pos + size


def parse_message(data: bytes) -> tuple[str, list]:
    if not data or data[0:1] != b"/":
        raise ValueError("OSC address must start with '/'")
    address, pos = _read_string(data, 0)
    args: list = []
    if pos >= len(data):
        return address, args
    tags, pos = _read_string(data, pos)
    if not tags.startswith(","):
        raise ValueError("Missing OSC type tag string")
    for tag in tags[1:]:
        if tag == "i":
            value, pos = _unpack(">i", data, pos)
        elif tag == "f":
            value, pos = _unpack(">f", data, pos)
        elif tag == "h":
            value, pos = _unpack(">q", data, pos)
        elif tag == "d":
            value, pos = _unpack(">d", data, pos)
        elif tag == "s":
            value, pos = _read_string(data, pos)
        elif tag == "T":
            value = True
        elif tag == "F":
            value = False
        elif tag == "N":
            value = None
        else:
            raise ValueError(f"Unsupported OSC type tag {tag!r}")
        args.append(value)
    return address, args


def parse_packet(data: bytes, _depth: int = 0) -> list[tuple[str, list]]:
    if len(data) > MAX_PACKET_SIZE:
        raise ValueError("OSC packet too large")
    if not data.startswith(b"#bundle\0"):
        return [parse_message(data)]
    if _depth >= _MAX_BUNDLE_DEPTH:
        raise ValueError("OSC bundles nested too deeply")
    pos = 16
    if len(data) < pos:
        raise ValueError("Truncated OSC bundle")
    messages = []
    while pos < len(data):
        size, pos = _unpack(">i", data, pos)
        if size < 0 or pos + size > len(data):
            raise ValueError("Invalid OSC bundle element size")
        messages.extend(parse_packet(data[pos : pos + size], _depth + 1))
        if len(messages) > MAX_MESSAGES_PER_PACKET:
            raise ValueError("OSC bundle holds too many messages")
        pos += size
    return messages


class OscServer:
    def __init__(
        self,
        on_play: Callable[[], None] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_tempo: Callable[[float], None] | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError(
                f"port must be a whole number from 0 to 65535, got {port!r}"
            )
        if not isinstance(host, str) or not _HOST_RE.match(host):
            raise ValueError(
                f"host must be an IP address or a host name, got {host!r}"
            )
        self.host = host
        self.port = port
        self._actions = {
            "/transport/play": on_play,
            "/transport/pause": on_pause,
            "/transport/stop": on_stop,
        }
        self._on_tempo = on_tempo
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.running:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            _logger.error(
                "OSC server cannot listen on %s:%s: %s",
                self.host,
                self.port,
                e,
            )
            sock.close()
            return False
        sock.settimeout(0.2)
        self.port = sock.getsockname()[1]
        self._socket = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="korshmer-osc", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, sock = self._thread, self._socket
        self._thread = self._socket = None
        if thread is not None:
            thread.join(2)
        if sock is not None:
            sock.close()

    def _run(self) -> None:
        sock = self._socket
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(MAX_PACKET_SIZE + 1)
            except TimeoutError:
                continue
            except OSError:
                break
            self.handle_packet(data)

    def handle_packet(self, data: bytes) -> None:
        try:
            messages = parse_packet(data)
        except (ValueError, UnicodeDecodeError, struct.error) as e:
            _logger.warning("Ignoring invalid OSC packet: %s", e)
            return
        for address, args in messages:
            try:
                self._dispatch(address, args)
            except Exception as e:
                _logger.error("OSC handler for %s failed: %s", address, e)

    def _dispatch(self, address: str, args: list) -> None:
        if address == "/tempo":
            self._dispatch_tempo(args)
            return
        if address in self._actions:
            callback = self._actions[address]
            if callback is not None:
                callback()
            return
        _logger.warning("Ignoring unknown OSC address %s", address)

    def _dispatch_tempo(self, args: list) -> None:
        if len(args) != 1 or isinstance(args[0], (bool, str, type(None))):
            _logger.warning("Ignoring /tempo without one numeric argument")
            return
        value = float(args[0])
        if not math.isfinite(value) or value <= 0:
            _logger.warning("Ignoring invalid /tempo value %r", args[0])
            return
        if self._on_tempo is not None:
            self._on_tempo(validate_bpm(value))
