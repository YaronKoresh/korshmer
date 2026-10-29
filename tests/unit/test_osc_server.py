import socket
import struct
import time

import pytest

from korshmer.network.osc_server import (
    OscServer,
    parse_message,
    parse_packet,
)


def _pad(b: bytes) -> bytes:
    b += b"\0"
    return b + b"\0" * (-len(b) % 4)


def osc(address: str, *args) -> bytes:
    tags = ","
    payload = b""
    for a in args:
        if isinstance(a, float):
            tags += "f"
            payload += struct.pack(">f", a)
        elif isinstance(a, int):
            tags += "i"
            payload += struct.pack(">i", a)
        else:
            tags += "s"
            payload += _pad(a.encode())
    return _pad(address.encode()) + _pad(tags.encode()) + payload


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a: self.calls.append((name, *a))


def _server(rec):
    return OscServer(
        on_play=lambda: rec.calls.append(("play",)),
        on_pause=lambda: rec.calls.append(("pause",)),
        on_stop=lambda: rec.calls.append(("stop",)),
        on_tempo=lambda v: rec.calls.append(("tempo", v)),
        port=0,
    )


def _wait(rec, n, timeout=2.0):
    end = time.time() + timeout
    while len(rec.calls) < n and time.time() < end:
        time.sleep(0.01)


def test_parse_message_float():
    assert parse_message(osc("/tempo", 140.0)) == ("/tempo", [140.0])


def test_parse_message_without_args():
    assert parse_message(_pad(b"/transport/play")) == ("/transport/play", [])


def test_parse_rejects_garbage():
    for data in (b"", b"abc", b"\xff\xfe\x00\x00", osc("/x", 1)[:-2]):
        with pytest.raises(ValueError):
            parse_message(data)


def test_parse_bundle():
    a, b = osc("/tempo", 100.0), osc("/transport/stop")
    bundle = (
        b"#bundle\0"
        + b"\0" * 8
        + struct.pack(">i", len(a))
        + a
        + struct.pack(">i", len(b))
        + b
    )
    assert [m[0] for m in parse_packet(bundle)] == ["/tempo", "/transport/stop"]


def test_udp_tempo_message_reaches_callback():
    rec = Recorder()
    server = _server(rec)
    assert server.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(osc("/tempo", 140.0), ("127.0.0.1", server.port))
        _wait(rec, 1)
    finally:
        server.stop()
    assert rec.calls == [("tempo", 140.0)]


def test_out_of_range_tempo_is_clamped():
    rec = Recorder()
    server = _server(rec)
    server.handle_packet(osc("/tempo", 999.0))
    assert rec.calls == [("tempo", 300.0)]


def test_transport_messages():
    rec = Recorder()
    server = _server(rec)
    for name in ("play", "pause", "stop"):
        server.handle_packet(_pad(f"/transport/{name}".encode()))
    assert rec.calls == [("play",), ("pause",), ("stop",)]


def test_invalid_input_is_ignored(caplog):
    rec = Recorder()
    server = _server(rec)
    for data in (
        b"\x00\x01garbage",
        osc("/tempo", "fast"),
        osc("/tempo", float("nan")),
        osc("/tempo"),
        osc("/unknown", 1),
        b"x" * 10000,
    ):
        server.handle_packet(data)
    assert rec.calls == []


def test_server_survives_invalid_datagram():
    rec = Recorder()
    server = _server(rec)
    assert server.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(b"\xff\x00\x01", ("127.0.0.1", server.port))
            s.sendto(osc("/transport/play"), ("127.0.0.1", server.port))
        _wait(rec, 1)
    finally:
        server.stop()
    assert rec.calls == [("play",)]


def test_default_host_is_loopback():
    assert OscServer().host == "127.0.0.1"


def test_port_in_use_returns_false():
    first = _server(Recorder())
    assert first.start()
    try:
        second = OscServer(port=first.port)
        assert not second.start()
    finally:
        first.stop()


def _bundle_of(message: bytes, count: int) -> bytes:
    element = struct.pack(">i", len(message)) + message
    return b"#bundle\0" + b"\0" * 8 + element * count


def test_parse_packet_rejects_bundle_with_too_many_messages():
    from korshmer.network.osc_server import MAX_MESSAGES_PER_PACKET

    message = osc("/transport/play")
    data = _bundle_of(message, MAX_MESSAGES_PER_PACKET + 1)
    with pytest.raises(ValueError, match="too many messages"):
        parse_packet(data)


def test_parse_packet_accepts_bundle_at_message_limit():
    from korshmer.network.osc_server import MAX_MESSAGES_PER_PACKET

    data = _bundle_of(osc("/transport/play"), MAX_MESSAGES_PER_PACKET)
    assert len(parse_packet(data)) == MAX_MESSAGES_PER_PACKET


def test_handle_packet_ignores_flooding_bundle():
    rec = Recorder()
    server = _server(rec)
    server.handle_packet(_bundle_of(osc("/transport/play"), 300))
    assert rec.calls == []
