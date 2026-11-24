import http.client
import socket

import pytest

from korshmer.network import health_server
from korshmer.network.health_server import HealthServer
from korshmer.utils.log_stream import LogStreamHandler

STREAM_LIMIT = 8


@pytest.fixture
def server():
    srv = HealthServer(port=0, log_stream=LogStreamHandler(capacity=10))
    assert srv.start()
    yield srv
    srv.stop()


def status_for_host(server, host, path="/health/logs", method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True)
        if host is not None:
            conn.putheader("Host", host)
        conn.endheaders()
        return conn.getresponse().status
    finally:
        conn.close()


def open_stream(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request(
        "GET",
        "/health/logs/stream",
        headers={"Host": f"127.0.0.1:{server.port}"},
    )
    return conn, conn.getresponse().status


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost", "[::1]"])
def test_requests_addressed_to_this_computer_are_answered_as_a_safeguard(
    server, name
):
    assert status_for_host(server, f"{name}:{server.port}") == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example.com",
        "evil.example.com:{port}",
        "127.0.0.1.evil.example.com:{port}",
        "localhost:1",
        "",
    ],
)
def test_requests_addressed_to_another_site_name_are_refused(server, host):
    assert status_for_host(server, host.format(port=server.port)) == 403


def test_a_request_without_a_host_name_is_refused(server):
    assert status_for_host(server, None) == 403


def test_the_live_log_stream_refuses_another_site_name(server):
    path = "/health/logs/stream"
    assert status_for_host(server, "evil.example.com", path=path) == 403


def test_a_post_addressed_to_another_site_name_is_refused(server):
    assert status_for_host(server, "evil.example.com", method="POST") == 403


def test_a_host_name_without_a_port_is_enough_on_the_standard_web_port():
    standard = HealthServer(port=80)
    assert standard.host_allowed("localhost")
    assert standard.host_allowed("127.0.0.1:80")
    assert not standard.host_allowed("evil.example.com")


def test_a_host_name_without_a_port_is_refused_on_any_other_port():
    assert not HealthServer(port=8123).host_allowed("localhost")


def test_a_server_bound_to_a_network_address_answers_to_that_address():
    lan = HealthServer(host="192.168.1.20", port=8123)
    assert lan.host_allowed("192.168.1.20:8123")
    assert not lan.host_allowed("192.168.1.21:8123")
    assert not lan.host_allowed("evil.example.com:8123")


def test_the_live_log_stream_limits_how_many_listeners_connect(server):
    open_streams = []
    try:
        for _ in range(STREAM_LIMIT):
            conn, status = open_stream(server)
            open_streams.append(conn)
            assert status == 200
        extra, status = open_stream(server)
        open_streams.append(extra)
        assert status == 503
    finally:
        for conn in open_streams:
            conn.close()


def test_listener_places_are_free_again_after_the_server_restarts_as_a_safeguard(
    server,
):
    conns = [open_stream(server)[0] for _ in range(STREAM_LIMIT)]
    server.stop()
    for conn in conns:
        conn.close()
    assert server.start()
    conn, status = open_stream(server)
    conn.close()
    assert status == 200


def test_a_silent_connection_is_dropped_after_the_timeout(monkeypatch):
    monkeypatch.setattr(
        health_server, "REQUEST_TIMEOUT_SECONDS", 0.2, raising=False
    )
    srv = HealthServer(port=0)
    assert srv.start()
    try:
        with socket.create_connection(("127.0.0.1", srv.port), 5) as sock:
            sock.settimeout(2)
            try:
                dropped = sock.recv(1) == b""
            except TimeoutError:
                dropped = False
    finally:
        srv.stop()
    assert dropped
