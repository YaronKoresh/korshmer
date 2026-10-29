import asyncio
import json
import socket
import struct
import urllib.error
import urllib.request

import pytest

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import CollaborationServer
from korshmer.network import health_server, osc_server
from korshmer.network.health_server import SECURITY_HEADERS, HealthServer
from korshmer.network.osc_server import OscServer


class ScriptedWS:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    def __aiter__(self):
        async def gen():
            for m in self.messages:
                yield m

        return gen()

    async def send(self, message):
        self.sent.append(message)

    async def close(self, code=1000, reason=""):
        pass


def _op(session_id, size=0):
    return Operation(
        op_type=OpType.CURSOR_MOVE.value,
        session_id=session_id,
        peer_id="x",
        data={"pad": "a" * size},
    ).to_json()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _errors(ws):
    return [
        json.loads(m)["data"]["error"]
        for m in ws.sent
        if json.loads(m)["op_type"] == OpType.ERROR.value
    ]


def _server(**kw):
    kw.setdefault("max_messages_per_second", 1e9)
    kw.setdefault("burst", 10**9)
    return CollaborationServer(**kw)


class TestCollaborationLimits:
    def test_session_count_is_capped(self):
        s = _server(max_sessions=3)
        ws = ScriptedWS([_op(f"s{i}") for i in range(10)])
        _run(s._handle_client(ws))
        assert len(s.sessions) <= 3
        assert _errors(ws).count("Too many sessions") == 7

    def test_operation_history_is_capped(self):
        s = _server(max_operations_per_session=5)
        s.create_session("s1")
        _run(s._handle_client(ScriptedWS([_op("s1")] * 40)))
        assert len(s.sessions["s1"]["operations"]) == 5

    def test_peers_per_session_are_capped(self):
        s = _server(max_peers_per_session=2)
        sid = s.create_session("s1")
        s.peers[sid]["A"] = ScriptedWS([])
        s.peers[sid]["B"] = ScriptedWS([])
        ws = ScriptedWS([_op("s1")])
        _run(s._handle_client(ws))
        assert sorted(s.peers[sid]) == ["A", "B"]
        assert _errors(ws) == ["Session is full"]

    def test_oversized_message_is_refused_before_parsing(self):
        s = _server(max_message_bytes=2000)
        s.create_session("s1")
        ws = ScriptedWS([_op("s1", size=5000), _op("s1")])
        _run(s._handle_client(ws))
        assert _errors(ws) == ["Message too large"]
        assert len(s.sessions["s1"]["operations"]) == 1

    def test_session_created_by_a_client_is_removed_when_it_empties(self):
        s = _server()
        _run(s._handle_client(ScriptedWS([_op("tmp")])))
        assert "tmp" not in s.sessions
        assert "tmp" not in s.peers

    def test_session_created_by_the_host_survives_empty(self):
        s = _server()
        s.create_session("host")
        _run(s._handle_client(ScriptedWS([_op("host")])))
        assert "host" in s.sessions

    @pytest.mark.parametrize(
        "name",
        [
            "max_sessions",
            "max_peers_per_session",
            "max_operations_per_session",
            "max_message_bytes",
        ],
    )
    def test_limits_must_be_positive(self, name):
        with pytest.raises(ValueError):
            CollaborationServer(**{name: 0})


class TestHealthServerExposure:
    @pytest.fixture
    def server(self):
        srv = HealthServer(port=0)
        assert srv.start()
        yield srv
        srv.stop()

    def _request(self, server, method, path):
        req = urllib.request.Request(
            f"http://{server.host}:{server.port}{path}",
            method=method,
            data=b"x" if method in ("POST", "PUT") else None,
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.headers

    def test_listens_on_loopback_by_default(self, server):
        assert health_server.DEFAULT_HOST == "127.0.0.1"
        assert server.host == "127.0.0.1"

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_state_changing_methods_are_refused(self, server, method):
        code, headers = self._request(server, method, "/health")
        assert code == 501
        for name, value in SECURITY_HEADERS:
            assert headers[name] == value

    @pytest.mark.parametrize(
        "path",
        ["/", "/etc/passwd", "/health/../../etc/passwd", "/health/x", "//"],
    )
    def test_unknown_paths_return_404(self, server, path):
        code, _ = self._request(server, "GET", path)
        assert code == 404


class TestOscExposure:
    def test_listens_on_loopback_by_default(self):
        assert osc_server.DEFAULT_HOST == "127.0.0.1"
        assert OscServer().host == "127.0.0.1"

    def test_malformed_datagrams_trigger_no_action(self):
        calls = []
        srv = OscServer(
            on_play=lambda: calls.append("play"),
            on_tempo=lambda v: calls.append(v),
            host="127.0.0.1",
            port=0,
        )
        assert srv.start()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for data in (
                b"",
                b"\xff" * 64,
                b"/transport/play",
                b"/tempo\0\0,f\0\0",
                b"/tempo\0\0,d\0\0" + struct.pack(">d", float("nan")),
                b"/a" * (osc_server.MAX_PACKET_SIZE),
                b"#bundle\0" + b"\0" * 8 + struct.pack(">i", 10**6),
            ):
                sock.sendto(data, (srv.host, srv.port))
            sock.close()
            srv.stop()
        finally:
            srv.stop()
        assert calls == []
