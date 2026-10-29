import json
import socket
import sys
import urllib.error
import urllib.request

import pytest

from korshmer.network import health_server as health_server_module
from korshmer.network.health_server import (
    CONTENT_SECURITY_POLICY,
    PORT_ENV_VAR,
    SECURITY_HEADERS,
    HealthServer,
    port_from_env,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _get(server, path):
    url = f"http://{server.host}:{server.port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.headers, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.headers, json.loads(e.read())


@pytest.fixture
def server():
    srv = HealthServer(port=0)
    yield srv
    srv.stop()


class TestPortFromEnv:
    def test_unset_disables_server(self):
        assert port_from_env({}) is None
        assert port_from_env({PORT_ENV_VAR: "  "}) is None

    def test_valid_port(self):
        assert port_from_env({PORT_ENV_VAR: "8765"}) == 8765
        assert port_from_env({PORT_ENV_VAR: "0"}) == 0

    @pytest.mark.parametrize("raw", ["abc", "-1", "65536", "80.5"])
    def test_invalid_port_is_ignored(self, raw):
        assert port_from_env({PORT_ENV_VAR: raw}) is None


class TestSnapshot:
    def test_no_heartbeat_is_degraded(self):
        snap = HealthServer(clock=FakeClock()).snapshot()
        assert snap["status"] == "degraded"
        assert snap["components"]["main_thread"]["status"] == "degraded"

    def test_fresh_heartbeat_is_ok_and_keeps_state(self):
        clock = FakeClock()
        health = HealthServer(clock=clock)
        health.beat(processing=False, playback="StoppedState")
        clock.now += 1
        snap = health.snapshot()
        assert snap["status"] == "ok"
        assert snap["uptime_seconds"] == 1.0
        assert snap["state"] == {
            "processing": False,
            "playback": "StoppedState",
        }

    def test_stale_heartbeat_is_error(self):
        clock = FakeClock()
        health = HealthServer(stale_after=5.0, clock=clock)
        health.beat()
        clock.now += 6
        snap = health.snapshot()
        assert snap["status"] == "error"
        main = snap["components"]["main_thread"]
        assert main["heartbeat_age_seconds"] == 6.0

    def test_checks_combine_into_overall_status(self):
        health = HealthServer(clock=FakeClock())
        health.beat()
        health.register_check("disk", lambda: True)
        health.register_check("gpu", lambda: {"status": "degraded"})
        assert health.snapshot()["status"] == "degraded"
        health.register_check("engine", lambda: False)
        snap = health.snapshot()
        assert snap["status"] == "error"
        assert snap["components"]["disk"] == {"status": "ok"}
        health.unregister_check("engine")
        assert health.snapshot()["status"] == "degraded"

    def test_raising_check_reports_error(self):
        health = HealthServer(clock=FakeClock())
        health.beat()

        def broken():
            raise RuntimeError("device gone")

        health.register_check("audio", broken)
        snap = health.snapshot()
        assert snap["status"] == "error"
        assert snap["components"]["audio"]["detail"] == (
            "RuntimeError: device gone"
        )

    def test_non_json_values_are_converted(self):
        class Scalar:
            def item(self):
                return 1.5

        health = HealthServer(clock=FakeClock())
        health.beat(cpu=Scalar(), path=object.__new__(object))
        health.register_check("value", lambda: 3)
        snap = health.snapshot()
        json.dumps(snap)
        assert snap["state"]["cpu"] == 1.5
        assert snap["components"]["value"] == {"status": "ok", "value": 3}

    def test_rejects_non_positive_stale_after(self):
        with pytest.raises(ValueError):
            HealthServer(stale_after=0)


class TestHttp:
    def test_binds_localhost_and_reports_real_port(self, server):
        assert server.start()
        assert server.running
        assert server.host == "127.0.0.1"
        assert server.port > 0

    def test_liveness_endpoint(self, server):
        assert server.start()
        status, headers, body = _get(server, "/health")
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert body["status"] == "ok"

    @pytest.mark.parametrize(
        "path", ["/health", "/health/state", "/health/metrics", "/admin"]
    )
    def test_responses_carry_security_headers(self, server, path):
        assert server.start()
        _, headers, _ = _get(server, path)
        for name, value in SECURITY_HEADERS:
            assert headers[name] == value
        assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY

    def test_policy_blocks_all_content_sources(self):
        directives = dict(
            part.split(" ", 1) for part in CONTENT_SECURITY_POLICY.split("; ")
        )
        assert directives["default-src"] == "'none'"
        assert directives["frame-ancestors"] == "'none'"
        assert "unsafe-inline" not in CONTENT_SECURITY_POLICY
        assert "unsafe-eval" not in CONTENT_SECURITY_POLICY

    def test_stream_and_error_pages_carry_security_headers(self, server):
        assert server.start()
        url = f"http://{server.host}:{server.port}/health/logs/stream"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.headers["Content-Security-Policy"] == (
                CONTENT_SECURITY_POLICY
            )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(url, timeout=5)
        assert err.value.code == 404
        assert err.value.headers["Content-Security-Policy"] == (
            CONTENT_SECURITY_POLICY
        )
        assert err.value.headers["X-Content-Type-Options"] == "nosniff"

    def test_state_endpoint_ok(self, server):
        server.beat(processing=True)
        assert server.start()
        status, _, body = _get(server, "/health/state/")
        assert status == 200
        assert body["status"] == "ok"
        assert body["state"]["processing"] is True

    def test_state_endpoint_returns_503_on_error(self, server):
        server.beat()
        server.register_check("engine", lambda: False)
        assert server.start()
        status, _, body = _get(server, "/health/state")
        assert status == 503
        assert body["components"]["engine"]["status"] == "error"

    def test_unknown_path_is_404(self, server):
        assert server.start()
        status, _, body = _get(server, "/admin")
        assert status == 404
        assert body["detail"] == "not found"

    def test_start_twice_and_stop(self, server):
        assert server.start()
        port = server.port
        assert server.start()
        assert server.port == port
        server.stop()
        assert not server.running
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)

    def test_port_in_use_returns_false(self, server):
        assert server.start()
        other = HealthServer(port=server.port)
        try:
            assert other.start() is False
            assert not other.running
        finally:
            other.stop()

    @pytest.mark.parametrize(
        "platform, allowed",
        [("win32", False), ("linux", True), ("darwin", True)],
    )
    def test_address_reuse_is_not_allowed_on_windows(
        self, monkeypatch, platform, allowed
    ):
        monkeypatch.setattr(sys, "platform", platform)
        assert health_server_module._reuse_address_allowed() is allowed

    @pytest.mark.parametrize("allowed", [True, False])
    def test_bind_follows_the_address_reuse_policy(
        self, server, monkeypatch, allowed
    ):
        monkeypatch.setattr(
            health_server_module, "_reuse_address_allowed", lambda: allowed
        )
        assert server.start()
        option = server._server.socket.getsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR
        )
        assert bool(option) is allowed
