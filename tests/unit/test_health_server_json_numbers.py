import json
import urllib.request

import pytest

from korshmer.network.health_server import HealthServer
from korshmer.utils.perf_metrics import metrics


def strict_json(raw):
    def refuse(token):
        raise ValueError(f"not valid JSON: {token}")

    return json.loads(raw, parse_constant=refuse)


def fetch(server, path):
    url = f"http://{server.host}:{server.port}{path}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read().decode("utf-8")


@pytest.fixture
def server():
    srv = HealthServer(port=0)
    yield srv
    srv.stop()


def test_state_with_non_finite_numbers_is_valid_json(server):
    server.beat(level=float("nan"), peak=float("inf"), ok=3.5)
    server.register_check("c", lambda: {"status": "ok", "x": float("-inf")})
    assert server.start()
    body = strict_json(fetch(server, "/health/state"))
    assert body["state"] == {"level": None, "peak": None, "ok": 3.5}
    assert body["components"]["c"]["x"] is None


def test_a_check_returning_a_bare_nan_is_valid_json(server):
    server.register_check("c", lambda: float("nan"))
    assert server.start()
    body = strict_json(fetch(server, "/health/state"))
    assert body["components"]["c"]["value"] is None


def test_metrics_endpoint_is_valid_json_even_with_infinite_values(server):
    metrics.reset()
    try:
        metrics.record("health.test", 0.5)
        stat = metrics._stats["health.test"]
        stat.minimum = float("inf")
        stat.total = float("nan")
        assert server.start()
        strict_json(fetch(server, "/health/metrics"))
    finally:
        metrics.reset()
