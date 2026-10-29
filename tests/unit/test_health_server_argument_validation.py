import pytest

from korshmer.network.health_server import STATUS_OK, HealthServer


@pytest.mark.parametrize(
    "stale", [float("nan"), float("inf"), 0, -1.0, "5", None, True]
)
def test_stale_limit_must_be_a_finite_positive_number(stale):
    with pytest.raises(ValueError, match="stale_after"):
        HealthServer(stale_after=stale)


@pytest.mark.parametrize("port", ["80", -5, 70000, 80.5, None, True])
def test_port_must_be_a_whole_number_in_range(port):
    with pytest.raises(ValueError, match="port"):
        HealthServer(port=port)


@pytest.mark.parametrize("port", [0, 80, 65535])
def test_valid_ports_are_kept(port):
    assert HealthServer(port=port).port == port


def test_a_fresh_heartbeat_is_ok_with_an_integer_limit():
    server = HealthServer(stale_after=5)
    server.beat()
    main = server.snapshot()["components"]["main_thread"]
    assert main["status"] == STATUS_OK
