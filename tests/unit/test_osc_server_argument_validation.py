import socket

import pytest

from korshmer.network.osc_server import OscServer


@pytest.mark.parametrize("port", [70000, -1, 65536, "abc", None, 1.5, True])
def test_port_must_be_a_whole_number_between_0_and_65535(port):
    with pytest.raises(ValueError, match="port"):
        OscServer(port=port)


@pytest.mark.parametrize("host", [None, 5, "", " ", "a b", "x\ny", "-h"])
def test_host_must_be_a_plain_address(host):
    with pytest.raises(ValueError, match="host"):
        OscServer(host=host)


def test_port_zero_binds_a_free_port():
    server = OscServer(port=0)
    try:
        assert server.start() is True
        assert 1 <= server.port <= 65535
    finally:
        server.stop()


def test_a_taken_port_still_returns_false():
    taken = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    taken.bind(("127.0.0.1", 0))
    try:
        server = OscServer(port=taken.getsockname()[1])
        assert server.start() is False
    finally:
        taken.close()


def test_default_settings_are_accepted():
    server = OscServer()
    assert (server.host, server.port) == ("127.0.0.1", 8000)
