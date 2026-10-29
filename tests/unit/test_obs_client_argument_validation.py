import pytest

from korshmer.network.obs_client import ObsClient

NAN = float("nan")


@pytest.mark.parametrize(
    "timeout", [NAN, float("inf"), -3, 0, "5", None, True, 301]
)
def test_timeout_must_be_a_bounded_positive_number(timeout):
    with pytest.raises(ValueError, match="timeout"):
        ObsClient(timeout=timeout)


@pytest.mark.parametrize("timeout", [0.5, 3, 300])
def test_valid_timeouts_are_kept(timeout):
    assert ObsClient(timeout=timeout).timeout == timeout


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:4455", "", None, "ws://", "not a url", "ws://[bad"],
)
def test_start_reports_an_address_that_is_not_a_websocket_url(url):
    messages = []
    client = ObsClient(url=url, on_status=messages.append)
    assert client.start() is False
    assert any("ws://" in m for m in messages)
    assert client._thread is None


def test_default_and_secure_addresses_are_accepted():
    assert ObsClient().url == "ws://127.0.0.1:4455"
    assert ObsClient(url="wss://example.org:4455").url.startswith("wss://")
