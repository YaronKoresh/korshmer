import pytest

from korshmer.network.twitch_client import (
    RateLimiter,
    TwitchClient,
    normalize_channel,
)

NAN = float("nan")


@pytest.mark.parametrize(
    "timeout", [NAN, float("inf"), -3, 0, "5", None, True, 301]
)
def test_timeout_must_be_a_bounded_positive_number(timeout):
    with pytest.raises(ValueError, match="timeout"):
        TwitchClient("chan", "tok", "nick", timeout=timeout)


@pytest.mark.parametrize("timeout", [0.5, 3, 300])
def test_valid_timeouts_are_kept(timeout):
    client = TwitchClient("chan", "tok", "nick", timeout=timeout)
    assert client.timeout == timeout


@pytest.mark.parametrize(
    "count", [0, -1, "20", None, True, 2.5, NAN, float("inf")]
)
def test_rate_limit_count_must_be_a_positive_whole_number(count):
    with pytest.raises(ValueError, match="count"):
        RateLimiter(count, 30.0)


@pytest.mark.parametrize("seconds", [NAN, float("inf"), -5, 0, "x", None, True])
def test_rate_limit_window_must_be_a_positive_finite_number(seconds):
    with pytest.raises(ValueError, match="seconds"):
        RateLimiter(20, seconds)


def test_valid_rate_limiter_settings_work():
    limiter = RateLimiter(1, 0.5, clock=lambda: 0.0)
    assert limiter.allow() is True
    assert limiter.allow() is False
    assert RateLimiter().count == 20


@pytest.mark.parametrize("name", [5, None, ["a"], b"chan", 1.5])
def test_channel_that_is_not_text_is_a_value_error(name):
    with pytest.raises(ValueError, match="channel"):
        normalize_channel(name)


def test_channel_with_hash_is_still_normalized():
    assert normalize_channel(" #Chan_1 ") == "chan_1"
