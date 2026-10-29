import pytest

from korshmer.collaboration.rate_limit import TokenBucket

NAN = float("nan")


@pytest.mark.parametrize("rate", [NAN, float("inf"), 0, -1.0, "5", None, True])
def test_rate_must_be_a_finite_positive_number(rate):
    with pytest.raises(ValueError, match="rate"):
        TokenBucket(rate, 5)


@pytest.mark.parametrize(
    "burst", [NAN, float("inf"), 0, -2, 2.7, "3", None, True]
)
def test_burst_must_be_a_whole_number_of_at_least_one(burst):
    with pytest.raises(ValueError, match="burst"):
        TokenBucket(5, burst)


def test_bucket_allows_a_burst_and_then_refills_with_time():
    now = [0.0]
    bucket = TokenBucket(2.0, 3, clock=lambda: now[0])
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]
    now[0] = 0.5
    assert bucket.allow() is True
    assert bucket.allow() is False


def test_bucket_never_holds_more_than_the_burst():
    now = [0.0]
    bucket = TokenBucket(100.0, 2, clock=lambda: now[0])
    now[0] = 1000.0
    assert [bucket.allow() for _ in range(3)] == [True, True, False]


@pytest.mark.parametrize("rate", [NAN, 0, -1, "5", None])
def test_server_rejects_an_invalid_message_rate_at_construction(rate):
    from korshmer.collaboration.server import CollaborationServer

    with pytest.raises(ValueError, match="rate"):
        CollaborationServer(max_messages_per_second=rate)


@pytest.mark.parametrize("burst", [0, 2.5, "3", None, NAN])
def test_server_rejects_an_invalid_burst_at_construction(burst):
    from korshmer.collaboration.server import CollaborationServer

    with pytest.raises(ValueError, match="burst"):
        CollaborationServer(burst=burst)
