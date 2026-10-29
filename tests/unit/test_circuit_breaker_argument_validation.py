import pytest

from korshmer.network.circuit_breaker import (
    CLOSED,
    OPEN,
    CircuitBreaker,
)

NAN = float("nan")


@pytest.mark.parametrize(
    "threshold", [NAN, float("inf"), "3", None, True, 2.5, 0, -1]
)
def test_failure_threshold_must_be_a_whole_number_of_at_least_one(threshold):
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=threshold)


@pytest.mark.parametrize("seconds", [NAN, float("inf"), "30", None, True, -1])
def test_recovery_seconds_must_be_a_finite_non_negative_number(seconds):
    with pytest.raises(ValueError, match="recovery_seconds"):
        CircuitBreaker(recovery_seconds=seconds)


@pytest.mark.parametrize("seconds", [0, 0.5, 30, 3600.0])
def test_valid_recovery_seconds_are_kept(seconds):
    assert CircuitBreaker(recovery_seconds=seconds).recovery_seconds == seconds


def test_breaker_opens_after_whole_threshold_and_recovers():
    now = [0.0]
    breaker = CircuitBreaker(
        failure_threshold=2, recovery_seconds=5, clock=lambda: now[0]
    )
    breaker.record_failure()
    assert breaker.state == CLOSED
    breaker.record_failure()
    assert breaker.state == OPEN
    now[0] = 6.0
    assert breaker.allow() is True
