import numpy as np
import pytest

from korshmer.network.youtube_live import BitrateAdapter

NAN = float("nan")


@pytest.mark.parametrize(
    "ladder",
    [
        [0.5, 1.5],
        [NAN, 96],
        [float("inf"), 96],
        ["96"],
        [True, 96],
        [None],
        [0, 96],
        [-64, 96],
        [],
    ],
)
def test_ladder_needs_positive_whole_bitrates(ladder):
    with pytest.raises(ValueError, match="ladder"):
        BitrateAdapter(ladder)


@pytest.mark.parametrize("value", [NAN, "3", None, 2.5, True])
def test_failure_and_success_counts_must_be_whole_numbers(value):
    with pytest.raises(ValueError, match="whole number"):
        BitrateAdapter([64, 96], failures_to_drop=value)
    with pytest.raises(ValueError, match="whole number"):
        BitrateAdapter([64, 96], successes_to_raise=value)


def test_whole_valued_floats_and_numpy_numbers_are_accepted():
    adapter = BitrateAdapter([np.int64(128), 64.0, np.float32(96)])
    assert adapter.ladder == (64, 96, 128)
    assert all(type(b) is int for b in adapter.ladder)


def test_ladder_is_sorted_without_duplicates_and_starts_at_the_top():
    adapter = BitrateAdapter([128, 64, 96, 64])
    assert adapter.ladder == (64, 96, 128)
    assert adapter.bitrate == 128


def test_bitrate_drops_after_enough_failures_and_rises_after_successes():
    adapter = BitrateAdapter([64, 96], failures_to_drop=2, successes_to_raise=3)
    assert adapter.record_failure() is False
    assert adapter.record_failure() is True
    assert adapter.bitrate == 64
    assert [adapter.record_success() for _ in range(3)] == [False, False, True]
    assert adapter.bitrate == 96


def test_non_positive_counts_still_mean_one():
    adapter = BitrateAdapter([64, 96], failures_to_drop=0)
    assert adapter.failures_to_drop == 1
