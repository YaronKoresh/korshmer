import pytest

from korshmer.utils.tempo import MAX_BPM, MIN_BPM, TapTempo


def test_half_second_taps_give_120():
    t = TapTempo()
    assert t.tap(0.0) is None
    assert t.tap(0.5) == pytest.approx(120.0)
    assert t.tap(1.0) == pytest.approx(120.0)


def test_single_tap_returns_none():
    assert TapTempo().tap(1.0) is None


def test_long_pause_resets():
    t = TapTempo()
    t.tap(0.0)
    t.tap(0.5)
    assert t.tap(10.0) is None
    assert t.tap(10.5) == pytest.approx(120.0)


def test_out_of_range_is_clamped():
    t = TapTempo()
    t.tap(0.0)
    assert t.tap(0.1) == MAX_BPM
    slow = TapTempo(reset_seconds=10.0)
    slow.tap(0.0)
    assert slow.tap(5.0) == MIN_BPM


def test_non_monotonic_or_invalid_restarts():
    t = TapTempo()
    t.tap(5.0)
    assert t.tap(4.0) is None
    assert t.tap(float("nan")) is None
    assert t.tap("x") is None
