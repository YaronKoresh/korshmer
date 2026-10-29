import time

import numpy as np
import pytest

from korshmer.audio_processing.pitch_estimator import (
    autocorrelation_pitch,
    piptrack_pitch,
)

SR = 44100


def _sine(freq, seconds):
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * freq * t)


def _reference_pitch(segment, sr, fmin, fmax, threshold):
    full = np.correlate(segment, segment, mode="full")
    autocorr = full[len(full) // 2 :]
    if autocorr[0] == 0:
        return 0, 0.0
    autocorr = autocorr / autocorr[0]
    min_period = max(1, int(sr / fmax))
    max_period = int(sr / fmin)
    if max_period >= len(autocorr):
        return 0, 0.0
    window = autocorr[min_period:max_period]
    if len(window) == 0:
        return 0, 0.0
    peak = int(np.argmax(window)) + min_period
    if autocorr[peak] > threshold:
        return sr / peak, autocorr[peak]
    return 0, 0.0


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0, 880.0])
def test_matches_direct_correlation(freq):
    segment = _sine(freq, 0.1) + 0.05 * np.random.default_rng(
        1
    ).standard_normal(int(SR * 0.1))
    expected = _reference_pitch(segment, SR, 50.0, 1000.0, 0.3)
    actual = autocorrelation_pitch(segment, SR, 50.0, 1000.0, 0.3)
    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_does_not_use_full_direct_correlation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("direct correlation is quadratic")

    monkeypatch.setattr(np, "correlate", forbidden)
    pitch, _ = autocorrelation_pitch(_sine(440.0, 0.5), SR, 50.0, 1000.0)
    assert pitch == pytest.approx(441.0)


def test_long_segment_is_fast():
    segment = _sine(220.0, 10.0)
    start = time.perf_counter()
    pitch, _ = autocorrelation_pitch(segment, SR, 50.0, 1000.0)
    assert time.perf_counter() - start < 2.0
    assert pitch == pytest.approx(220.0, rel=0.01)


def test_silence_and_short_segments_return_zero():
    assert autocorrelation_pitch(np.zeros(5000), SR, 50.0, 1000.0) == (0, 0.0)
    assert autocorrelation_pitch(np.ones(100), SR, 50.0, 1000.0) == (0, 0.0)
    assert autocorrelation_pitch(np.array([]), SR, 50.0, 1000.0) == (0, 0.0)


def test_integer_input_is_accepted():
    segment = (_sine(440.0, 0.2) * 1000).astype(np.int16)
    pitch, _ = autocorrelation_pitch(segment, SR, 50.0, 1000.0)
    assert pitch == pytest.approx(441.0)


BAD_RANGES = [
    (0, 1000.0),
    (-5.0, 1000.0),
    (50.0, 0),
    (50.0, -1.0),
    (50.0, float("inf")),
    (float("nan"), 1000.0),
    (50.0, float("nan")),
    (1000.0, 50.0),
    (500.0, 500.0),
    ("x", 1000.0),
]


@pytest.mark.parametrize("fmin,fmax", BAD_RANGES)
def test_invalid_frequency_range_is_rejected(fmin, fmax):
    segment = _sine(440.0, 0.1)
    with pytest.raises(ValueError):
        autocorrelation_pitch(segment, SR, fmin, fmax)
    with pytest.raises(ValueError):
        piptrack_pitch(segment, SR, fmin, fmax)


@pytest.mark.parametrize("sr", [0, -1, float("nan"), "x", None])
def test_invalid_sample_rate_is_rejected(sr):
    segment = _sine(440.0, 0.1)
    with pytest.raises(ValueError):
        autocorrelation_pitch(segment, sr, 50.0, 1000.0)
    with pytest.raises(ValueError):
        piptrack_pitch(segment, sr, 50.0, 1000.0)
