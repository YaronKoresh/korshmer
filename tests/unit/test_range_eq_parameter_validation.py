import numpy as np
import pytest

from korshmer.audio_processing.range_eq import RangeEQ

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]

PARAMS = [
    "freq_start",
    "freq_end",
    "total_gain_db",
    "peak_pos_pct",
    "curve_left",
    "curve_right",
]


@pytest.fixture
def tone():
    return np.sin(np.arange(4410) / 7.0)


@pytest.mark.parametrize("param", PARAMS)
@pytest.mark.parametrize("value", NOT_NUMBERS + [1e18, -1e18])
def test_parameter_rejected(param, value, tone):
    with pytest.raises(ValueError):
        RangeEQ().process(tone, **{param: value})


@pytest.mark.parametrize("param", ["freq_start", "freq_end"])
@pytest.mark.parametrize("value", [0, -5, 0.5])
def test_non_positive_frequency_rejected(param, value, tone):
    with pytest.raises(ValueError, match="frequency"):
        RangeEQ().process(tone, **{param: value})


def test_empty_audio_is_passed_through():
    assert len(RangeEQ().process(np.zeros(0))) == 0


def test_valid_extremes_accepted(tone):
    out = RangeEQ().process(
        tone,
        freq_start=2000,
        freq_end=30000,
        total_gain_db=12,
        peak_pos_pct=99,
        curve_left=-10,
        curve_right=10,
    )
    assert np.all(np.isfinite(out))
