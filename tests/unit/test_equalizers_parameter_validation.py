import numpy as np
import pytest

from korshmer.audio_processing.equalizers import GraphicEQ, ParametricEQ

SR = 44100
NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]


@pytest.fixture
def stereo():
    rng = np.random.default_rng(0)
    return (0.3 * rng.standard_normal((2, SR // 4))).astype(np.float32)


@pytest.mark.parametrize("value", NOT_NUMBERS + [0, 9.9, 22051, 1e9])
def test_band_frequency_rejected(value):
    with pytest.raises(ValueError, match="Frequency"):
        ParametricEQ(SR).add_band(value, 3.0)


@pytest.mark.parametrize("value", NOT_NUMBERS + [-49, 49, 1e9])
def test_band_gain_rejected(value):
    with pytest.raises(ValueError, match="Gain"):
        ParametricEQ(SR).add_band(1000.0, value)


@pytest.mark.parametrize("value", ["notch", None, 3, ""])
def test_band_filter_type_rejected(value):
    with pytest.raises(ValueError, match="Unknown filter_type"):
        ParametricEQ(SR).add_band(1000.0, 3.0, 1.0, value)


@pytest.mark.parametrize("value", NOT_NUMBERS)
def test_graphic_gain_rejected(value):
    with pytest.raises(ValueError, match="Gain"):
        GraphicEQ(SR).set_gain(0, value)
    with pytest.raises(ValueError, match="Gain"):
        GraphicEQ(SR).set_gain(0, 49)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), "x", None, True, 1.5]
)
def test_graphic_band_index_rejected(value):
    with pytest.raises(ValueError, match="Band index"):
        GraphicEQ(SR).set_gain(value, 3.0)


def test_valid_bands_keep_audio_finite(stereo):
    eq = ParametricEQ(SR)
    eq.add_band(10.0, -48.0)
    eq.add_band(SR / 2, 48.0, 20.0)
    eq.add_band(1000.0, 3.0, 1.0, "lowpass")
    eq.add_band(1000.0, 3.0, 1.0, "highpass")
    assert np.all(np.isfinite(eq.process(stereo)))


def test_graphic_out_of_range_integer_index_is_ignored(stereo):
    eq = GraphicEQ(SR)
    eq.set_gain(99, 3.0)
    eq.set_gain(-99, 3.0)
    assert not np.any(eq.gains)
    eq.set_gain(2, 6.0)
    assert np.all(np.isfinite(eq.process(stereo)))


def test_graphic_bands_above_nyquist_are_skipped():
    audio = np.sin(np.arange(4000) / 5.0).astype(np.float32)
    eq = GraphicEQ(8000, bands=10)
    for index in range(10):
        eq.set_gain(index, 6.0)
    assert np.all(np.isfinite(eq.process(audio)))
