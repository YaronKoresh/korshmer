import numpy as np
import pytest

from korshmer.audio_processing.metering import AudioMetering
from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
from korshmer.audio_processing.validation import validate_hop_length

NOT_INTEGERS = [float("nan"), float("inf"), 512.5, "x", None, True]


@pytest.fixture
def tone():
    return np.sin(np.arange(22050) / 7.0)


@pytest.mark.parametrize("value", NOT_INTEGERS)
def test_hop_length_must_be_an_integer(value):
    with pytest.raises(ValueError, match="hop_length"):
        validate_hop_length(value)


def test_hop_length_accepts_numpy_integers():
    assert validate_hop_length(np.int64(512), 2048) == 512


@pytest.mark.parametrize("cls", [SpectralAnalyzer, AudioMetering])
@pytest.mark.parametrize("value", NOT_INTEGERS + [0, -1, 10**18])
def test_zero_crossing_frame_length_rejected(cls, value, tone):
    with pytest.raises(ValueError):
        cls(44100).zero_crossing_rate(tone, frame_length=value)


@pytest.mark.parametrize("value", NOT_INTEGERS + [0, -1, 129, 10**18])
def test_mfcc_count_rejected(value, tone):
    with pytest.raises(ValueError):
        SpectralAnalyzer(44100).mfcc(tone, n_mfcc=value)


@pytest.mark.parametrize("value", [float("nan"), "x", None, True, -1, 101])
def test_dynamic_range_percentile_rejected(value, tone):
    with pytest.raises(ValueError):
        AudioMetering(44100).dynamic_range(tone, percentile=value)


def test_valid_values_still_work(tone):
    spectral = SpectralAnalyzer(44100)
    assert spectral.mfcc(tone, n_mfcc=20).shape[0] == 20
    assert len(spectral.zero_crossing_rate(tone, frame_length=1024)) > 0
    assert np.isfinite(AudioMetering(44100).dynamic_range(tone, 90))
