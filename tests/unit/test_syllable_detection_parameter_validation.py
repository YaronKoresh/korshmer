import numpy as np
import pytest

from korshmer.audio_processing.vocal_processing import VocalSyllableDetector

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]

PARAMS = [
    "min_syllable_duration",
    "max_syllable_duration",
    "min_gap_duration",
    "noise_threshold_db",
]


@pytest.fixture
def voice():
    sr = 22050
    t = np.arange(sr) / sr
    return 0.5 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)


@pytest.mark.parametrize("param", PARAMS)
@pytest.mark.parametrize("value", NOT_NUMBERS + [1e18, -1e18])
def test_numeric_parameter_rejected(param, value, voice):
    with pytest.raises(ValueError):
        VocalSyllableDetector(22050).detect_syllables(voice, **{param: value})


def test_max_duration_must_be_positive(voice):
    with pytest.raises(ValueError):
        VocalSyllableDetector(22050).detect_syllables(
            voice, max_syllable_duration=0
        )


def test_minimum_cannot_exceed_maximum(voice):
    with pytest.raises(ValueError, match="cannot exceed"):
        VocalSyllableDetector(22050).detect_syllables(
            voice, min_syllable_duration=2.0, max_syllable_duration=1.0
        )


def test_defaults_and_valid_values_still_work(voice):
    detector = VocalSyllableDetector(22050)
    assert isinstance(detector.detect_syllables(voice), list)
    assert isinstance(
        detector.detect_syllables(
            voice,
            min_syllable_duration=0.1,
            max_syllable_duration=3.0,
            min_gap_duration=0.05,
            noise_threshold_db=-50,
        ),
        list,
    )
    assert detector.detect_syllables(np.zeros(100)) == []
