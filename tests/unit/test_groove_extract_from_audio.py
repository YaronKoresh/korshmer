import numpy as np
import pytest

from korshmer.midi_processing.quantization import GrooveExtractor

SR = 22050


def _clicks(seconds=6.0, interval=0.5):
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    for t in np.arange(0, seconds, interval):
        start = int(t * SR)
        audio[start : start + 200] = 1.0
    return audio


def test_audio_with_beats_returns_template():
    template = GrooveExtractor(120.0).extract_from_audio(_clicks(), SR)

    assert template.name == "Audio Extracted Groove"
    assert template.timing_offsets
    assert all(isinstance(k, float) for k in template.timing_offsets)
    assert all(isinstance(v, float) for v in template.timing_offsets.values())


def test_silence_returns_empty_template():
    audio = np.zeros(SR * 3, dtype=np.float32)

    template = GrooveExtractor(120.0).extract_from_audio(audio, SR)

    assert template.timing_offsets == {}
    assert template.velocity_curve == {}


@pytest.mark.parametrize("grid", [0, -0.25, float("nan"), float("inf")])
def test_invalid_reference_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        GrooveExtractor(120.0).extract_from_audio(_clicks(), SR, grid)


@pytest.mark.parametrize(
    "sr", [0, -5, float("nan"), float("inf"), "x", True, None]
)
def test_invalid_sample_rate_is_rejected(sr):
    with pytest.raises(ValueError):
        GrooveExtractor(120.0).extract_from_audio(_clicks(), sr)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_audio_is_rejected(bad):
    audio = _clicks()
    audio[10] = bad

    with pytest.raises(ValueError):
        GrooveExtractor(120.0).extract_from_audio(audio, SR)


def test_multichannel_audio_is_rejected():
    audio = np.zeros((2, SR), dtype=np.float32)

    with pytest.raises(ValueError):
        GrooveExtractor(120.0).extract_from_audio(audio, SR)


def test_empty_audio_returns_empty_template():
    template = GrooveExtractor(120.0).extract_from_audio(
        np.zeros(0, dtype=np.float32), SR
    )

    assert template.timing_offsets == {}
