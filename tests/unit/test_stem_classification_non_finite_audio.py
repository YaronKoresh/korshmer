import numpy as np
import pytest

from korshmer.audio_processing.stem_classification import (
    StemClassifier,
    VocalProcessingAI,
)

SAMPLE_RATE = 22050


def _audio_with(value):
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    audio[100] = value
    return audio


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_classify_stem_rejects_non_finite_audio(value):
    with pytest.raises(ValueError, match="NaN or infinite"):
        StemClassifier().classify_stem(_audio_with(value), SAMPLE_RATE)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_detect_emotion_rejects_non_finite_audio(value):
    with pytest.raises(ValueError, match="NaN or infinite"):
        VocalProcessingAI().detect_emotion(_audio_with(value), SAMPLE_RATE)
