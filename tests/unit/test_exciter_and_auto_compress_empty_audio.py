import numpy as np

from korshmer.audio_processing.smart_mastering import apply_exciter
from korshmer.audio_processing.smart_mixing import SmartMixingAssistant

SAMPLE_RATE = 22050


def test_apply_exciter_returns_empty_audio_for_empty_input():
    result = apply_exciter(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    assert result.size == 0


def test_auto_compress_returns_empty_audio_for_empty_input():
    result = SmartMixingAssistant().auto_compress(
        np.zeros(0, dtype=np.float32), SAMPLE_RATE
    )
    assert result.size == 0
