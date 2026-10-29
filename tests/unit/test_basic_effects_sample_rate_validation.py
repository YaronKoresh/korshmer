import numpy as np
import pytest

from korshmer.audio_processing.basic_effects import (
    apply_filter,
    apply_noise_gate,
    apply_transient_shaper,
    slice_audio_onsets,
)

EFFECTS = [
    apply_filter,
    apply_noise_gate,
    apply_transient_shaper,
    slice_audio_onsets,
]
INVALID_RATES = [None, 0, -5, float("nan"), float("inf"), True, "44100"]


@pytest.fixture
def tone():
    return np.sin(np.arange(44100) / 10.0)


@pytest.mark.parametrize("effect", EFFECTS)
@pytest.mark.parametrize("rate", INVALID_RATES)
def test_array_mode_rejects_invalid_sample_rate(effect, rate, tone):
    with pytest.raises(ValueError, match="[Ss]ample rate"):
        effect(tone, rate)


@pytest.mark.parametrize("effect", EFFECTS)
def test_array_mode_accepts_valid_sample_rate(effect, tone):
    assert effect(tone, 44100) is not None


@pytest.mark.parametrize("effect", EFFECTS)
def test_empty_array_returned_before_sample_rate_check(effect):
    result = effect(np.zeros(0), None)
    assert len(result) == 0
