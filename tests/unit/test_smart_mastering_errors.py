import numpy as np
import pytest

from korshmer.audio_processing.smart_mastering import SmartMastering
from korshmer.exceptions import AudioProcessingError


@pytest.mark.parametrize(
    "bad",
    [
        "abc",
        np.int64(5),
        np.zeros((2, 2, 100)),
        np.full((2, 4410), np.nan),
        np.full((2, 4410), np.inf),
        np.array(["a", "b"]),
    ],
)
def test_invalid_input_raises_audio_processing_error(bad):
    with pytest.raises(AudioProcessingError) as info:
        SmartMastering().process(bad)
    assert info.value.recovery_hint


def test_none_and_bad_sample_rate_stay_value_errors():
    with pytest.raises(ValueError):
        SmartMastering().process(None)
    with pytest.raises(ValueError):
        SmartMastering().process(np.zeros((2, 100)), sr=-5)
    with pytest.raises(AudioProcessingError):
        SmartMastering().process(np.zeros((2, 100)), sr=-5)


def test_unknown_phase_type_raises_audio_processing_error():
    with pytest.raises(AudioProcessingError):
        SmartMastering()._phase(np.zeros((2, 4410)), "bogus")


def test_empty_array_still_passes_through():
    out, _ = SmartMastering().process(np.zeros((2, 0)))
    assert out.shape == (2, 0)
