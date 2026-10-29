import numpy as np
import pytest

from korshmer.audio_processing.validation import validate_sample_rate


@pytest.mark.parametrize(
    "value",
    [
        np.int64(44100),
        np.int32(48000),
        np.float32(44100.0),
        np.float64(22050.0),
    ],
)
def test_numpy_numbers_are_accepted_like_python_numbers(value):
    assert validate_sample_rate(value) == value


@pytest.mark.parametrize(
    "bad",
    [
        np.int64(0),
        np.int64(7999),
        np.float32("nan"),
        np.float64("inf"),
        np.bool_(True),
        float("inf"),
        float("-inf"),
    ],
)
def test_numpy_numbers_out_of_range_or_not_finite_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_sample_rate(bad)
