from unittest import mock

import numpy as np
import pytest

from korshmer.audio_processing.dynamics import (
    Compressor,
    MultiBandCompressor,
)
from korshmer.audio_processing.validation import (
    MAX_COMPRESSION_RATIO,
    MIN_COMPRESSION_RATIO,
    validate_compression_ratio,
)


@pytest.fixture
def sine():
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    return (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


class TestValidateCompressionRatio:
    @pytest.mark.parametrize("value", [1.0, 1, 2, 4.0, 10, 20, 20.0])
    def test_valid_values_unchanged(self, value):
        assert validate_compression_ratio(value) == value

    def test_numpy_float_accepted(self):
        assert validate_compression_ratio(np.float64(4.0)) == 4.0

    def test_numpy_integer_accepted(self):
        assert validate_compression_ratio(np.int64(4)) == 4

    def test_bounds_constants(self):
        assert (MIN_COMPRESSION_RATIO, MAX_COMPRESSION_RATIO) == (1.0, 20.0)

    @pytest.mark.parametrize("value", [0, -2, 0.5, 0.99, 20.01, 100])
    def test_out_of_range(self, value):
        with pytest.raises(
            ValueError, match="Compression ratio must be between"
        ):
            validate_compression_ratio(value)

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), "4", None, True, False],
    )
    def test_wrong_types_and_non_finite(self, value):
        with pytest.raises(
            ValueError, match="Compression ratio must be a finite number"
        ):
            validate_compression_ratio(value)


class TestCompressor:
    def test_default_ratio_works(self, sine):
        out = Compressor().process(sine)
        assert np.all(np.isfinite(out))
        rms_in = np.sqrt(np.mean(sine**2))
        rms_out = np.sqrt(np.mean(out**2))
        assert rms_out < rms_in

    @pytest.mark.parametrize("value", [1.0, 20.0])
    def test_edges_work(self, sine, value):
        out = Compressor().process(sine, ratio=value)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize(
        "value",
        [
            0,
            -2,
            0.5,
            0.99,
            20.01,
            100,
            float("nan"),
            float("inf"),
            "4",
            None,
            True,
        ],
    )
    def test_invalid_ratio_raises(self, sine, value):
        with pytest.raises(ValueError, match="Compression ratio"):
            Compressor().process(sine, ratio=value)


class TestMultiBandCompressor:
    def test_default_ratio_works(self, sine):
        out = MultiBandCompressor().process(sine)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("value", [1.0, 20.0])
    def test_edges_work(self, sine, value):
        out = MultiBandCompressor().process(sine, ratio=value)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize(
        "value",
        [
            0,
            -2,
            0.5,
            0.99,
            20.01,
            100,
            float("nan"),
            float("inf"),
            "4",
            None,
            True,
        ],
    )
    def test_invalid_ratio_raises(self, sine, value):
        with pytest.raises(ValueError, match="Compression ratio"):
            MultiBandCompressor().process(sine, ratio=value)

    def test_ratio_checked_before_split_bands(self, sine):
        with mock.patch.object(MultiBandCompressor, "_split_bands") as split:
            with pytest.raises(ValueError, match="Compression ratio"):
                MultiBandCompressor().process(sine, ratio=0)
            split.assert_not_called()
