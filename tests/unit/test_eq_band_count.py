import numpy as np
import pytest

from korshmer.audio_processing.equalizers import GraphicEQ, ParametricEQ
from korshmer.audio_processing.validation import (
    MAX_EQ_BANDS,
    validate_eq_band_count,
)


class TestParametricEQBandLimit:
    def test_limit_is_31(self):
        assert MAX_EQ_BANDS == 31

    def test_31_bands_are_accepted(self):
        eq = ParametricEQ()
        for i in range(MAX_EQ_BANDS):
            eq.add_band(100 + 50 * i, 1.0)
        assert len(eq.bands) == MAX_EQ_BANDS

    def test_32nd_band_is_rejected_and_not_added(self):
        eq = ParametricEQ()
        for i in range(MAX_EQ_BANDS):
            eq.add_band(100 + 50 * i, 1.0)
        with pytest.raises(ValueError, match="maximum of 31"):
            eq.add_band(5000, 1.0)
        assert len(eq.bands) == MAX_EQ_BANDS

    def test_bands_can_be_added_again_after_clear(self):
        eq = ParametricEQ()
        for i in range(MAX_EQ_BANDS):
            eq.add_band(100 + 50 * i, 1.0)
        eq.clear_bands()
        eq.add_band(1000, 6.0)
        assert len(eq.bands) == 1

    def test_full_eq_still_processes_audio(self):
        eq = ParametricEQ()
        for i in range(MAX_EQ_BANDS):
            eq.add_band(100 + 50 * i, 1.0)
        out = eq.process(
            np.random.default_rng(0).standard_normal(4096).astype(np.float32)
        )
        assert np.all(np.isfinite(out))


class TestGraphicEQBandCount:
    def test_default_is_ten_bands(self):
        eq = GraphicEQ()
        assert eq.n_bands == 10
        assert len(eq.gains) == 10

    @pytest.mark.parametrize("bands", [1, 10, 31])
    def test_valid_counts_are_accepted(self, bands):
        eq = GraphicEQ(bands=bands)
        assert eq.n_bands == bands
        assert len(eq.gains) == bands
        assert len(eq.frequencies) == bands

    @pytest.mark.parametrize("bands", [0, -1, 32, 1000])
    def test_out_of_range_counts_are_rejected(self, bands):
        with pytest.raises(ValueError, match="between 1 and 31"):
            GraphicEQ(bands=bands)

    @pytest.mark.parametrize("bands", [2.5, "10", True, None])
    def test_non_integer_counts_are_rejected(self, bands):
        with pytest.raises(ValueError, match="must be an integer"):
            GraphicEQ(bands=bands)

    def test_graphic_eq_with_31_bands_processes_audio(self):
        eq = GraphicEQ(bands=31)
        eq.set_gain(15, 6.0)
        out = eq.process(
            np.random.default_rng(1).standard_normal(4096).astype(np.float32)
        )
        assert np.all(np.isfinite(out))


class TestValidateEqBandCount:
    def test_returns_the_value_when_valid(self):
        assert validate_eq_band_count(7) == 7

    def test_accepts_numpy_integers(self):
        assert validate_eq_band_count(np.int64(12)) == 12
