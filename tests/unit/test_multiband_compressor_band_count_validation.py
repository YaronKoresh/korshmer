import numpy as np
import pytest

from korshmer.audio_processing.dynamics import MultiBandCompressor


@pytest.fixture
def tone():
    return np.sin(np.arange(22050) / 7.0)


@pytest.mark.parametrize(
    "n_bands",
    [float("nan"), float("inf"), 1e18, 2.5, "4", None, True, 0, -1, 6, 100],
)
def test_rejects_invalid_band_count(n_bands):
    with pytest.raises(ValueError, match="Band count"):
        MultiBandCompressor(44100, n_bands=n_bands)


@pytest.mark.parametrize("n_bands", [1, 2, 3, 4, 5, np.int64(3)])
def test_accepts_valid_band_count(n_bands, tone):
    comp = MultiBandCompressor(44100, n_bands=n_bands)
    assert len(comp.crossover_freqs) == n_bands - 1
    assert len(comp.compressors) == n_bands
    out = comp.process(tone)
    assert out.shape == tone.shape
    assert np.all(np.isfinite(out))
