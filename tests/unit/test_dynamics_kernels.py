import time

import numpy as np
import pytest

from korshmer.audio_processing import kernels
from korshmer.audio_processing.dynamics import Compressor, Limiter


def _python_compress(audio, **kw):
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        kernels.compress_channel_python(audio[ch], out[ch], **kw)
    return out


@pytest.fixture
def stereo():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, (2, 5 * 44100))


def test_compressor_matches_python_loop(stereo):
    sr = 44100
    kw = {
        "threshold_linear": 10 ** (-20 / 20.0),
        "ratio": 4.0,
        "attack_coef": np.exp(-1.0 / int(sr * 10 / 1000.0)),
        "release_coef": np.exp(-1.0 / int(sr * 100 / 1000.0)),
        "makeup_linear": 1.0,
    }
    expected = _python_compress(stereo, **kw)
    result = Compressor(sr).process(stereo)
    assert result.shape == stereo.shape
    assert np.allclose(result, expected)


def test_limiter_matches_python_loop(stereo):
    sr = 44100
    threshold = 10 ** (-0.3 / 20.0)
    coef = np.exp(-1.0 / int(sr * 50 / 1000.0))
    expected = np.zeros_like(stereo)
    for ch in range(2):
        kernels.limit_channel_python(stereo[ch], expected[ch], threshold, coef)
    assert np.allclose(Limiter(sr).process(stereo), expected)


def test_mono_input_stays_mono():
    mono = np.linspace(-1, 1, 1000)
    assert Compressor().process(mono).shape == mono.shape
    assert Limiter().process(mono).shape == mono.shape


def test_compressor_still_validates_ratio(stereo):
    with pytest.raises(ValueError):
        Compressor().process(stereo, ratio=0.5)


@pytest.mark.skipif(not kernels.HAS_NUMBA, reason="numba not installed")
def test_kernel_is_faster_than_python_loop():
    audio = np.random.default_rng(1).uniform(-1, 1, 44100)
    out = np.zeros_like(audio)
    args = (0.1, 4.0, 0.9, 0.99, 1.0)
    kernels.compress_channel(audio, out, *args)
    start = time.perf_counter()
    kernels.compress_channel(audio, out, *args)
    compiled = time.perf_counter() - start
    start = time.perf_counter()
    kernels.compress_channel_python(audio, out, *args)
    python = time.perf_counter() - start
    assert compiled < python
