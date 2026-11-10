import numpy as np
import pytest

from korshmer.audio_processing import smart_mastering as sm


@pytest.mark.parametrize("channels", [1, 2])
def test_freq_cut_without_filters_removes_each_channel_mean(channels):
    rng = np.random.default_rng(7)
    shape = (channels, 4096) if channels > 1 else (4096,)
    original = rng.standard_normal(shape) + 0.3
    expected = np.empty_like(original)
    if channels > 1:
        for i in range(channels):
            expected[i] = original[i] - np.mean(original[i])
    else:
        expected[:] = original - np.mean(original)

    result = sm.freq_cut(original.copy(), 44100, None, None)

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_decoupled_envelope_matches_python_reference():
    rng = np.random.default_rng(11)
    env_db = rng.uniform(-80.0, 0.0, size=5000)
    attack, release = 0.9, 0.999

    expected = sm._decoupled_envelope.py_func(env_db, attack, release)
    result = sm._decoupled_envelope(env_db, attack, release)

    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


def test_limiter_release_matches_python_reference():
    rng = np.random.default_rng(13)
    gain_db = -rng.uniform(0.0, 12.0, size=5000)
    rc_fast, rc_slow, w_slow = 0.95, 0.999, 0.3

    expected = sm._limiter_release.py_func(gain_db, rc_fast, rc_slow, w_slow)
    result = sm._limiter_release(gain_db, rc_fast, rc_slow, w_slow)

    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)
