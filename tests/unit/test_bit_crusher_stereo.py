from __future__ import annotations

import numpy as np

from korshmer.audio_processing.distortion_noise import Distortion


def test_bit_crusher_stereo_sample_rate_reduction():
    dist = Distortion()
    stereo = np.random.default_rng(42).uniform(-1, 1, (2, 1000))
    result = dist.bit_crusher(stereo, bits=8, sample_rate_reduction=4.0)
    assert result.shape == stereo.shape


def test_bit_crusher_mono_sample_rate_reduction():
    dist = Distortion()
    mono = np.random.default_rng(42).uniform(-1, 1, 1000)
    result = dist.bit_crusher(mono, bits=8, sample_rate_reduction=4.0)
    assert result.shape == mono.shape


def test_bit_crusher_stereo_holds_per_channel():
    dist = Distortion()
    stereo = np.zeros((2, 12))
    stereo[0, :] = np.arange(12, dtype=float)
    stereo[1, :] = np.arange(12, dtype=float) * -1
    result = dist.bit_crusher(stereo, bits=16, sample_rate_reduction=3.0)
    assert result.shape == (2, 12)
    assert result[0, 0] == result[0, 1] == result[0, 2]
    assert result[1, 0] == result[1, 1] == result[1, 2]
