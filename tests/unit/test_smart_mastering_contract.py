import warnings

import numpy as np
import pytest

from korshmer.audio_processing.smart_mastering import (
    SmartMastering,
    SmartMasteringConfig,
    benchmark,
)
from korshmer.exceptions import AudioProcessingError


def _noise(seconds=1.0, sr=44100):
    return np.random.default_rng(3).standard_normal(int(sr * seconds))


def test_default_config_smooths_the_measured_spectrum():
    audio = _noise()
    default_out, _ = SmartMastering(SmartMasteringConfig()).process(audio)
    raw_out, _ = SmartMastering(
        SmartMasteringConfig(smoothing_fraction=0.0)
    ).process(audio)

    assert not np.allclose(default_out, raw_out)


def test_explicit_zero_smoothing_is_not_replaced_by_the_default():
    master = SmartMastering(SmartMasteringConfig(smoothing_fraction=0.0))

    assert master.smoothing_fraction == 0.0
    assert SmartMastering().smoothing_fraction > 0.0


def test_benchmark_reports_every_timed_stage():
    stats = benchmark(SmartMastering(), duration_sec=0.5)

    assert {"eq", "multiband", "stereo_widen", "limiter"} <= set(stats)


@pytest.mark.parametrize("samples", [1, 2, 9, 10, 64])
def test_process_accepts_signals_shorter_than_the_filter_padding(samples):
    audio = np.full(samples, 0.25, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, sr = SmartMastering().process(audio, sr=44100)

    assert sr == 44100
    assert out.shape == (2, samples)
    assert np.max(np.abs(out)) <= 1.0
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("bad_rate", [-5, 0])
def test_bad_sample_rate_is_a_hinted_audio_processing_error(bad_rate):
    with pytest.raises(AudioProcessingError) as info:
        SmartMastering().process(np.zeros((2, 100)), sr=bad_rate)

    assert isinstance(info.value, ValueError)
    assert info.value.recovery_hint


@pytest.mark.parametrize(
    "bad",
    [
        "abc",
        np.zeros((2, 2, 100)),
        np.full((2, 4410), np.nan),
        np.array(["a", "b"]),
    ],
)
def test_rejected_audio_carries_a_recovery_hint(bad):
    with pytest.raises(AudioProcessingError) as info:
        SmartMastering().process(bad)

    assert info.value.recovery_hint
