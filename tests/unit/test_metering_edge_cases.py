import numpy as np
import pytest

from korshmer.audio_processing.metering import Analyzer, AudioMetering


def test_stereo_correlation_with_one_silent_channel_is_zero():
    audio = np.vstack(
        [np.zeros(1000), np.random.default_rng(0).standard_normal(1000)]
    )
    assert AudioMetering().stereo_correlation(audio) == 0.0


def test_stereo_correlation_with_two_silent_channels_is_one():
    assert AudioMetering().stereo_correlation(np.zeros((2, 1000))) == 1.0


def test_stereo_correlation_with_different_constant_channels_is_zero():
    audio = np.vstack([np.full(1000, 0.2), np.full(1000, 0.5)])
    assert AudioMetering().stereo_correlation(audio) == 0.0


def test_stereo_correlation_of_normal_signal_is_unchanged():
    noise = np.random.default_rng(1).standard_normal(1000)
    assert AudioMetering().stereo_correlation(
        np.vstack([noise, noise])
    ) == pytest.approx(1.0)
    assert AudioMetering().stereo_correlation(
        np.vstack([noise, -noise])
    ) == pytest.approx(-1.0)


@pytest.mark.parametrize("percentile", [100, 99.99])
def test_dynamic_range_accepts_top_percentile(percentile):
    audio = np.random.default_rng(2).standard_normal(1000)
    assert np.isfinite(AudioMetering().dynamic_range(audio, percentile))


@pytest.mark.parametrize("percentile", [-1, 101, 150, True])
def test_dynamic_range_rejects_percentile_outside_0_to_100(percentile):
    with pytest.raises(ValueError):
        AudioMetering().dynamic_range(np.ones(100), percentile)


def test_dynamic_range_of_stereo_is_a_single_number():
    audio = np.random.default_rng(3).standard_normal((2, 1000))
    assert np.ndim(AudioMetering().dynamic_range(audio)) == 0


def test_analyzer_reports_scalar_dynamic_range_and_finite_correlation_for_stereo():
    audio = np.vstack(
        [np.zeros(2000), np.random.default_rng(4).standard_normal(2000)]
    )
    analysis = Analyzer().analyze_audio(audio)
    assert np.ndim(analysis["dynamic_range_db"]) == 0
    assert analysis["stereo_correlation"] == 0.0
