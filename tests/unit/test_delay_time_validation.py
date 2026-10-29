import os
import tempfile

import numpy as np
import pytest

from korshmer.audio_processing.effects import Delay
from korshmer.audio_processing.validation import (
    MAX_DELAY_TIME_MS,
    MIN_DELAY_TIME_MS,
    validate_delay_time,
)


@pytest.fixture
def sine():
    sr = 44100
    t = np.arange(int(sr * 0.5)) / sr
    return (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


class TestValidateDelayTime:
    def test_bounds_constants(self):
        assert (MIN_DELAY_TIME_MS, MAX_DELAY_TIME_MS) == (1.0, 10000.0)

    @pytest.mark.parametrize("value", [1, 1.0, 250, 500.0, 10000, 10000.0])
    def test_valid_values_unchanged(self, value):
        assert validate_delay_time(value) == value

    def test_numpy_float_accepted(self):
        assert validate_delay_time(np.float64(500.0)) == 500.0

    def test_numpy_int_accepted(self):
        assert validate_delay_time(np.int64(500)) == 500

    @pytest.mark.parametrize("value", [0, 0.5, -5, 10001, 100000])
    def test_out_of_range(self, value):
        with pytest.raises(ValueError, match="Delay time must be between"):
            validate_delay_time(value)

    @pytest.mark.parametrize(
        "value",
        [True, False, float("nan"), float("inf"), float("-inf"), "500", None],
    )
    def test_rejects_non_finite_and_wrong_types(self, value):
        with pytest.raises(ValueError, match="Delay time must be a finite"):
            validate_delay_time(value)

    def test_error_message_reports_value(self):
        with pytest.raises(ValueError, match="got -5"):
            validate_delay_time(-5)


class TestDelayProcessValidation:
    def test_default_delay_still_works(self, sine):
        out = Delay(44100).process(sine)
        assert out.shape == sine.shape
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("value", [1, 500, 10000])
    def test_valid_delay_produces_finite_output(self, sine, value):
        out = Delay(44100).process(sine, delay_ms=value)
        assert out.shape == sine.shape
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize(
        "bad",
        [0, 0.5, -5, 10001, float("nan"), float("inf"), None, "500", True],
    )
    def test_invalid_delay_rejected(self, sine, bad):
        with pytest.raises(ValueError, match="Delay time"):
            Delay(44100).process(sine, delay_ms=bad)


class TestProcessDelayWrapper:
    def test_default_seconds_converted_to_ms_and_runs(self):
        soundfile = pytest.importorskip("soundfile")

        from korshmer.audio_processors import process_delay

        with tempfile.TemporaryDirectory() as d:
            in_path = os.path.join(d, "clip.wav")
            rng = np.random.default_rng(0)
            soundfile.write(
                in_path,
                (0.3 * rng.standard_normal(44100)).astype(np.float32),
                44100,
            )

            out_path = process_delay(in_path)
            assert os.path.exists(out_path)

    def test_invalid_delay_time_fails_before_write(self):
        soundfile = pytest.importorskip("soundfile")

        from korshmer.audio_processors import process_delay

        with tempfile.TemporaryDirectory() as d:
            in_path = os.path.join(d, "clip.wav")
            rng = np.random.default_rng(0)
            soundfile.write(
                in_path,
                (0.3 * rng.standard_normal(4410)).astype(np.float32),
                44100,
            )

            before = set(os.listdir(d))

            with pytest.raises(ValueError, match="Delay time"):
                process_delay(in_path, delay_time=20)
            assert set(os.listdir(d)) == before


def _lfilter_reference(x, sr, delay_ms, feedback, mix):
    from scipy.signal import lfilter

    d = max(1, int(sr * delay_ms / 1000.0))
    b = np.zeros(d + 1)
    b[-1] = 1.0
    a = np.zeros(d + 1)
    a[0] = 1.0
    a[-1] = -feedback
    return x * (1 - mix) + lfilter(b, a, x) * mix


class TestDelayAlgorithm:
    @pytest.mark.parametrize("delay_ms", [1.0, 7.3, 50.0])
    @pytest.mark.parametrize("length", [10, 44, 1000, 5000])
    def test_matches_direct_form_filter(self, delay_ms, length):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(length)
        out = Delay(44100).process(x, delay_ms, 0.4, 0.3)
        ref = _lfilter_reference(x, 44100, delay_ms, 0.4, 0.3)
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_stereo_channels_independent(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((2, 3000))
        out = Delay(44100).process(x, 20.0, 0.5, 0.5)
        for ch in range(2):
            ref = _lfilter_reference(x[ch], 44100, 20.0, 0.5, 0.5)
            np.testing.assert_allclose(out[ch], ref, atol=1e-12)

    def test_long_delay_on_long_audio_finishes_quickly(self):
        import time

        x = np.random.default_rng(5).standard_normal((2, 44100 * 60)) * 0.1
        Delay(44100).process(x[:, :44100], 500.0, 0.4, 0.3)
        elapsed = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            out = Delay(44100).process(x, 500.0, 0.4, 0.3)
            elapsed = min(elapsed, time.perf_counter() - start)
        assert elapsed < 3.0
        assert out.shape == x.shape
