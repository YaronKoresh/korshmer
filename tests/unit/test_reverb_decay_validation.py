import os
import tempfile

import numpy as np
import pytest

from korshmer.audio_processing.effects import Reverb
from korshmer.audio_processing.validation import (
    MAX_REVERB_DECAY,
    MAX_REVERB_DECAY_TIME_S,
    MIN_REVERB_DECAY,
    MIN_REVERB_DECAY_TIME_S,
    validate_reverb_decay,
    validate_reverb_decay_time,
)


@pytest.fixture
def sine():
    sr = 44100
    t = np.arange(int(sr * 0.5)) / sr
    return (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


class TestValidateReverbDecay:
    def test_bounds_constants(self):
        assert (MIN_REVERB_DECAY, MAX_REVERB_DECAY) == (0.0, 1.0)

    @pytest.mark.parametrize("value", [0.0, 0.5, 0.9, 1.0])
    def test_valid_values_unchanged(self, value):
        assert validate_reverb_decay(value) == value

    def test_numpy_float_accepted(self):
        assert validate_reverb_decay(np.float64(0.9)) == 0.9

    @pytest.mark.parametrize("value", [1.1, 30, -1, 2.0])
    def test_out_of_range(self, value):
        with pytest.raises(ValueError, match="Reverb decay must be between"):
            validate_reverb_decay(value)

    @pytest.mark.parametrize(
        "value", [True, float("nan"), float("inf"), "0.9", None]
    )
    def test_rejects_non_finite_and_wrong_types(self, value):
        with pytest.raises(ValueError, match="Reverb decay"):
            validate_reverb_decay(value)


class TestValidateReverbDecayTime:
    def test_bounds_constants(self):
        assert (MIN_REVERB_DECAY_TIME_S, MAX_REVERB_DECAY_TIME_S) == (
            0.1,
            30.0,
        )

    @pytest.mark.parametrize("value", [0.1, 2.0, 30.0])
    def test_valid_values_unchanged(self, value):
        assert validate_reverb_decay_time(value) == value

    def test_numpy_float_accepted(self):
        assert validate_reverb_decay_time(np.float64(2.0)) == 2.0

    @pytest.mark.parametrize("value", [0.05, 0, -1, 30.5, 100])
    def test_out_of_range(self, value):
        with pytest.raises(
            ValueError, match="Reverb decay time must be between"
        ):
            validate_reverb_decay_time(value)

    @pytest.mark.parametrize(
        "value", [True, float("nan"), float("inf"), "2", None]
    )
    def test_rejects_non_finite_and_wrong_types(self, value):
        with pytest.raises(ValueError, match="Reverb decay time"):
            validate_reverb_decay_time(value)


class TestReverbProcessDecay:
    def test_default_decay_still_works(self, sine):
        out = Reverb(44100).process(sine, decay=0.9)
        assert out.shape[0] == len(sine) + int(44100 * 2.0)
        assert np.all(np.isfinite(out))

        assert np.max(np.abs(out)) < 1.0

    @pytest.mark.parametrize(
        "bad",
        [1.1, 2.0, 30.0, -1.0, float("nan"), float("inf"), None, "0.9", True],
    )
    def test_invalid_decay_rejected(self, sine, bad):
        with pytest.raises(ValueError, match="Reverb decay"):
            Reverb(44100).process(sine, decay=bad)


class TestReverbProcessDecayTime:
    @pytest.mark.parametrize("value", [0.1, 2.0, 30.0])
    def test_valid_decay_time_produces_finite_output(self, sine, value):
        out = Reverb(44100).process(sine, decay_time_s=value)
        assert np.all(np.isfinite(out))

        assert out.shape[0] == len(sine) + int(44100 * value)

    @pytest.mark.parametrize(
        "bad", [0.05, 0, 30.5, float("nan"), float("inf"), "2", True]
    )
    def test_invalid_decay_time_rejected(self, sine, bad):
        with pytest.raises(ValueError, match="Reverb decay time"):
            Reverb(44100).process(sine, decay_time_s=bad)

    def test_longer_decay_time_gives_more_tail_energy(self, sine):
        sr = 44100
        short = Reverb(sr).process(sine, decay_time_s=1.0)
        long = Reverb(sr).process(sine, decay_time_s=4.0)

        window = slice(len(sine), len(sine) + sr)
        assert np.sum(long[window] ** 2) > np.sum(short[window] ** 2)


class TestProcessReverbWrapper:
    def test_invalid_damping_fails_before_write(self):
        soundfile = pytest.importorskip("soundfile")

        from korshmer.audio_processors import process_reverb

        with tempfile.TemporaryDirectory() as d:
            in_path = os.path.join(d, "clip.wav")
            rng = np.random.default_rng(0)
            soundfile.write(
                in_path,
                (0.3 * rng.standard_normal(4410)).astype(np.float32),
                44100,
            )

            before = set(os.listdir(d))

            with pytest.raises(ValueError, match="Reverb decay"):
                process_reverb(in_path, damping=30)

            assert set(os.listdir(d)) == before
