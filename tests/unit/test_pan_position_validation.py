import numpy as np
import pytest

from korshmer.audio_processing.smart_mixing import SmartMixingAssistant
from korshmer.audio_processing.validation import (
    pan_to_unit,
    validate_pan_position,
)


class TestValidatePanPosition:
    @pytest.mark.parametrize("pan", [-100, 0, 100, 12.5, -100.0, 100.0])
    def test_accepts_in_range(self, pan):
        assert validate_pan_position(pan) == pan

    @pytest.mark.parametrize("pan", [-100.01, 101, 200, -500])
    def test_rejects_out_of_range(self, pan):
        with pytest.raises(ValueError):
            validate_pan_position(pan)

    @pytest.mark.parametrize(
        "pan", [float("nan"), float("inf"), "50", None, True, False]
    )
    def test_rejects_invalid_types(self, pan):
        with pytest.raises(ValueError):
            validate_pan_position(pan)


class TestPanToUnit:
    def test_values(self):
        assert pan_to_unit(-100) == -1.0
        assert pan_to_unit(0) == 0.0
        assert pan_to_unit(100) == 1.0


class TestAutoPanStems:
    def _assistant(self):
        return SmartMixingAssistant()

    def _stereo(self, n=100):
        rng = np.random.default_rng(1)
        return rng.standard_normal((2, n))

    def test_default_matches_previous_behavior(self):
        a = self._assistant()
        stems = [(self._stereo(), 44100) for _ in range(5)]
        out_new = a.auto_pan_stems(stems)

        old_positions = [-0.5, 0.5, -0.3, 0.3, 0.0]
        for idx, (audio, _sr) in enumerate(stems):
            pan = old_positions[idx % len(old_positions)]
            rad = (pan + 1.0) * (np.pi / 4.0)
            expected = np.zeros_like(audio)
            expected[0] = audio[0] * np.cos(rad)
            expected[1] = audio[1] * np.sin(rad)
            assert np.allclose(out_new[idx], expected)

    def test_hard_left_zeros_right_channel(self):
        a = self._assistant()
        audio = self._stereo()
        out = a.auto_pan_stems([(audio, 44100)], pan_positions=[-100])
        assert np.allclose(out[0][1], 0.0)
        assert np.allclose(out[0][0], audio[0])

    def test_out_of_range_raises(self):
        a = self._assistant()
        with pytest.raises(ValueError):
            a.auto_pan_stems([(self._stereo(), 44100)], pan_positions=[150])
