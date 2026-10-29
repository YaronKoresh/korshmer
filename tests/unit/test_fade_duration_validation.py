import numpy as np
import pytest

from korshmer.audio_processing.smart_mastering import mix_audio
from korshmer.audio_processing.validation import validate_fade_duration


class TestValidateFadeDuration:
    def test_accepts_normal(self):
        assert validate_fade_duration(2.0, 10.0) == 2.0

    def test_accepts_equal_to_region(self):
        assert validate_fade_duration(10.0, 10.0) == 10.0

    @pytest.mark.parametrize("fade", [0.0, -1.0, -0.001])
    def test_rejects_non_positive(self, fade):
        with pytest.raises(ValueError):
            validate_fade_duration(fade, 10.0)

    def test_rejects_longer_than_region(self):
        with pytest.raises(ValueError):
            validate_fade_duration(100.0, 10.0)

    @pytest.mark.parametrize(
        "fade", [float("nan"), float("inf"), float("-inf"), "2", None, True]
    )
    def test_rejects_invalid_values(self, fade):
        with pytest.raises(ValueError):
            validate_fade_duration(fade, 10.0)


class TestMixAudioFade:
    SR = 8000

    def _audio(self, n=8000):
        rng = np.random.default_rng(0)
        return rng.standard_normal(n) * 0.1

    def test_zero_fade_raises_clear_error(self):
        y1, y2 = self._audio(), self._audio()
        with pytest.raises(ValueError, match="[Ff]ade duration"):
            mix_audio(y1, y2, sr=self.SR, fade_duration=0.0)

    @pytest.mark.parametrize("fade", [-1.0, float("nan"), float("inf")])
    def test_bad_fade_raises(self, fade):
        y1, y2 = self._audio(), self._audio()
        with pytest.raises(ValueError):
            mix_audio(y1, y2, sr=self.SR, fade_duration=fade)

    def test_fade_longer_than_audio_raises(self):

        y1, y2 = self._audio(8000), self._audio(8000)
        with pytest.raises(ValueError):
            mix_audio(y1, y2, sr=self.SR, fade_duration=100.0)

    def test_fade_equal_to_audio_length_ok(self):
        y1, y2 = self._audio(8000), self._audio(8000)
        out = mix_audio(y1, y2, sr=self.SR, fade_duration=1.0)
        assert out is not None

    def test_normal_fade_runs(self):
        y1, y2 = self._audio(8000), self._audio(8000)
        out = mix_audio(y1, y2, sr=self.SR, fade_duration=0.5)
        assert out is not None

    def test_subsample_fade_no_broadcast_error(self):
        y1, y2 = self._audio(8000), self._audio(8000)

        out = mix_audio(y1, y2, sr=self.SR, fade_duration=0.0001)
        assert out is not None
