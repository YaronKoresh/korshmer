import numpy as np
import pytest

from korshmer.core.audio_engine import AudioEngine
from korshmer.utils.playback import validate_loop_points


class DummyTime:
    def __init__(self, dac_time):
        self.outputBufferDacTime = dac_time


class TestValidateLoopPoints:
    def test_valid(self):
        assert validate_loop_points(1000, 3000, 10000) == (1000.0, 3000.0)

    def test_inverted_returns_none(self):
        assert validate_loop_points(3000, 1000, 10000) is None

    def test_equal_returns_none(self):
        assert validate_loop_points(2000, 2000, 10000) is None

    def test_negative_start_clamped(self):
        assert validate_loop_points(-500, 3000, 10000) == (0.0, 3000.0)

    def test_end_clamped_to_duration(self):
        assert validate_loop_points(1000, 999999, 10000) == (1000.0, 10000.0)

    @pytest.mark.parametrize(
        "start,end",
        [
            (float("nan"), 3000),
            (1000, float("inf")),
            ("abc", 3000),
            (None, 3000),
        ],
    )
    def test_invalid_values_return_none(self, start, end):
        assert validate_loop_points(start, end, 10000) is None


class TestAudioEngineLoop:
    def _engine(self, samples=10000, fs=1000):
        eng = AudioEngine()
        eng.data = np.zeros((samples, 1), dtype="float32")
        eng.fs = fs
        return eng

    def test_set_loop_valid(self):
        eng = self._engine()
        assert eng.set_loop(1000, 3000) is True
        assert eng.loop_start_idx < eng.loop_end_idx

    def test_set_loop_inverted(self):
        eng = self._engine()
        assert eng.set_loop(3000, 1000) is False
        assert eng.loop_start_idx is None

    def test_set_loop_equal(self):
        eng = self._engine()
        assert eng.set_loop(2000, 2000) is False

    def test_set_loop_negative_start(self):
        eng = self._engine()
        assert eng.set_loop(-500, 3000) is True
        assert eng.loop_start_idx == 0

    def test_set_loop_end_clamped(self):
        eng = self._engine()
        assert eng.set_loop(1000, 999999) is True
        assert eng.loop_end_idx == len(eng.data)

    @pytest.mark.parametrize(
        "start,end",
        [(float("nan"), 3000), (1000, float("inf")), ("abc", 3000)],
    )
    def test_set_loop_invalid(self, start, end):
        eng = self._engine()
        assert eng.set_loop(start, end) is False

    def test_set_loop_no_audio(self):
        eng = AudioEngine()
        assert eng.set_loop(1000, 3000) is False

    def test_invalid_does_not_disturb_existing_loop(self):
        eng = self._engine()
        assert eng.set_loop(1000, 3000) is True
        before = (eng.loop_start_idx, eng.loop_end_idx)
        assert eng.set_loop(3000, 1000) is False
        assert (eng.loop_start_idx, eng.loop_end_idx) == before

    def test_cb_wraps_at_loop_end(self):
        eng = self._engine()
        eng.data = np.arange(10000, dtype="float32").reshape(-1, 1)
        eng._volume = 1.0
        assert eng.set_loop(1000, 2000) is True
        eng.idx = 1950
        out = np.zeros((100, 1), dtype="float32")
        eng._cb(out, 100, DummyTime(0.0), None)

        assert eng.idx == 1050

    def test_cb_does_not_stop_with_loop(self):
        import sounddevice as sd

        eng = self._engine()
        eng.data = np.arange(10000, dtype="float32").reshape(-1, 1)
        assert eng.set_loop(0, 500) is True
        eng.idx = 480
        out = np.zeros((100, 1), dtype="float32")
        try:
            eng._cb(out, 100, DummyTime(0.0), None)
        except sd.CallbackStop:
            pytest.fail("loop playback must not raise CallbackStop")

    def test_load_clears_loop(self, monkeypatch, tmp_path):
        import soundfile as sf

        eng = self._engine()
        eng.set_loop(1000, 3000)
        wav = tmp_path / "s.wav"
        sf.write(str(wav), np.zeros(5000, dtype="float32"), 8000)
        monkeypatch.setattr(
            "korshmer.core.audio_engine.check_decoded_audio_memory",
            lambda *a, **k: 0,
        )
        monkeypatch.setattr(eng, "_check_memory_usage", lambda *a, **k: False)
        monkeypatch.setattr(
            "korshmer.core.audio_engine.sd.query_devices",
            lambda *a, **k: {"default_samplerate": 8000},
        )
        assert eng.load(str(wav)) is True
        assert eng.loop_start_idx is None
        assert eng.loop_end_idx is None
