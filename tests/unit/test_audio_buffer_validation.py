import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio import loader
from korshmer.audio.loader import (
    check_decoded_audio_memory,
    validate_buffer_size,
)
from korshmer.constants import (
    DEFAULT_PLAYBACK_BUFFER_SIZE,
    MAX_PLAYBACK_BUFFER_SIZE,
    MIN_PLAYBACK_BUFFER_SIZE,
)
from korshmer.core.audio_engine import AudioEngine, PlaybackState
from korshmer.exceptions import AudioFileTooLargeError, AudioFormatError


class RecordingStream:
    calls = []

    def __init__(self, *args, **kwargs):
        RecordingStream.calls.append(kwargs)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def recording_stream(monkeypatch):
    RecordingStream.calls = []
    monkeypatch.setattr(
        "korshmer.core.audio_engine.sd.OutputStream", RecordingStream
    )
    return RecordingStream


@pytest.fixture
def small_wav(tmp_path):
    path = tmp_path / "small.wav"
    sf.write(str(path), np.zeros(1000, dtype="float32"), 44100)
    return str(path)


class TestValidateBufferSize:
    @pytest.mark.parametrize(
        "value",
        [MIN_PLAYBACK_BUFFER_SIZE, 1024, 2048, MAX_PLAYBACK_BUFFER_SIZE],
    )
    def test_valid_values(self, value):
        assert validate_buffer_size(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            0,
            -1,
            511,
            MAX_PLAYBACK_BUFFER_SIZE + 1,
            10**9,
            True,
            False,
            "abc",
            2048.0,
            float("nan"),
            None,
        ],
    )
    def test_invalid_values(self, value):
        with pytest.raises(ValueError):
            validate_buffer_size(value)


class TestPlayBufferFallback:
    def test_default_buffer_size(self, recording_stream):
        eng = AudioEngine()
        assert eng._current_buffer_size == DEFAULT_PLAYBACK_BUFFER_SIZE == 2048
        eng.data = np.zeros((10, 1), dtype="float32")
        eng.fs = 10
        eng.play()
        assert eng.playbackState() == PlaybackState.PLAYING
        assert recording_stream.calls[-1]["blocksize"] == 2048

    @pytest.mark.parametrize("bad", [0, -1, 10**9, True, "abc"])
    def test_invalid_buffer_falls_back(self, recording_stream, caplog, bad):
        eng = AudioEngine()
        eng.data = np.zeros((10, 1), dtype="float32")
        eng.fs = 10
        eng._current_buffer_size = bad
        with caplog.at_level(logging.ERROR):
            eng.play()
        assert eng.playbackState() == PlaybackState.PLAYING
        assert recording_stream.calls[-1]["blocksize"] == 2048
        assert eng._current_buffer_size == 2048
        assert "Invalid playback buffer size" in caplog.text

    def test_reduce_buffer_size_floor(self):
        eng = AudioEngine()
        sizes = []
        for _ in range(4):
            eng._reduce_buffer_size()
            sizes.append(eng._current_buffer_size)
        assert sizes == [1024, 512, 512, 512]


def _fake_info(frames, channels):
    return lambda path: SimpleNamespace(
        frames=frames, channels=channels, samplerate=44100
    )


class TestCheckDecodedAudioMemory:
    def test_small_file_passes(self, small_wav):
        assert check_decoded_audio_memory(small_wav) == 1000 * 1 * 4

    def test_huge_frame_count_refused(self, monkeypatch, small_wav):
        monkeypatch.setattr(loader.sf, "info", _fake_info(10**12, 2))
        with pytest.raises(AudioFileTooLargeError):
            check_decoded_audio_memory(small_wav)

    @pytest.mark.parametrize("channels", [0, 9, 64])
    def test_channel_count_refused(self, monkeypatch, small_wav, channels):
        monkeypatch.setattr(loader.sf, "info", _fake_info(1000, channels))
        with pytest.raises(AudioFileTooLargeError):
            check_decoded_audio_memory(small_wav)

    def test_zero_frames_refused(self, monkeypatch, small_wav):
        monkeypatch.setattr(loader.sf, "info", _fake_info(0, 1))
        with pytest.raises(AudioFormatError):
            check_decoded_audio_memory(small_wav)

    def test_available_memory_refused(self, monkeypatch, small_wav):
        monkeypatch.setattr(
            loader.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(available=100, percent=10.0),
        )
        with pytest.raises(AudioFileTooLargeError):
            check_decoded_audio_memory(small_wav)


class TestEngineLoad:
    def test_small_wav_loads(self, monkeypatch, small_wav):
        monkeypatch.setattr(
            "korshmer.core.audio_engine.sd.query_devices",
            lambda kind=None: {"default_samplerate": 44100.0},
        )
        eng = AudioEngine()
        assert eng.load(small_wav) is True
        assert eng.data.shape == (1000, 1)

    @pytest.mark.parametrize("frames,channels", [(10**12, 2), (1000, 64)])
    def test_bad_header_refused_before_read(
        self, monkeypatch, caplog, small_wav, frames, channels
    ):
        monkeypatch.setattr(loader.sf, "info", _fake_info(frames, channels))
        read_calls = []
        monkeypatch.setattr(
            "korshmer.core.audio_engine.sf.read",
            lambda *a, **k: read_calls.append(a),
        )
        eng = AudioEngine()
        with caplog.at_level(logging.ERROR):
            assert eng.load(small_wav) is False
        assert read_calls == []
        assert "Refusing to load audio" in caplog.text

    def test_low_available_memory_refused(self, monkeypatch, small_wav):
        monkeypatch.setattr(
            loader.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(available=10, percent=10.0),
        )
        eng = AudioEngine()
        assert eng.load(small_wav) is False
        assert eng.data is None


class TestBufferAutoAdjust:
    def _engine(self):
        eng = AudioEngine()
        eng.data = np.zeros((1000, 1), dtype="float32")
        eng.fs = 10
        return eng

    def test_underruns_double_size_on_next_start(self, recording_stream):
        eng = self._engine()
        out = np.zeros((4, 1), dtype="float32")
        for _ in range(5):
            eng._cb(out, 4, None, "output underflow")
        eng._adjust_buffer_size(cpu_percent=50.0, now=1.0)
        assert eng._current_buffer_size == 4096
        eng.play()
        assert recording_stream.calls[-1]["blocksize"] == 4096

    def test_high_cpu_doubles_up_to_max(self):
        eng = self._engine()
        for i in range(6):
            eng._adjust_buffer_size(cpu_percent=95.0, now=float(i))
        assert eng._current_buffer_size == MAX_PLAYBACK_BUFFER_SIZE

    def test_sustained_low_load_halves_to_floor(self):
        eng = self._engine()
        eng._adjust_buffer_size(cpu_percent=5.0, now=0.0)
        assert eng._current_buffer_size == 2048
        t = 0.0
        for _ in range(8):
            t += 31.0
            eng._adjust_buffer_size(cpu_percent=5.0, now=t)
        assert eng._current_buffer_size == MIN_PLAYBACK_BUFFER_SIZE

    def test_medium_load_resets_low_load_timer(self):
        eng = self._engine()
        eng._adjust_buffer_size(cpu_percent=5.0, now=0.0)
        eng._adjust_buffer_size(cpu_percent=60.0, now=20.0)
        eng._adjust_buffer_size(cpu_percent=5.0, now=40.0)
        assert eng._current_buffer_size == 2048

    def test_callback_does_not_resize_running_stream(self):
        eng = self._engine()
        out = np.zeros((4, 1), dtype="float32")
        eng._cb(out, 4, None, "output underflow")
        assert eng._current_buffer_size == 2048
