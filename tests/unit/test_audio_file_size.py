import os

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio.loader import check_audio_file_size
from korshmer.audio_file import AudioFile
from korshmer.constants import MAX_AUDIO_FILE_SIZE_BYTES
from korshmer.core.audio_engine import AudioEngine
from korshmer.exceptions import AudioError, AudioFileTooLargeError


def _write_small_wav(path):
    data = np.zeros(1000, dtype="float32")
    sf.write(str(path), data, 44100)


def test_limit_is_two_gibibytes():
    assert MAX_AUDIO_FILE_SIZE_BYTES == 2 * 1024 * 1024 * 1024


class TestCheckAudioFileSize:
    def test_small_file_passes(self, tmp_path):
        f = tmp_path / "small.wav"
        _write_small_wav(f)
        size = check_audio_file_size(str(f))
        assert size == os.path.getsize(str(f))

    def test_exactly_at_limit_passes(self, tmp_path):
        f = tmp_path / "at_limit.bin"
        f.write_bytes(b"\0" * 10)

        assert check_audio_file_size(str(f), max_bytes=10) == 10

    def test_one_byte_over_limit_raises(self, tmp_path):
        f = tmp_path / "over.bin"
        f.write_bytes(b"\0" * 11)
        with pytest.raises(AudioFileTooLargeError) as exc:
            check_audio_file_size(str(f), max_bytes=10)
        msg = str(exc.value)
        assert "11" in msg
        assert "10" in msg

    def test_error_is_audio_error_subclass(self):
        assert issubclass(AudioFileTooLargeError, AudioError)


class TestAudioFileLoad:
    def test_too_large_raises(self, tmp_path, monkeypatch):
        f = tmp_path / "big.wav"
        _write_small_wav(f)
        huge = MAX_AUDIO_FILE_SIZE_BYTES + 1
        monkeypatch.setattr(os.path, "getsize", lambda p: huge)
        with pytest.raises(AudioFileTooLargeError) as exc:
            AudioFile.load(str(f))
        assert str(huge) in str(exc.value)
        assert str(MAX_AUDIO_FILE_SIZE_BYTES) in str(exc.value)

    def test_small_file_still_loads(self, tmp_path):
        f = tmp_path / "ok.wav"
        _write_small_wav(f)
        af = AudioFile.load(str(f))
        assert af is not None
        assert af.sr == 44100


class TestAudioEngineLoad:
    def test_too_large_returns_false(self, tmp_path, monkeypatch):
        f = tmp_path / "big.wav"
        _write_small_wav(f)
        monkeypatch.setattr(
            os.path, "getsize", lambda p: MAX_AUDIO_FILE_SIZE_BYTES + 1
        )
        eng = AudioEngine()
        assert eng.load(str(f)) is False
