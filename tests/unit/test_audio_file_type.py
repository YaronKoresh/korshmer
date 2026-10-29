import numpy as np
import pytest
import soundfile as sf

from korshmer.audio.loader import check_audio_file_type
from korshmer.audio_file import AudioFile
from korshmer.core.audio_engine import AudioEngine
from korshmer.exceptions import AudioError, AudioFormatError
from korshmer.project_management.import_manager import ImportManager


def _write_wav(path):
    sf.write(str(path), np.zeros(1000, dtype="float32"), 44100)


class TestCheckAudioFileType:
    def test_real_wav_accepted(self, tmp_path):
        f = tmp_path / "real.wav"
        _write_wav(f)
        assert check_audio_file_type(str(f)) is True

    def test_mp3_with_and_without_id3(self, tmp_path):
        id3 = tmp_path / "a.mp3"
        id3.write_bytes(b"ID3\x03\x00" + b"\0" * 20)
        assert check_audio_file_type(str(id3)) is True

        no_id3 = tmp_path / "b.mp3"
        no_id3.write_bytes(bytes([0xFF, 0xFB, 0x90, 0x00]) + b"\0" * 20)
        assert check_audio_file_type(str(no_id3)) is True

    def test_flac_ogg_aiff_rf64(self, tmp_path):
        cases = {
            "x.flac": b"fLaC" + b"\0" * 20,
            "x.ogg": b"OggS" + b"\0" * 20,
            "x.aiff": b"FORM\x00\x00\x00\x00AIFF" + b"\0" * 8,
            "x.rf64": b"RF64\x00\x00\x00\x00WAVE" + b"\0" * 8,
        }
        for name, content in cases.items():
            f = tmp_path / name
            f.write_bytes(content)
            assert check_audio_file_type(str(f)) is True

    def test_text_file_renamed_wav_rejected(self, tmp_path):
        f = tmp_path / "fake.wav"
        f.write_text("this is plain text, not audio at all")
        with pytest.raises(AudioFormatError):
            check_audio_file_type(str(f))

    def test_empty_file_rejected(self, tmp_path):
        f = tmp_path / "empty.wav"
        f.write_bytes(b"")
        with pytest.raises(AudioFormatError):
            check_audio_file_type(str(f))

    def test_error_is_audio_error_subclass(self):
        assert issubclass(AudioFormatError, AudioError)


class TestAudioFileLoadType:
    def test_fake_wav_raises(self, tmp_path):
        f = tmp_path / "fake.wav"
        f.write_text("not audio")
        with pytest.raises(AudioFormatError):
            AudioFile.load(str(f))

    def test_real_wav_loads(self, tmp_path):
        f = tmp_path / "ok.wav"
        _write_wav(f)
        assert AudioFile.load(str(f)) is not None


class TestAudioEngineLoadType:
    def test_fake_wav_returns_false(self, tmp_path):
        f = tmp_path / "fake.wav"
        f.write_text("not audio")
        eng = AudioEngine()
        assert eng.load(str(f)) is False


class TestImportManagerType:
    def test_only_valid_audio_imported(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        good1 = tmp_path / "good1.wav"
        _write_wav(good1)
        good2 = tmp_path / "good2.wav"
        _write_wav(good2)
        bad = tmp_path / "fake.mp3"
        bad.write_text("not audio")

        mgr = ImportManager(base)
        imported = mgr.import_audio_files(
            [good1, bad, good2], auto_organize=False
        )
        assert len(imported) == 2
        names = {p.name for p in imported}
        assert "fake.mp3" not in names
