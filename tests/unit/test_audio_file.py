import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_file import AudioFile


class TestAudioFileInit:
    def test_defaults(self):
        af = AudioFile()
        assert af.y is None
        assert af.sr == 44100
        assert af.bits == 32
        assert af.path is None

    def test_with_data(self):
        y = np.zeros(1000)
        af = AudioFile(y=y, sr=22050, bits=16, path="/tmp/test.wav")
        assert af.sr == 22050
        assert af.bits == 16
        assert af.path == "/tmp/test.wav"
        assert np.array_equal(af.y, y)

    def test_path_converted_to_str(self):
        from pathlib import Path

        af = AudioFile(path=Path("/tmp/test.wav"))
        assert isinstance(af.path, str)


class TestAudioFileProperties:
    def test_channels_mono(self):
        af = AudioFile(y=np.zeros(1000))
        assert af.channels == 1

    def test_channels_stereo(self):
        af = AudioFile(y=np.zeros((2, 1000)))
        assert af.channels == 2

    def test_channels_none(self):
        af = AudioFile()
        assert af.channels == 0

    def test_samples_mono(self):
        af = AudioFile(y=np.zeros(500))
        assert af.samples == 500

    def test_samples_stereo(self):
        af = AudioFile(y=np.zeros((2, 800)))
        assert af.samples == 800

    def test_samples_none(self):
        af = AudioFile()
        assert af.samples == 0

    def test_duration(self):
        af = AudioFile(y=np.zeros(44100), sr=44100)
        assert af.duration == pytest.approx(1.0)

    def test_duration_half_rate(self):
        af = AudioFile(y=np.zeros(22050), sr=44100)
        assert af.duration == pytest.approx(0.5)

    def test_duration_none(self):
        af = AudioFile()
        assert af.duration == 0.0

    def test_is_mono(self):
        af = AudioFile(y=np.zeros(1000))
        assert af.is_mono is True
        assert af.is_stereo is False

    def test_is_stereo(self):
        af = AudioFile(y=np.zeros((2, 1000)))
        assert af.is_mono is False
        assert af.is_stereo is True

    def test_not_stereo_multichannel(self):
        af = AudioFile(y=np.zeros((4, 1000)))
        assert af.is_stereo is False


class TestEnsure2D:
    def test_mono_becomes_2d(self):
        af = AudioFile(y=np.ones(100))
        assert af.y.ndim == 1
        af.ensure_2d()
        assert af.y.ndim == 2
        assert af.y.shape == (1, 100)

    def test_stereo_unchanged(self):
        y = np.ones((2, 100))
        af = AudioFile(y=y)
        af.ensure_2d()
        assert af.y.shape == (2, 100)

    def test_returns_self(self):
        af = AudioFile(y=np.ones(100))
        result = af.ensure_2d()
        assert result is af

    def test_none_data(self):
        af = AudioFile()
        af.ensure_2d()
        assert af.y is None


class TestWriteFormat:
    def test_mono(self):
        af = AudioFile(y=np.ones(100))
        out = af.to_write_format()
        assert out.shape == (100,)

    def test_stereo_transposed(self):
        af = AudioFile(y=np.ones((2, 100)))
        out = af.to_write_format()
        assert out.shape == (100, 2)

    def test_none(self):
        af = AudioFile()
        assert af.to_write_format() is None


class TestSave:
    def test_save_mono(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.wav")
            af.save(out)
            assert os.path.exists(out)
            data, sr = sf.read(out)
            assert sr == 44100
            assert len(data) == 4410

    def test_save_stereo(self):
        y = np.random.uniform(-0.5, 0.5, (2, 4410)).astype(np.float32)
        af = AudioFile(y=y, sr=22050)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.wav")
            af.save(out)
            data, sr = sf.read(out)
            assert sr == 22050
            assert data.shape == (4410, 2)

    def test_save_creates_dirs(self):
        y = np.zeros(100, dtype=np.float32)
        af = AudioFile(y=y, sr=44100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sub", "dir", "out.wav")
            af.save(out)
            assert os.path.exists(out)

    def test_save_updates_path(self):
        af = AudioFile(y=np.zeros(100, dtype=np.float32), sr=44100)
        assert af.path is None
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.wav")
            af.save(out)
            assert af.path == out

    def test_save_no_data_raises(self):
        af = AudioFile()
        with pytest.raises(ValueError, match="No audio data"):
            af.save("/tmp/test.wav")


class TestLoad:
    def test_load_mono(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            sf.write(path, y, 44100)
            af = AudioFile.load(path, mono=True)
            assert af.sr == 44100
            assert af.is_mono
            assert af.path == path
            assert len(af.y) == 4410

    def test_load_stereo(self):
        y = np.random.uniform(-0.5, 0.5, (4410, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            sf.write(path, y, 22050)
            af = AudioFile.load(path)
            assert af.sr == 22050

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            AudioFile.load("/nonexistent/path.wav")

    def test_load_not_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="not a file"):
                AudioFile.load(tmpdir)


class TestCopy:
    def test_copy_independent(self):
        y = np.ones(100)
        af = AudioFile(y=y, sr=22050, bits=16, path="/tmp/a.wav")
        af2 = af.copy()
        af2.y[0] = 999
        assert af.y[0] == 1.0
        assert af2.sr == 22050
        assert af2.bits == 16
        assert af2.path == "/tmp/a.wav"

    def test_copy_none(self):
        af = AudioFile()
        af2 = af.copy()
        assert af2.y is None


class TestRepr:
    def test_repr(self):
        af = AudioFile(y=np.zeros((2, 44100)), sr=44100, bits=32)
        r = repr(af)
        assert "sr=44100" in r
        assert "channels=2" in r
        assert "1.00s" in r
        assert "bits=32" in r


class TestSaveFlac:
    def test_save_flac_mono(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=24)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.flac")
            af.save(out)
            assert os.path.exists(out)
            info = sf.info(out)
            assert info.samplerate == 44100
            assert info.format == "FLAC"

    def test_save_flac_stereo(self):
        y = np.random.uniform(-0.5, 0.5, (2, 4410)).astype(np.float32)
        af = AudioFile(y=y, sr=48000, bits=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.flac")
            af.save(out)
            info = sf.info(out)
            assert info.samplerate == 48000
            assert info.channels == 2
            assert info.subtype == "PCM_16"

    def test_save_flac_fallback_bit_depth(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.flac")
            af.save(out)
            info = sf.info(out)
            assert info.subtype == "PCM_24"


class TestSaveAiff:
    def test_save_aiff_mono(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.aiff")
            af.save(out)
            assert os.path.exists(out)
            info = sf.info(out)
            assert info.samplerate == 44100
            assert info.format == "AIFF"

    def test_save_aiff_24bit(self):
        y = np.random.uniform(-0.5, 0.5, (2, 4410)).astype(np.float32)
        af = AudioFile(y=y, sr=48000, bits=24)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.aiff")
            af.save(out)
            info = sf.info(out)
            assert info.samplerate == 48000
            assert info.channels == 2
            assert info.subtype == "PCM_24"

    def test_save_aif_extension(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.aif")
            af.save(out)
            info = sf.info(out)
            assert info.format == "AIFF"

    def test_save_aiff_32bit_float(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.aiff")
            af.save(out)
            info = sf.info(out)
            assert info.subtype == "FLOAT"


class TestSaveFormatOverride:
    def test_explicit_format(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=24)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.dat")
            af.save(out, fmt="FLAC")
            info = sf.info(out)
            assert info.format == "FLAC"

    def test_wav_subtype_selection(self):
        y = np.random.uniform(-0.5, 0.5, 4410).astype(np.float32)
        af = AudioFile(y=y, sr=44100, bits=16)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.wav")
            af.save(out)
            info = sf.info(out)
            assert info.subtype == "PCM_16"


class TestYSetter:
    def test_set_y(self):
        af = AudioFile(y=np.zeros(100))
        new_y = np.ones(200)
        af.y = new_y
        assert af.samples == 200
        assert np.array_equal(af.y, new_y)
