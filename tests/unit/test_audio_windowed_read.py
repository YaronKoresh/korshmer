import numpy as np
import pytest
import soundfile as sf

import korshmer.audio.loader as loader
from korshmer.audio.loader import read_audio_frames


@pytest.fixture
def ramp_wav(tmp_path):
    path = tmp_path / "ramp.wav"
    data = np.linspace(-0.5, 0.5, 1000, dtype="float32").reshape(-1, 1)
    sf.write(str(path), data, 44100, subtype="FLOAT")
    return path, data


def test_window_returns_only_requested_frames(ramp_wav):
    path, data = ramp_wav
    window = read_audio_frames(path, 100, 300)
    assert window.dtype == np.float32
    assert window.shape == (200, 1)
    np.testing.assert_allclose(window, data[100:300], atol=1e-6)


def test_full_range_matches_full_decode(ramp_wav):
    path, data = ramp_wav
    full = read_audio_frames(path, 0, len(data))
    np.testing.assert_allclose(full, data, atol=1e-6)


def test_stereo_window_keeps_channels(tmp_path):
    path = tmp_path / "stereo.wav"
    data = np.random.default_rng(0).uniform(-1, 1, (500, 2)).astype("float32")
    sf.write(str(path), data, 44100, subtype="FLOAT")
    window = read_audio_frames(path, 50, 120)
    assert window.shape == (70, 2)
    np.testing.assert_allclose(window, data[50:120], atol=1e-6)


def test_window_request_never_asks_for_whole_file(ramp_wav, monkeypatch):
    path, _ = ramp_wav
    requested = []
    real_soundfile = sf.SoundFile

    class RecordingSoundFile(real_soundfile):
        def read(self, frames=-1, *args, **kwargs):
            requested.append(frames)
            return super().read(frames, *args, **kwargs)

    monkeypatch.setattr(loader.sf, "SoundFile", RecordingSoundFile)
    read_audio_frames(path, 400, 410)
    assert requested == [10]


@pytest.mark.parametrize(
    "start, stop",
    [(-1, 10), (10, 10), (20, 10), (0, 1001)],
)
def test_out_of_range_window_is_rejected(ramp_wav, start, stop):
    path, _ = ramp_wav
    with pytest.raises(ValueError, match="outside the file"):
        read_audio_frames(path, start, stop)


@pytest.mark.parametrize("start, stop", [(0.5, 10), (0, "10"), (True, 10)])
def test_non_integer_bounds_are_rejected(ramp_wav, start, stop):
    path, _ = ramp_wav
    with pytest.raises(ValueError, match="must be an integer"):
        read_audio_frames(path, start, stop)


def test_numpy_integer_bounds_are_accepted(ramp_wav):
    path, data = ramp_wav
    window = read_audio_frames(path, np.int64(5), np.int64(15))
    np.testing.assert_allclose(window, data[5:15], atol=1e-6)
