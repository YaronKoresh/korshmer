import importlib

import numpy as np
import pytest
import soundfile as sf

MODULES = ["korshmer.audio.io", "korshmer.audio_processing.audio_io"]


@pytest.fixture(params=MODULES)
def save(request):
    return importlib.import_module(request.param).save_normalized_audio


def test_nan_sample_becomes_silence_not_a_click(save, tmp_path):
    out = tmp_path / "a.wav"
    save(out, np.array([0.1, np.nan, 0.5, -0.3], dtype=np.float32), 44100)
    data, _ = sf.read(out)
    assert abs(data[1]) < 1e-3
    assert abs(data[2]) == pytest.approx(0.95, abs=1e-3)


def test_infinite_sample_does_not_silence_the_rest(save, tmp_path):
    out = tmp_path / "a.wav"
    save(out, np.array([0.1, np.inf, 0.5, -np.inf], dtype=np.float32), 44100)
    data, _ = sf.read(out)
    assert abs(data[1]) < 1e-3
    assert abs(data[3]) < 1e-3
    assert abs(data[2]) == pytest.approx(0.95, abs=1e-3)


def test_empty_audio_raises_clear_error(save, tmp_path):
    with pytest.raises(ValueError, match="no samples"):
        save(tmp_path / "a.wav", np.zeros(0), 44100)
    assert not (tmp_path / "a.wav").exists()


def test_finite_audio_is_still_normalized(save, tmp_path):
    out = tmp_path / "a.wav"
    save(out, np.array([0.1, -0.2, 0.1], dtype=np.float32), 44100)
    data, _ = sf.read(out)
    assert np.max(np.abs(data)) == pytest.approx(0.95, abs=1e-3)
