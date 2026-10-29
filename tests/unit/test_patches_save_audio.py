import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from korshmer import patches

DEMUCS_KWARGS = {
    "bitrate": 320,
    "preset": 2,
    "clip": "rescale",
    "as_float": False,
    "bits_per_sample": 16,
}


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.device = SimpleNamespace(type="cpu")

    @property
    def ndim(self):
        return self.data.ndim

    def dim(self):
        return self.data.ndim

    def __getitem__(self, key):
        return FakeTensor(self.data[key])

    def __truediv__(self, value):
        return FakeTensor(self.data / value)

    def abs(self):
        return FakeTensor(np.abs(self.data))

    def max(self):
        return float(self.data.max())

    def clamp(self, lo, hi):
        return FakeTensor(np.clip(self.data, lo, hi))

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def t(self):
        return FakeTensor(self.data.T)

    def numpy(self):
        return self.data


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    fake = SimpleNamespace(tanh=lambda wav: FakeTensor(np.tanh(wav.data)))
    monkeypatch.setattr(
        patches, "_get_torch_modules", lambda: (fake, SimpleNamespace())
    )


def _wave(peak=0.5):
    t = np.linspace(0, 1, 4410, endpoint=False)
    return FakeTensor(np.stack([peak * np.sin(2 * np.pi * 220 * t)] * 2))


def test_signature_accepts_every_keyword_demucs_passes():
    inspect.signature(patches.patched_save_audio).bind(
        None, "a.wav", 44100, **DEMUCS_KWARGS
    )


@pytest.mark.parametrize(
    ("bits", "as_float", "subtype"),
    [(16, False, "PCM_16"), (24, False, "PCM_24"), (16, True, "FLOAT")],
)
def test_wav_subtype_follows_bits_and_float_flag(
    tmp_path, bits, as_float, subtype
):
    path = tmp_path / "stem.wav"
    patches.patched_save_audio(
        _wave(),
        str(path),
        44100,
        **{**DEMUCS_KWARGS, "bits_per_sample": bits, "as_float": as_float},
    )
    info = sf.info(str(path))
    assert info.subtype == subtype
    assert info.samplerate == 44100
    assert info.channels == 2


def test_flac_is_written(tmp_path):
    path = tmp_path / "stem.flac"
    patches.patched_save_audio(_wave(), str(path), 44100, bits_per_sample=24)
    assert sf.info(str(path)).subtype == "PCM_24"


def test_rescale_keeps_loud_audio_below_full_scale(tmp_path):
    path = tmp_path / "loud.wav"
    patches.patched_save_audio(
        _wave(peak=3.0), str(path), 44100, as_float=True, clip="rescale"
    )
    data, _ = sf.read(str(path))
    assert np.abs(data).max() < 1.0


def test_clamp_limits_peaks(tmp_path):
    path = tmp_path / "clamp.wav"
    patches.patched_save_audio(
        _wave(peak=3.0), str(path), 44100, as_float=True, clip="clamp"
    )
    data, _ = sf.read(str(path))
    assert np.abs(data).max() == pytest.approx(1.0)


def test_none_clip_leaves_samples_untouched(tmp_path):
    path = tmp_path / "none.wav"
    patches.patched_save_audio(
        _wave(peak=0.25), str(path), 44100, as_float=True, clip="none"
    )
    data, _ = sf.read(str(path))
    assert np.abs(data).max() == pytest.approx(0.25, abs=1e-3)


def test_mono_input_is_written_as_one_channel(tmp_path):
    path = tmp_path / "mono.wav"
    patches.patched_save_audio(_wave()[0], str(path), 44100)
    assert sf.info(str(path)).channels == 1


def test_unknown_suffix_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid suffix"):
        patches.patched_save_audio(_wave(), str(tmp_path / "a.ogg"), 44100)
