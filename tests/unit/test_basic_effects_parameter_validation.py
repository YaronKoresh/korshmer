import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.basic_effects import (
    apply_filter,
    apply_noise_gate,
    apply_transient_shaper,
    slice_audio_onsets,
    trim_audio_silence,
)
from korshmer.audio_processing.core_analysis import (
    trim_audio_silence as core_trim_audio_silence,
)

SR = 44100
NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]


@pytest.fixture
def audio():
    t = np.arange(SR) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def wav(tmp_path, audio):
    path = tmp_path / "in.wav"
    sf.write(path, audio, SR)
    return path


@pytest.mark.parametrize("value", NOT_NUMBERS + [-1, 1e18])
def test_filter_cutoff_rejected(value, audio, wav, tmp_path):
    with pytest.raises(ValueError, match="Cutoff"):
        apply_filter(audio, SR, "low", value)
    with pytest.raises(ValueError, match="Cutoff"):
        apply_filter(wav, tmp_path / "out.wav", "low", value)
    assert not (tmp_path / "out.wav").exists()


@pytest.mark.parametrize("ftype", ["band", "bandpass", "foo", None, 3])
def test_filter_type_rejected(ftype, audio, wav, tmp_path):
    with pytest.raises(ValueError, match="Unknown filter_type"):
        apply_filter(audio, SR, ftype, 1000)
    with pytest.raises(ValueError, match="Unknown filter_type"):
        apply_filter(wav, tmp_path / "out.wav", ftype, 1000)


def test_filter_cutoff_at_or_above_nyquist_still_passes_through(audio):
    result = apply_filter(audio, SR, "low", 30000)
    np.testing.assert_array_equal(result, audio)
    assert np.all(np.isfinite(apply_filter(audio, SR, "high", 1000)))


@pytest.mark.parametrize("value", NOT_NUMBERS + [1, 1e18, -200])
def test_gate_threshold_rejected(value, audio, wav, tmp_path):
    with pytest.raises(ValueError, match="Threshold"):
        apply_noise_gate(audio, SR, value)
    with pytest.raises(ValueError, match="Threshold"):
        apply_noise_gate(wav, tmp_path / "out.wav", value)


def test_gate_edges_still_work(audio):
    assert np.all(np.isfinite(apply_noise_gate(audio, SR, -40.0)))
    assert np.all(np.isfinite(apply_noise_gate(audio, SR, -120.0)))


@pytest.mark.parametrize("name", ["attack", "sustain"])
@pytest.mark.parametrize("value", NOT_NUMBERS + [-0.1, 11, 1e18])
def test_transient_shaper_values_rejected(name, value, audio, wav, tmp_path):
    with pytest.raises(ValueError, match=name.capitalize()):
        apply_transient_shaper(audio, SR, **{name: value})
    with pytest.raises(ValueError, match=name.capitalize()):
        apply_transient_shaper(wav, tmp_path / "out.wav", **{name: value})


@pytest.mark.parametrize(
    "value", [v for v in NOT_NUMBERS if v is not None] + [-11, 11]
)
def test_transient_shaper_amount_rejected(value, wav, tmp_path):
    with pytest.raises(ValueError, match="Amount"):
        apply_transient_shaper(wav, tmp_path / "out.wav", amount=value)


def test_transient_shaper_defaults_still_work(audio):
    assert np.all(np.isfinite(apply_transient_shaper(audio, SR, 1.5, 0.5)))


@pytest.mark.parametrize("value", NOT_NUMBERS + [0, -5, 1000])
def test_trim_threshold_rejected(value, audio, wav, tmp_path):
    for trim in (trim_audio_silence, core_trim_audio_silence):
        with pytest.raises(ValueError, match="Silence threshold"):
            trim(audio, SR, value)
        with pytest.raises(ValueError, match="Silence threshold"):
            trim(wav, str(tmp_path), value)
    assert not (tmp_path / "temp_trimmed.wav").exists()


@pytest.mark.parametrize("value", NOT_NUMBERS + [0, -5, 1000])
def test_slice_threshold_rejected(value, wav, tmp_path):
    with pytest.raises(ValueError, match="Slice threshold"):
        slice_audio_onsets(wav, str(tmp_path), value)


def test_trim_default_still_works(audio):
    assert len(trim_audio_silence(audio, SR)) > 0
