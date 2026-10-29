import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing import basic_effects, core_analysis

SAMPLE_RATE = 22050


@pytest.fixture
def wav(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "in.wav"
    audio = (rng.standard_normal(SAMPLE_RATE) * 0.1).astype("float32")
    sf.write(path, audio, SAMPLE_RATE)
    return str(path)


@pytest.mark.parametrize(
    "func",
    [
        basic_effects.apply_filter,
        basic_effects.apply_normalize,
        basic_effects.apply_reverse_audio,
        basic_effects.apply_mono_mix,
        basic_effects.apply_transient_shaper,
        basic_effects.apply_noise_gate,
    ],
)
@pytest.mark.parametrize("target", [None, "", "   ", 44100])
def test_file_effects_reject_missing_output_file(wav, func, target):
    with pytest.raises(ValueError, match="Output file path is required"):
        func(wav, target)


@pytest.mark.parametrize(
    "func",
    [
        basic_effects.trim_audio_silence,
        basic_effects.slice_audio_onsets,
        core_analysis.trim_audio_silence,
    ],
)
@pytest.mark.parametrize("target", [None, "", 44100])
def test_file_effects_reject_missing_output_folder(wav, func, target):
    with pytest.raises(ValueError, match="Output folder is required"):
        func(wav, target)


def test_apply_filter_honours_output_path_keyword(wav, tmp_path):
    out = tmp_path / "filtered.wav"
    basic_effects.apply_filter(wav, output_path=str(out))
    assert out.exists()
    data, rate = sf.read(out)
    assert rate == SAMPLE_RATE
    assert len(data) == SAMPLE_RATE


def test_apply_filter_positional_output_still_works(wav, tmp_path):
    out = tmp_path / "filtered.wav"
    basic_effects.apply_filter(
        wav, str(out), filter_type="highpass", cutoff=500
    )
    assert out.exists()
