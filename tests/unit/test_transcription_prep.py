import inspect

import librosa
import numpy as np
import pytest
import soundfile as sf

from korshmer import track_processor
from korshmer.audio_processing.transcription_prep import (
    ModelOptions,
    PrepOptions,
    band_limit,
    harmonic_part,
    highpass,
    normalize,
    octave_layer,
    options_for_stem,
    prepare_for_transcription,
    process_audio,
    soften_transients,
    to_mono,
    variants_for_stem,
)
from tests.fixtures.transcription_fixtures import (
    FIXTURE_DIR,
    SONGS,
    load_truth,
    note_scores,
)

SR = 22050


def _read(name):
    audio, sr = sf.read(FIXTURE_DIR / f"{name}.wav")
    return audio, sr


def _absolute_threshold_notes(y, sr, floor=2.0, min_len=0.08):
    hop = 256
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop,
            fmin=librosa.midi_to_hz(24),
            n_bins=72,
            bins_per_octave=12,
        )
    )
    notes = []
    for k in range(cqt.shape[0]):
        active = cqt[k] > floor
        i = 0
        while i < len(active):
            if not active[i]:
                i += 1
                continue
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) * hop / sr >= min_len:
                notes.append((24 + k, i * hop / sr, j * hop / sr))
            i = j
    return notes


def test_to_mono_handles_both_layouts():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1)
    assert to_mono(stereo).shape == (100,)
    assert np.allclose(to_mono(stereo), 0.5)
    assert np.allclose(to_mono(stereo.T), 0.5)


def test_highpass_removes_dc_and_rumble():
    t = np.arange(SR) / SR
    y = 0.5 + np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 440 * t)
    out = highpass(y, SR, 30.0)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    assert spec[freqs < 10].max() < 0.02 * spec.max()
    assert abs(np.mean(out)) < 0.01


def test_band_limit_cuts_outside_band():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 6000 * t)
    out = band_limit(y, SR, 35.0, 1200.0)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    assert spec[np.argmin(abs(freqs - 6000))] < 0.01 * spec.max()


def test_normalize_sets_peak():
    out = normalize(np.array([0.01, -0.02, 0.0]), -6.0)
    assert np.isclose(np.max(np.abs(out)), 10 ** (-6 / 20))
    assert np.all(normalize(np.zeros(5), -3.0) == 0)


def test_harmonic_part_reduces_hat_energy():
    audio, sr = _read("full_mix_bb_major")
    y = to_mono(audio)
    out = harmonic_part(y, 2.0)
    hi = librosa.stft(y)[300:]
    hi_out = librosa.stft(out)[300:]
    assert np.abs(hi_out).sum() < np.abs(hi).sum()


def test_soften_transients_lowers_attack_more_than_sustain():
    t = np.arange(SR // 2) / SR
    y = np.sin(2 * np.pi * 220 * t) * 0.3
    y[2000:2100] += 0.8 * np.sin(2 * np.pi * 3000 * t[:100])
    out = soften_transients(y, SR, 1.0)
    assert np.abs(out[2000:2100]).max() < np.abs(y[2000:2100]).max()
    assert np.allclose(out[8000:9000], y[8000:9000], atol=0.05)
    assert np.array_equal(soften_transients(y, SR, 0.0), y)


def test_octave_layer_adds_energy_an_octave_up():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 220 * t)
    out = octave_layer(y, SR, 0.5)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    assert spec[np.argmin(abs(freqs - 440))] > 0.1 * spec.max()
    assert np.array_equal(octave_layer(y, SR, 0.0), y)


def test_process_audio_resamples_and_levels():
    audio, sr = _read("piano_c_major")
    quiet = audio * 0.02
    out, rate = process_audio(quiet, sr, PrepOptions())
    assert rate == SR
    assert out.ndim == 1
    assert np.isclose(np.max(np.abs(out)), 10 ** (-3 / 20))
    wide, rate = process_audio(
        np.stack([audio[:, 0]] * 2, axis=1), 44100, PrepOptions()
    )
    assert rate == SR
    assert abs(len(wide) - len(audio) / 2) < 5


def test_options_for_stem():
    assert options_for_stem("drums.wav").enabled is False
    bass = options_for_stem("bass.wav")
    assert bass.band_limit and bass.instrument == "bass"
    assert options_for_stem("other.wav").band_limit is False
    assert options_for_stem("vocals.wav").instrument == "vocals"


def test_prepare_writes_mono_file_and_falls_back(tmp_path):
    src = FIXTURE_DIR / "lead_e_minor.wav"
    out = prepare_for_transcription(str(src), str(tmp_path))
    info = sf.info(out)
    assert info.channels == 1 and info.samplerate == SR
    assert prepare_for_transcription(
        str(src), str(tmp_path), PrepOptions(enabled=False)
    ) == str(src)
    missing = str(tmp_path / "nope.wav")
    assert prepare_for_transcription(missing, str(tmp_path)) == missing


def test_track_processor_accepts_prep_options():
    params = inspect.signature(track_processor.process_track).parameters
    assert params["prep_options"].default is None


def test_normalizing_quiet_stems_restores_notes_for_fixed_threshold():
    before, after = [], []
    for song in SONGS:
        audio, sr = _read(song[0])
        truth = load_truth(song[0])
        quiet = to_mono(audio) * 0.03
        prepared, rate = process_audio(quiet, sr, PrepOptions())
        before.append(
            note_scores(truth, _absolute_threshold_notes(quiet, sr), 0.08)[2]
        )
        after.append(
            note_scores(truth, _absolute_threshold_notes(prepared, rate), 0.08)[
                2
            ]
        )
    print("F1 before", np.round(before, 2), "after", np.round(after, 2))
    assert np.mean(after) > np.mean(before) + 0.3


def test_variants_pair_main_with_sustained_only_copy():
    main, sustained = variants_for_stem("piano.wav")
    assert not main.harmonic_only
    assert sustained.harmonic_only and sustained.harmonic_margin == 1.0
    assert sustained.instrument == main.instrument == "piano"
    assert len(variants_for_stem("drums.wav")) == 1
    assert len(variants_for_stem("x.wav", PrepOptions(ensemble=False))) == 1
    assert len(variants_for_stem("x.wav", PrepOptions(enabled=False))) == 1


def test_model_options_default_and_kwargs():
    options = ModelOptions()
    kwargs = options.as_kwargs()
    assert kwargs["onset_threshold"] == 0.55
    assert kwargs["frame_threshold"] == 0.375
    assert set(kwargs) == {
        "onset_threshold",
        "frame_threshold",
        "minimum_note_length",
        "minimum_frequency",
        "maximum_frequency",
    }


@pytest.mark.parametrize(
    "name", ["bass.wav", "Song_Bass.WAV", "my bass stem.wav"]
)
def test_bass_stems_get_softer_model_thresholds(name):
    bass = ModelOptions.for_stem(name)
    general = ModelOptions()

    assert bass.onset_threshold < general.onset_threshold
    assert bass.frame_threshold < general.frame_threshold
    assert bass.maximum_frequency <= 500
    assert bass.minimum_frequency <= 41


@pytest.mark.parametrize("name", ["vocals.wav", "guitar.wav", "piano.wav"])
def test_other_stems_keep_the_general_model_thresholds(name):
    assert ModelOptions.for_stem(name) == ModelOptions()


def test_prepare_tag_keeps_variant_files_apart(tmp_path):
    src = FIXTURE_DIR / "lead_e_minor.wav"
    a = prepare_for_transcription(str(src), str(tmp_path), tag="_0")
    b = prepare_for_transcription(str(src), str(tmp_path), tag="_1")
    assert a != b and a.endswith("_prep_0.wav") and b.endswith("_prep_1.wav")


def test_bass_stem_is_transcribed_once_without_voting():
    assert len(variants_for_stem("bass.wav")) == 1
    assert variants_for_stem("bass.wav")[0].instrument == "bass"
    assert len(variants_for_stem("piano.wav")) == 2
