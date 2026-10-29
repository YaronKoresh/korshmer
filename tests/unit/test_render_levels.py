import numpy as np
import pretty_midi
import pytest

import korshmer.audio_processors as ap


def _rms_db(audio):
    return 20 * np.log10(np.sqrt(np.mean(np.square(audio))) + 1e-12)


def test_level_render_lifts_quiet_peaky_audio_without_clipping():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.01, (44100, 2)).astype(np.float32)
    audio[::5000] = 0.8
    leveled = ap.level_render(audio)
    assert _rms_db(leveled) > _rms_db(audio) + 8
    assert np.max(np.abs(leveled)) <= 1.0


def test_level_render_keeps_silence_silent():
    audio = np.zeros((1000, 2), dtype=np.float32)
    assert not np.any(ap.level_render(audio))


def test_chorus_is_enabled_and_reverb_is_moderate():
    assert ap.CHORUS_SETTINGS[0] >= 1
    assert ap.CHORUS_SETTINGS[1] > 0.0
    assert ap.REVERB_SETTINGS[3] <= 0.3


def test_render_applies_the_render_settings(tmp_path, monkeypatch):
    calls = {}

    class Fake:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def sf_load(self, path):
            return 1

        def set_reverb(self, *args):
            calls["reverb"] = args

        def set_chorus(self, *args):
            calls["chorus"] = args

        def program_select(self, *args):
            pass

        def note_on(self, *args):
            pass

        def note_off(self, *args):
            pass

        def get_samples(self, n):
            return np.full(n, 0.01), np.full(n, 0.01)

    monkeypatch.setattr(ap, "FluidSynth", Fake)
    monkeypatch.setattr(ap, "save_normalized_audio", lambda *a, **k: None)
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(100, 60, 0.0, 0.5))
    midi.instruments.append(inst)
    path = tmp_path / "a.mid"
    midi.write(str(path))
    ap.render_midi_to_wav(str(path), str(tmp_path / "o.wav"), "x", "y")
    assert calls["reverb"] == ap.REVERB_SETTINGS
    assert calls["chorus"] == ap.CHORUS_SETTINGS


def test_real_synth_stereo_stays_in_phase():
    from pathlib import Path

    from korshmer.core.fluidsynth import FluidSynth

    fluid_dir = Path(ap.__file__).resolve().parents[2] / "fluidsynth"
    sf3 = fluid_dir / "soundfont.sf3"
    if not sf3.exists():
        pytest.skip("soundfont missing")
    try:
        synth = FluidSynth(lib_dir=str(fluid_dir), samplerate=22050)
    except Exception:
        pytest.skip("FluidSynth library unavailable")
    with synth as fs:
        sfid = fs.sf_load(str(sf3))
        fs.set_reverb(*ap.REVERB_SETTINGS)
        fs.set_chorus(*ap.CHORUS_SETTINGS)
        fs.program_select(0, sfid, 0, 27)
        left, right = [], []
        for pitch in (48, 60, 72):
            fs.note_on(0, pitch, 100)
            lo, ro = fs.get_samples(11025)
            fs.note_off(0, pitch)
            left.append(lo)
            right.append(ro)
    corr = np.corrcoef(np.concatenate(left), np.concatenate(right))[0, 1]
    assert corr > 0.8
