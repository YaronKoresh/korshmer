import numpy as np
import pretty_midi
import pytest

from korshmer.audio_processing import guitar_amp
from korshmer.audio_processing.timbre import (
    CLEAN_ELECTRIC_GUITAR,
    DISTORTED_GUITAR,
    PLUCKED_KEYS,
    SOFT_SUSTAINED,
    analyze_timbre,
    preferred_programs,
)
from korshmer.midi_algorithms import smart_instrument_selector
from tests.fixtures.timbre_signals import (
    SR,
    bright_lead,
    distorted,
    plucked,
    soft_strings,
)


@pytest.fixture(scope="module")
def profiles():
    return {
        name: analyze_timbre(make(), SR)
        for name, make in {
            "plucked": plucked,
            "distorted": distorted,
            "strings": soft_strings,
            "lead": bright_lead,
        }.items()
    }


def test_distortion_is_detected_only_on_the_distorted_stem(profiles):
    assert profiles["distorted"].distorted
    for name in ("plucked", "strings", "lead"):
        assert not profiles[name].distorted, name


def test_picked_notes_are_plucked_and_held_notes_are_not(profiles):
    assert profiles["plucked"].plucked
    assert not profiles["strings"].plucked
    assert not profiles["lead"].plucked


def test_other_stem_gets_an_instrument_that_matches_its_sound(profiles):
    assert preferred_programs("other", profiles["distorted"])[0] in (29, 30)
    assert preferred_programs("other", profiles["plucked"]) == PLUCKED_KEYS
    assert preferred_programs("other", profiles["lead"])[0] == 81
    assert preferred_programs("other", profiles["strings"]) == SOFT_SUSTAINED


def test_guitar_stem_choice(profiles):
    assert preferred_programs("guitar", profiles["distorted"]) == (
        DISTORTED_GUITAR
    )
    assert preferred_programs("guitar", profiles["plucked"]) == (
        CLEAN_ELECTRIC_GUITAR
    )


def test_untouched_stems_and_unknown_audio():
    assert preferred_programs("bass", None) == ()
    assert preferred_programs("drums", object()) == ()
    assert analyze_timbre(np.zeros(SR * 5), SR) is None
    assert analyze_timbre(np.zeros(100), SR) is None


def _riff_midi():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i, p in enumerate([40, 43, 45, 52, 55, 57] * 3):
        inst.notes.append(pretty_midi.Note(100, p, i * 0.25, i * 0.25 + 0.2))
    pm.instruments.append(inst)
    return pm


def test_selector_uses_the_preferred_instrument():
    plain = smart_instrument_selector("other", _riff_midi())
    assert plain.instruments[0].program not in DISTORTED_GUITAR
    picked = smart_instrument_selector(
        "other", _riff_midi(), preferred_programs=DISTORTED_GUITAR
    )
    assert picked.instruments[0].program == 29
    assert "Overdrive" in picked.instruments[0].name


def test_amp_distorts_keeps_level_and_makes_stereo():
    t = np.arange(SR) / SR
    clean = 0.3 * np.sin(2 * np.pi * 110 * t)
    out = guitar_amp.amp(clean, SR, guitar_amp.HIGH_GAIN)
    assert out.shape == (SR, 2)
    assert out.dtype == np.float32
    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    assert rms(out[:, 0]) == pytest.approx(rms(clean), rel=0.05)
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    harmonics = spectrum[(freqs > 300) & (freqs < 3000)].sum()
    fundamental = spectrum[(freqs > 100) & (freqs < 120)].sum()
    assert harmonics > fundamental
    assert not np.allclose(out[:, 0], out[:, 1])


def test_amp_handles_silence_and_empty_input():
    assert guitar_amp.amp(np.zeros(1000), SR).shape == (1000, 2)
    assert not guitar_amp.amp(np.zeros(1000), SR).any()
    assert guitar_amp.amp(np.zeros(0), SR).shape == (0, 2)


def test_only_electric_guitar_programs_are_amped():
    assert set(guitar_amp.AMP_FOR_PROGRAM) == {27, 28, 29, 30}


def test_render_sends_only_guitar_tracks_through_the_amp(monkeypatch):
    from korshmer import audio_processors

    calls = []

    def fake_batch(instruments, level=True):
        calls.append(([i.program for i in instruments], level))
        return np.full((SR, 2), 0.1, np.float32)

    amped = []
    monkeypatch.setattr(
        guitar_amp,
        "amp",
        lambda audio, sr, settings: amped.append(settings) or audio,
    )
    piano = pretty_midi.Instrument(0)
    piano.notes.append(pretty_midi.Note(90, 60, 0, 1))
    guitar = pretty_midi.Instrument(30)
    guitar.notes.append(pretty_midi.Note(90, 52, 0, 1))
    mix = audio_processors._render_with_guitar_amps(
        [piano, guitar], fake_batch, SR
    )
    assert calls == [([0], False), ([30], False)]
    assert amped == [guitar_amp.HIGH_GAIN]
    assert mix.shape == (SR, 2)

    calls.clear()
    audio_processors._render_with_guitar_amps([piano], fake_batch, SR)
    assert calls == [([0], True)]
