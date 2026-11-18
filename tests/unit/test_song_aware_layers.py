import random

import numpy as np
import pretty_midi
import pytest

from korshmer.audio_processing import vocal_melody
from korshmer.audio_processing.vocal_melody import VocalMelodyOptions
from korshmer.midi_generation import composer
from korshmer.midi_generation.composer import (
    ANSWER_NAME,
    ComposeOptions,
    answer_phrases,
    compose,
    lead_spans,
    match_palette,
    resolve_style,
    song_style,
    sung_share,
)

BPM = 148.0
BAR = 240.0 / BPM
BEAT = BAR / 4
BARS = 32


def _song(programs=(0, 48, 25), vocals=True):
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    for program in programs:
        inst = pretty_midi.Instrument(program)
        for index in range(BARS):
            chord = (57, 60, 64) if index % 4 < 2 else (53, 57, 60)
            for pitch in chord:
                inst.notes.append(
                    pretty_midi.Note(70, pitch, index * BAR, (index + 1) * BAR)
                )
        song.instruments.append(inst)
    if vocals:
        voice = pretty_midi.Instrument(53, name="Vocals")
        for index in range(0, BARS, 4):
            start = index * BAR
            for step in range(12):
                at = start + step * BEAT / 2
                voice.notes.append(
                    pretty_midi.Note(80, 69 + step % 5, at, at + BEAT / 2)
                )
            held = start + 6 * BEAT
            voice.notes.append(pretty_midi.Note(80, 72, held, held + BEAT * 2))
        song.instruments.append(voice)
    return song


def _layer(song, seed=3, style=composer.AUTO_STYLE):
    options = ComposeOptions(
        "A",
        "Minor",
        bpm=BPM,
        style=style,
        seed=seed,
        mode="Layer over editor",
        roles=composer.ROLE_KEYS,
    )
    return compose(options, existing=song)


def test_an_acoustic_song_does_not_get_a_phonk_layer():
    assert resolve_style(composer.AUTO_STYLE, BPM, "Minor") == "Phonk"
    assert song_style(_song(), BPM, "Minor") == "Pop"
    assert _layer(_song()).style == "Pop"


@pytest.mark.parametrize(
    "programs, bpm, expected",
    [
        ((81, 89, 38), 148.0, "Phonk"),
        ((30, 29, 33), 148.0, "Rock"),
        ((0, 48), 70.0, "Lo-Fi"),
        ((0, 48, 25), 110.0, "Pop"),
    ],
)
def test_auto_style_is_read_from_the_song_instruments(programs, bpm, expected):
    song = _song(programs, vocals=False)
    assert song_style(song, bpm, "Minor") == expected
    assert resolve_style("Jazz", bpm, "Minor", song) == "Jazz"


def test_an_empty_editor_falls_back_to_tempo_and_scale():
    empty = pretty_midi.PrettyMIDI()
    assert song_style(empty, 148.0, "Minor") == "Phonk"


def test_layer_sounds_come_from_the_families_in_the_song():
    song = _song((0, 48))
    programs = {"pad": (89, 50, 49), "chords": (0, 4, 25), "arp": (98, 100)}
    for seed in range(5):
        picked = match_palette(random.Random(seed), programs, song)
        assert picked["pad"] in (50, 49)
        assert picked["chords"] in (0, 4)
        assert picked["arp"] in (98, 100)


def test_lead_spans_join_close_notes_and_measure_singing():
    song = _song()
    voice = song.instruments[-1].notes
    spans = lead_spans(voice, 0.0, 4 * BAR, BEAT)
    assert len(spans) == 1
    assert spans[0][0] == 0.0 and spans[0][1] == pytest.approx(8 * BEAT)
    assert sung_share(voice, 0.0, 4 * BAR, BEAT) == pytest.approx(0.5)
    assert sung_share(voice, 2 * BAR, 4 * BAR, BEAT) == 0.0


def test_answers_are_kept_only_in_the_gaps_of_the_lead():
    lead = [pretty_midi.Note(80, 60, 0.0, 2.0)]
    tune = [
        pretty_midi.Note(80, 64, start, start + 0.6)
        for start in (0.5, 1.5, 2.2, 3.0, 3.8)
    ]
    kept = answer_phrases(tune, lead, 0.0, 4.0, 0.5)
    assert [n.start for n in kept] == [2.2, 3.0, 3.8]
    assert kept[-1].end == 4.0
    none = answer_phrases(tune, lead, 0.0, 2.5, 0.5)
    assert none == []


def _voice_of(made, word):
    return [
        n
        for inst in made.instruments
        if word in (inst.name or "")
        for n in inst.notes
    ]


def _while_sung(notes, vocal):
    spans = lead_spans(vocal, 0.0, BARS * BAR, BEAT)
    inside = [n for n in notes if any(a <= n.start < b for a, b in spans)]
    return len(inside) / len(notes)


@pytest.mark.parametrize("seed", [3, 7])
def test_the_layer_answers_the_singer_instead_of_playing_over_them(seed):
    song = _song()
    vocal = song.instruments[-1].notes
    made = _layer(song, seed)
    answers = _voice_of(made, ANSWER_NAME)
    assert len(answers) >= 16
    assert _while_sung(answers, vocal) <= 0.1
    harmony = _voice_of(made, "Melody harmony")
    assert 0 < len(harmony) <= len(vocal) / 4
    starts = {round(n.start, 3) for n in vocal if n.end - n.start >= BEAT}
    assert {round(n.start, 3) for n in harmony} <= starts


def test_busy_parts_thin_out_while_the_singer_sings():
    song = _song()
    vocal = song.instruments[-1].notes
    made = _layer(song, 3)
    names = [inst.name for inst in made.instruments]
    assert not any("Pluck" in name for name in names) or (
        _while_sung(_voice_of(made, "Pluck"), vocal) <= 0.1
    )
    assert composer._calm_rate(("1/16", "1/8")) == "1/8"
    assert composer._calm_rate(("1/8 Triplet", "1/8")) == "1/8"


def test_a_song_without_a_singer_keeps_the_full_harmony():
    made = _layer(_song(vocals=False), 3)
    assert _voice_of(made, ANSWER_NAME) == []


def _spans():
    return [(0, 40, 60), (50, 90, 62), (100, 140, 64)]


def test_a_loud_note_is_kept_even_when_the_pitch_tracker_is_unsure():
    probability = np.full(160, 0.5)
    probability[50:90] = 0.001
    probability[100:140] = 0.001
    level = np.full(160, -20.0)
    level[50:90] = -24.0
    level[100:140] = -45.0
    kept = vocal_melody._confident_or_loud(
        _spans(), probability, level, VocalMelodyOptions()
    )
    assert [s[2] for s in kept] == [60, 62]
    strict = vocal_melody._confident_or_loud(
        _spans(), probability, level, VocalMelodyOptions(loud_voice_db=0.0)
    )
    assert [s[2] for s in strict] == [60]
    with pytest.raises(ValueError, match="Loud voice range"):
        VocalMelodyOptions(loud_voice_db=-1)


def test_a_short_semitone_wobble_becomes_one_held_note():
    hop = 256 / 22050
    wobble = [(0, 40, 60), (40, 46, 61), (46, 90, 60), (90, 130, 64)]
    assert vocal_melody._join_wobbles(wobble, hop) == [
        (0, 90, 60),
        (90, 130, 64),
    ]
    step = [(0, 40, 60), (40, 80, 61), (80, 120, 60)]
    assert vocal_melody._join_wobbles(step, hop) == step
    apart = [(0, 40, 60), (44, 50, 61), (50, 90, 60)]
    assert vocal_melody._join_wobbles(apart, hop) == apart
