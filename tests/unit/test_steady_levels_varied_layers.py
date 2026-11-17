import math
import random

import pretty_midi

from korshmer.midi_generation import composer
from korshmer.midi_generation.composer import (
    AUDIBLE_VELOCITY,
    ComposeOptions,
    Section,
    compose,
    keep_audible,
    song_bar_gains,
    vary_phrase,
)

BPM = 120.0
BAR = 240.0 / BPM
BLOCK = 4
STEP_DB = 3.0


def _wobbly_song(bars=64, seed=7, drums=False):
    rng = random.Random(seed)
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    lead = pretty_midi.Instrument(0, name="Vocals")
    keys = pretty_midi.Instrument(4, name="Keys")
    kit = pretty_midi.Instrument(0, is_drum=True, name="Drums")
    for index in range(bars):
        level = 60 + rng.choice((-14, -9, 0, 8, 13))
        chord = (57, 60, 64) if index % 4 < 2 else (53, 57, 60)
        for step in range(4):
            start = index * BAR + step * BAR / 4
            lead.notes.append(
                pretty_midi.Note(
                    max(1, level - 55 * (rng.random() < 0.1)),
                    69 + (step * 2 + index) % 7,
                    start,
                    start + 0.3,
                )
            )
            kit.notes.append(pretty_midi.Note(90, 36, start, start + 0.1))
        for pitch in chord:
            keys.notes.append(
                pretty_midi.Note(level, pitch, index * BAR, (index + 1) * BAR)
            )
    song.instruments += [lead, keys] + ([kit] if drums else [])
    return song


def _layer(style="Phonk", seed=3, drums=False):
    options = ComposeOptions(
        "A",
        "Minor",
        bpm=BPM,
        style=style,
        seed=seed,
        mode="Layer over editor",
        roles=("drums", "bass", "chords", "pad", "arp", "melody"),
    )
    return compose(options, existing=_wobbly_song(drums=drums))


def _block_levels(made):
    levels = []
    for first in range(0, made.total_bars, BLOCK):
        lo = first * BAR + made.grid_offset
        hi = lo + BLOCK * BAR
        heard = [
            n.velocity
            for inst in made.instruments
            for n in inst.notes
            if lo <= n.start < hi
        ]
        if heard:
            levels.append(sum(heard) / len(heard))
    return levels


def _steps_db(levels):
    return [
        abs(40 * math.log10(after / before))
        for before, after in zip(levels, levels[1:], strict=False)
    ]


def test_a_layer_does_not_pump_with_the_noise_in_the_song_velocities():
    for style, seed in (("Phonk", 3), ("Pop", 5)):
        body = _block_levels(_layer(style, seed))[1:-1]
        assert max(_steps_db(body)) < STEP_DB, style
        assert max(body) / min(body) < 1.25, style


def test_no_layer_note_is_too_quiet_to_hear():
    made = _layer()
    quietest = min(n.velocity for inst in made.instruments for n in inst.notes)
    assert quietest >= AUDIBLE_VELOCITY
    voice = pretty_midi.Instrument(0)
    voice.notes = [pretty_midi.Note(3, 60, 0.0, 1.0)]
    keep_audible([("melody", voice)])
    assert voice.notes[0].velocity == AUDIBLE_VELOCITY


def test_small_level_changes_are_ignored_and_big_ones_followed():
    song = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    keys = pretty_midi.Instrument(0)
    for index, level in enumerate((60, 66, 55, 60, 60, 60, 20, 20)):
        keys.notes.append(
            pretty_midi.Note(level, 60, index * BAR, (index + 1) * BAR)
        )
    song.instruments.append(keys)
    sections = [Section("Verse", 0, 6, 0.6), Section("Break", 6, 2, 0.4)]
    gains = song_bar_gains(song, BAR, 8, sections)
    assert gains[:6] == [1.0] * 6
    assert gains[6:] == [composer.SONG_GAIN_RANGE[0]] * 2
    per_bar = song_bar_gains(song, BAR, 8)
    assert per_bar[1] == per_bar[2] == 1.0


def _bar_patterns(inst, made):
    patterns = []
    for index in range(made.total_bars):
        lo = index * BAR + made.grid_offset
        notes = [n for n in inst.notes if lo <= n.start < lo + BAR]
        if notes:
            patterns.append(
                tuple(
                    sorted(
                        (round((n.start - lo) / BAR * 16), n.pitch)
                        for n in notes
                    )
                )
            )
    return patterns


def _variety(made, word):
    shares = []
    for inst in made.instruments:
        if word == "drums" and not inst.is_drum:
            continue
        if word != "drums" and word not in (inst.name or "").lower():
            continue
        patterns = _bar_patterns(inst, made)
        if len(patterns) >= 8:
            shares.append(len(set(patterns)) / len(patterns))
    return min(shares)


def test_layer_bars_do_not_repeat_like_a_loop():
    for style, seed in (("Phonk", 3), ("Pop", 5)):
        made = _layer(style, seed)
        assert _variety(made, "drums") >= 0.35, style
        assert _variety(made, "chords") >= 0.25, style
        assert _variety(made, "pad") >= 0.15, style


def test_a_long_section_gets_different_phrases():
    made = _layer(style="Pop", seed=5)
    long_sections = [s for s in made.sections if s.bars >= 8]
    assert long_sections
    bass = next(i for i in made.instruments if "bass" in (i.name or "").lower())
    section = long_sections[0]
    first = section.start_bar * BAR + made.grid_offset
    phrase = [
        tuple(
            sorted(
                round((n.start - first - k * 4 * BAR) / BAR * 16)
                for n in bass.notes
                if first + k * 4 * BAR <= n.start < first + (k + 1) * 4 * BAR
            )
        )
        for k in range(2)
    ]
    assert phrase[0] and phrase[1] and phrase[0] != phrase[1]


def test_a_varied_phrase_revoices_chords_and_adds_a_bass_pickup():
    chord = pretty_midi.Instrument(0)
    chord.notes = [pretty_midi.Note(80, p, 0.0, 2.0) for p in (57, 60, 64)]
    vary_phrase(chord, "chords", 1, 8.0, BAR)
    assert sorted(n.pitch for n in chord.notes) == [60, 64, 69]
    vary_phrase(chord, "pad", 2, 8.0, BAR)
    assert sorted(n.pitch for n in chord.notes) == [57, 60, 64]
    bass = pretty_midi.Instrument(33)
    bass.notes = [pretty_midi.Note(90, 33, 6.0, 8.0)]
    vary_phrase(bass, "bass", 0, 8.0, BAR)
    assert len(bass.notes) == 1
    vary_phrase(bass, "bass", 1, 8.0, BAR)
    pickup = max(bass.notes, key=lambda n: n.start)
    assert pickup.pitch == 45 and pickup.start == 8.0 - BAR / 8
    assert bass.notes[0].end == pickup.start


def test_a_new_song_is_composed_exactly_as_before():
    options = ComposeOptions("C", "Major", bpm=BPM, minutes=1.0, seed=11)
    first, second = compose(options), compose(options)
    signature = [
        (i.program, len(i.notes), sum(n.pitch for n in i.notes))
        for i in first.instruments
    ]
    assert signature == [
        (i.program, len(i.notes), sum(n.pitch for n in i.notes))
        for i in second.instruments
    ]
    assert all(len(i.notes) for i in first.instruments)
