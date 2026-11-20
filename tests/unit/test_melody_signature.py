import functools
import statistics
from collections import Counter

import pytest

from korshmer.midi_generation import ComposeOptions, compose
from korshmer.midi_generation.composer import NOTE_NAMES
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 120.0
BAR = 60.0 / BPM * 4
EPS = 1e-6
SEEDS = (1, 2, 3, 4)
STRAIGHT = ("Pop", "EDM", "Rock")
MANY_SEEDS = range(1, 21)
KEY_SEEDS = range(1, 9)
SPREAD_ROOTS = ("C", "D", "E", "F#", "G#", "A#")
CHORUS_REGISTER = {"Phonk": (54, 81), "other": (60, 89)}
MELODY_RANGE = {"Phonk": (45, 96), "other": (48, 96)}


def _song(style, seed=3, root="A", scale="Minor", minutes=3.0):
    return compose(
        ComposeOptions(
            root=root,
            scale=scale,
            bpm=BPM,
            minutes=minutes,
            style=style,
            seed=seed,
            roles=("melody",),
            texture_batches=0,
            structure="Verse / Chorus",
        )
    )


def _notes(comp):
    return sorted(
        (n for inst in comp.instruments for n in inst.notes),
        key=lambda n: n.start,
    )


def _bars(comp):
    grouped = {}
    for n in _notes(comp):
        grouped.setdefault(int(n.start / BAR + EPS), []).append(n)
    return grouped


def _step(note, bar):
    return round((note.start - bar * BAR) / BAR * 16)


def _on_strong_beat(note, bar):
    return any(
        abs(note.start - (bar + step / 16) * BAR) < EPS for step in (0, 8)
    )


def _signature(grouped, bar):
    return tuple((_step(n, bar), n.pitch) for n in grouped.get(bar, ()))


def _rhythm(grouped, bar):
    return tuple(_step(n, bar) for n in grouped.get(bar, ()))


def _pitches(comp, name, act=None):
    grouped = _bars(comp)
    return [
        n.pitch
        for s in comp.sections
        if s.name == name and (act is None or s.act_kind == act)
        for b in range(s.start_bar, s.start_bar + s.bars)
        for n in grouped.get(b, ())
    ]


def _top_rhythm_share(comp):
    grouped = _bars(comp)
    counts = Counter(
        _rhythm(grouped, b) for b in grouped if len(grouped[b]) >= 2
    )
    return max(counts.values()) / sum(counts.values())


def _recurrence(comp):
    notes = _notes(comp)
    moves = [
        (round((b.start % BAR) / BAR * 16), b.pitch - a.pitch)
        for a, b in zip(notes, notes[1:], strict=False)
    ]
    grams = Counter(tuple(moves[i : i + 4]) for i in range(len(moves) - 3))
    return sum(c for c in grams.values() if c >= 3) / sum(grams.values())


def _choruses(comp):
    return [
        s
        for s in comp.sections
        if s.name == "Chorus" and s.bars >= 8 and len(s.progression) == 4
    ]


@pytest.mark.parametrize("style", SONG_STYLES)
def test_every_song_carries_its_hook(style):
    comp = _song(style)
    assert comp.hook is not None
    assert comp.hook.notes
    assert comp.hook.gate > 0


@pytest.mark.parametrize("style", SONG_STYLES)
def test_auto_structure_chorus_parts_restate_the_hook(style):
    checked = 0
    for seed in SEEDS:
        comp = compose(
            ComposeOptions(
                root="A",
                scale="Minor",
                bpm=BPM,
                minutes=3.0,
                style=style,
                seed=seed,
                roles=("melody",),
                texture_batches=0,
            )
        )
        grouped = _bars(comp)
        for s in comp.sections:
            if s.name in ("Chorus", "Drop", "B") and s.bars >= 8:
                first = s.start_bar
                assert _signature(grouped, first) == _signature(
                    grouped, first + 4
                )
                assert _signature(grouped, first + 1) == _signature(
                    grouped, first + 5
                )
                checked += 1
    assert checked >= len(SEEDS)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_chorus_restates_the_hook_bar_for_bar(style):
    checked = 0
    for seed in SEEDS:
        comp = _song(style, seed)
        grouped = _bars(comp)
        for s in _choruses(comp):
            first = s.start_bar
            assert grouped.get(first)
            assert _signature(grouped, first) == _signature(grouped, first + 4)
            assert _signature(grouped, first + 1) == _signature(
                grouped, first + 5
            )
            checked += 1
    assert checked >= len(SEEDS)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_same_hook_sits_differently_over_each_chord(style):
    comp = _song(style)
    grouped = _bars(comp)
    for s in _choruses(comp):
        first = s.start_bar
        if s.progression[0] != s.progression[1]:
            assert _signature(grouped, first) != _signature(grouped, first + 1)
            assert _rhythm(grouped, first) == _rhythm(grouped, first + 1)


@pytest.mark.parametrize("style", STRAIGHT)
def test_hook_rhythm_opens_intro_chorus_and_outro(style):
    for seed in SEEDS:
        comp = _song(style, seed)
        grouped = _bars(comp)
        rhythm = tuple(onset for onset, _, _ in comp.hook.notes)
        for name in ("Intro", "Chorus", "Outro"):
            section = next(s for s in comp.sections if s.name == name)
            assert _rhythm(grouped, section.start_bar) == rhythm


@pytest.mark.parametrize("style", SONG_STYLES)
def test_strong_beats_land_on_chord_tones(style):
    checked = 0
    for seed in MANY_SEEDS:
        comp = _song(style, seed)
        for bar, notes in _bars(comp).items():
            for n in notes:
                if _on_strong_beat(n, bar):
                    assert n.pitch % 12 in comp.bar_chords[bar]
                    checked += 1
    assert checked >= 50 * len(MANY_SEEDS)


def _share(values, holds):
    values = list(values)
    return sum(1 for value in values if holds(value)) / len(values)


@functools.cache
def _chorus_and_verse_height(style, seed, root, scale):
    comp = _song(style, seed, root=root, scale=scale)
    return (
        statistics.mean(_pitches(comp, "Chorus")),
        statistics.mean(_pitches(comp, "Verse")),
    )


@pytest.mark.parametrize("style", SONG_STYLES)
def test_melody_repeats_its_own_rhythm_and_phrases(style):
    comps = [_song(style, seed) for seed in MANY_SEEDS]
    shares = [_top_rhythm_share(c) for c in comps]
    recurrence = [_recurrence(c) for c in comps]
    assert _share(shares, lambda found: found >= 0.35) >= 0.8
    assert _share(recurrence, lambda found: found >= 0.3) >= 0.8
    assert statistics.mean(shares) >= 0.42
    assert statistics.mean(recurrence) >= 0.38


@pytest.mark.parametrize("style", SONG_STYLES)
@pytest.mark.parametrize("scale", ["Minor", "Major"])
def test_chorus_usually_sits_above_the_verse_whatever_the_key(style, scale):
    gaps = [
        chorus - verse
        for chorus, verse in (
            _chorus_and_verse_height(style, seed, root, scale)
            for seed in KEY_SEEDS
            for root in SPREAD_ROOTS
        )
    ]
    assert _share(gaps, lambda gap: gap > 0) >= 0.8
    assert statistics.mean(gaps) >= 5


@pytest.mark.parametrize("style", SONG_STYLES)
def test_chorus_register_does_not_drift_with_the_key(style):
    low, high = CHORUS_REGISTER.get(style, CHORUS_REGISTER["other"])
    spreads = []
    heights = []
    for seed in KEY_SEEDS:
        means = [
            _chorus_and_verse_height(style, seed, root, "Minor")[0]
            for root in SPREAD_ROOTS
        ]
        spreads.append(max(means) - min(means))
        heights.extend(means)
    assert _share(heights, lambda mean: low <= mean <= high) >= 0.9
    assert max(spreads) <= 10
    assert statistics.mean(spreads) <= 6.5


@pytest.mark.parametrize("style", SONG_STYLES)
def test_climax_chorus_is_usually_higher_than_the_first_statement(style):
    gaps = []
    for seed in range(1, 13):
        comp = _song(style, seed)
        first = _pitches(comp, "Chorus", "Statement")
        last = _pitches(comp, "Chorus", "Climax")
        if first and last:
            gaps.append(statistics.mean(last) - statistics.mean(first))
    assert len(gaps) >= 10
    assert _share(gaps, lambda gap: gap > 0) >= 0.9
    assert statistics.mean(gaps) >= 5


@pytest.mark.parametrize("style", SONG_STYLES)
@pytest.mark.parametrize("root", ["C", "F#", "A"])
def test_song_ends_on_the_tonic(style, root):
    for seed in SEEDS:
        comp = _song(style, seed, root=root)
        final = comp.sections[-1]
        tonic = NOTE_NAMES.index(final.key.split()[0])
        assert (_notes(comp)[-1].pitch - tonic) % 12 == 0


def test_hooks_differ_between_songs_of_one_style():
    found = {_song("Pop", seed).hook.notes for seed in range(12)}
    assert len(found) >= 9


@pytest.mark.parametrize("style", SONG_STYLES)
def test_melody_notes_stay_inside_a_singable_range(style):
    low, high = MELODY_RANGE.get(style, MELODY_RANGE["other"])
    for root in ("A", "C", "F#"):
        for seed in MANY_SEEDS:
            pitches = [n.pitch for n in _notes(_song(style, seed, root=root))]
            assert low <= min(pitches)
            assert max(pitches) <= high
