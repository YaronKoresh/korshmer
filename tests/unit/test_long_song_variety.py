from collections import defaultdict

import pretty_midi
import pytest

from korshmer.midi_generation import engine
from korshmer.midi_generation.composer import (
    ACT_BARS,
    ComposeOptions,
    compose,
)

BPM = 120.0
BAR = 240.0 / BPM
NO_EXTRAS = ("drums", "bass", "chords", "pad", "melody")


def _song(minutes=10, style="Pop", seed=1, roles=NO_EXTRAS, **kw):
    return compose(
        ComposeOptions(
            root="C",
            scale="Major",
            bpm=BPM,
            minutes=minutes,
            style=style,
            seed=seed,
            roles=roles,
            texture_batches=0,
            **kw,
        )
    )


def _by_role(comp):
    merged = {}
    for inst in comp.instruments:
        name = inst.name.rstrip("0123456789 ")
        voice = merged.setdefault(
            name, pretty_midi.Instrument(inst.program, inst.is_drum, name)
        )
        voice.notes.extend(inst.notes)
    return merged


def _phrase_counts(inst, total_bars):
    bars = defaultdict(list)
    for note in inst.notes:
        bars[int(note.start // BAR)].append(
            (
                round(note.start / BAR % 1, 3),
                note.pitch,
                round((note.end - note.start) / BAR, 3),
            )
        )
    sigs = [tuple(sorted(bars[b])) for b in range(total_bars)]
    phrases = [tuple(sigs[i : i + 4]) for i in range(0, total_bars - 3, 4)]
    phrases = [p for p in phrases if any(p)]
    return len(phrases), len(set(phrases))


@pytest.mark.parametrize("style", ["Pop", "Rock", "Jazz", "EDM"])
def test_ten_minute_song_keeps_changing(style):
    comp = _song(style=style)
    counts = {
        name: _phrase_counts(inst, comp.total_bars)
        for name, inst in _by_role(comp).items()
    }
    total, distinct = counts["Drums"]
    assert distinct >= 20
    total, distinct = counts["Melody"]
    assert distinct / total >= 0.6
    total, distinct = counts["Chords"]
    assert distinct / total >= 0.25


def test_ten_minute_song_uses_many_chord_progressions():
    comp = _song()
    assert len({s.progression for s in comp.sections}) >= 8


def test_short_song_stays_in_one_act():
    comp = _song(minutes=1)
    assert comp.total_bars <= ACT_BARS
    assert len({s.progression for s in comp.sections}) <= 5


@pytest.mark.parametrize("scale", ["Minor", "Dorian", "Pentatonic Major"])
def test_ten_minute_song_stays_in_the_chosen_scale(scale):
    from korshmer.constants import NOTE_NAMES, SCALES
    from korshmer.midi_generation.composer import harmony_scale

    comp = compose(
        ComposeOptions(
            root="F#",
            scale=scale,
            bpm=BPM,
            minutes=10,
            seed=4,
            roles=NO_EXTRAS,
            texture_batches=0,
        )
    )
    allowed = {(NOTE_NAMES.index("F#") + i) % 12 for i in SCALES[scale]}
    allowed |= {
        (NOTE_NAMES.index("F#") + i) % 12 for i in SCALES[harmony_scale(scale)]
    }
    for inst in comp.instruments:
        if not inst.is_drum:
            assert {n.pitch % 12 for n in inst.notes} <= allowed


@pytest.mark.parametrize("style", ["Pop", "Jazz", "EDM"])
def test_pitches_stay_in_midi_range_and_seed_is_reproducible(style):
    first = _song(style=style, seed=7)
    second = _song(style=style, seed=7)
    for a, b in zip(first.instruments, second.instruments, strict=True):
        assert [(n.pitch, n.start) for n in a.notes] == [
            (n.pitch, n.start) for n in b.notes
        ]
        assert all(0 <= n.pitch <= 127 for n in a.notes)


def test_drum_variation_off_by_default():
    ctx = engine.HarmonicContext(root=0, scale="Major", bpm=BPM)
    plain = engine.generate_drums(ctx, bars=8, seed=3)
    again = engine.generate_drums(ctx, bars=8, seed=3, variation=0.0)
    assert [(n.pitch, n.start) for n in plain.notes] == [
        (n.pitch, n.start) for n in again.notes
    ]
    varied = engine.generate_drums(ctx, bars=8, seed=3, variation=1.0)
    assert len(varied.notes) > len(plain.notes)


@pytest.mark.parametrize(
    "structure", ["Verse / Chorus", "Build & Drop", "Journey (A-B-A)"]
)
@pytest.mark.parametrize("seed", range(3))
def test_long_structures_get_a_break_in_every_stretch(structure, seed):
    import random

    from korshmer.midi_generation.composer import plan_sections

    total = 300
    sections = plan_sections(random.Random(seed), total, structure)
    assert sum(s.bars for s in sections) == total
    assert sections[0].name == "Intro"
    assert sections[-1].name == "Outro"
    breaks = [s for s in sections if s.name == "Break"]
    assert len(breaks) >= total // (ACT_BARS * 2)
    longest_run = 0
    run = 0
    for s in sections:
        run = 0 if s.name == "Break" else run + s.bars
        longest_run = max(longest_run, run)
    assert longest_run <= ACT_BARS * 2


@pytest.mark.parametrize("total", [8, 48, 64, 120, 300])
def test_long_loop_is_cut_into_acts(total):
    import random

    from korshmer.midi_generation.composer import plan_sections

    sections = plan_sections(random.Random(1), total, "Loop")
    assert sum(s.bars for s in sections) == total
    assert all(s.name == "Loop" and s.bars <= ACT_BARS for s in sections)
    starts = [s.start_bar for s in sections]
    assert starts == sorted(starts)
    assert len(sections) == -(-total // ACT_BARS)


def test_ten_minute_loop_song_keeps_changing():
    comp = _song(structure="Loop")
    total, distinct = _phrase_counts(
        next(i for i in comp.instruments if i.name == "Melody"),
        comp.total_bars,
    )
    assert distinct / total >= 0.5


ALL_VOICES = (*NO_EXTRAS, "arp", "pluck")


def test_pad_arp_and_pluck_change_over_a_long_song():
    comp = _song(roles=ALL_VOICES, seed=1)
    by_name = _by_role(comp)
    total, distinct = _phrase_counts(by_name["Pad"], comp.total_bars)
    assert distinct / total >= 0.3
    for name in ("Arp", "Pluck"):
        lengths = {
            round((n.end - n.start) / BAR, 3) for n in by_name[name].notes
        }
        assert len(lengths) >= 2, name
        total, distinct = _phrase_counts(by_name[name], comp.total_bars)
        assert distinct / total >= 0.2, name


def test_pad_and_arp_options_default_to_previous_behaviour():
    ctx = engine.HarmonicContext(root=0, scale="Major", bpm=BPM)
    pad = engine.generate_pad(ctx, seed=1, bars=4)
    assert min(n.pitch for n in pad.notes) >= 48
    explicit = engine.generate_pad(ctx, seed=1, bars=4, octave=3)
    assert [(n.pitch, n.start) for n in pad.notes] == [
        (n.pitch, n.start) for n in explicit.notes
    ]
    up = engine.generate_arp(ctx, "1/8", seed=1, bars=1, shape="up")
    down = engine.generate_arp(ctx, "1/8", seed=1, bars=1, shape="down")
    assert [n.pitch for n in up.notes] == [n.pitch for n in down.notes][::-1]
