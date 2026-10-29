import itertools

import pytest

from korshmer.constants import SCALES
from korshmer.midi_generation.engine import (
    MELODY_MOODS,
    HarmonicContext,
    build_song,
    generate_melody,
)

SEEDS = range(12)


def _ctx(scale="Major"):
    return HarmonicContext(root=0, scale=scale, bpm=120.0)


def _bar_rhythms(inst, ctx, bars):
    out = []
    for b in range(bars):
        out.append(
            tuple(
                round((n.start - b * ctx.bar) / ctx.bar, 3)
                for n in inst.notes
                if b * ctx.bar - 1e-6 <= n.start < (b + 1) * ctx.bar - 1e-6
            )
        )
    return out


@pytest.mark.parametrize("mood", MELODY_MOODS)
def test_same_seed_gives_same_melody(mood):
    ctx = _ctx()
    a = generate_melody(ctx, bars=8, mood=mood, seed=7)
    b = generate_melody(ctx, bars=8, mood=mood, seed=7)
    assert [(n.pitch, n.start, n.end) for n in a.notes] == [
        (n.pitch, n.start, n.end) for n in b.notes
    ]


@pytest.mark.parametrize("mood", MELODY_MOODS)
def test_melodies_differ_between_seeds(mood):
    ctx = _ctx()
    pitch_lines = {
        tuple(n.pitch for n in generate_melody(ctx, 8, mood, s).notes)
        for s in SEEDS
    }
    assert len(pitch_lines) == len(list(SEEDS))


@pytest.mark.parametrize("mood", MELODY_MOODS)
def test_rhythm_has_several_patterns_within_eight_bars(mood):
    ctx = _ctx()
    counts = [
        len(set(_bar_rhythms(generate_melody(ctx, 8, mood, s), ctx, 8)))
        for s in SEEDS
    ]
    assert min(counts) >= 2
    assert sum(counts) / len(counts) >= 3


@pytest.mark.parametrize("mood", MELODY_MOODS)
def test_uses_varied_pitches_and_durations(mood):
    ctx = _ctx()
    duration_counts = []
    for s in SEEDS:
        inst = generate_melody(ctx, 8, mood, s)
        assert len({n.pitch for n in inst.notes}) >= 5
        duration_counts.append(
            len({round(n.end - n.start, 2) for n in inst.notes})
        )
    assert min(duration_counts) >= 2
    assert sum(duration_counts) / len(duration_counts) >= 3


def test_phrases_repeat_a_motif_with_variation():
    ctx = _ctx()
    for s in SEEDS:
        inst = generate_melody(ctx, 8, "Uplifting", s)
        rhythms = _bar_rhythms(inst, ctx, 8)
        assert any(
            rhythms[i] == rhythms[j]
            for i, j in itertools.combinations(range(8), 2)
        )


@pytest.mark.parametrize("scale", ["Major", "Minor"])
def test_notes_stay_in_scale_and_bounds(scale):
    ctx = _ctx(scale)
    allowed = {i % 12 for i in SCALES[scale]}
    for s in SEEDS:
        inst = generate_melody(ctx, 8, "Dreamy", s)
        assert inst.notes
        for n in inst.notes:
            assert n.pitch % 12 in allowed
            assert 55 <= n.pitch <= 96
            assert 0 <= n.start < n.end <= 8 * ctx.bar + 1e-6


def test_phrases_end_on_a_chord_tone():
    ctx = _ctx()
    for s in SEEDS:
        inst = generate_melody(ctx, 4, "Uplifting", s)
        last = max(inst.notes, key=lambda n: n.start)
        degree_pcs = {(ctx.root + i) % 12 for i in ctx.intervals}
        assert last.pitch % 12 in degree_pcs
        assert last.end - last.start >= ctx.bar * 0.4


def test_start_bar_offsets_the_melody():
    ctx = _ctx()
    inst = generate_melody(ctx, 4, "Uplifting", 3, start_bar=8)
    assert min(n.start for n in inst.notes) >= 8 * ctx.bar - 1e-6
    assert max(n.end for n in inst.notes) <= 12 * ctx.bar + 1e-6


def test_build_song_melody_changes_between_sections():
    pm = build_song("C", "Major", seed=5, parts=["melody"])
    melody = pm.instruments[0]
    bar = 60.0 / 120.0 * 4
    sections = [
        tuple(
            n.pitch
            for n in melody.notes
            if s * 8 * bar <= n.start < (s + 1) * 8 * bar
        )
        for s in range(8)
    ]
    assert len(set(sections)) >= 6
