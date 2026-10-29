import functools
import itertools
import random
import statistics
from collections import Counter

import numpy as np
import pytest

from korshmer.midi_generation import ComposeOptions, compose, genre_feel
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 120.0
BEAT = 60.0 / BPM
BAR = BEAT * 4
SEEDS = (1, 2, 3, 4)
KICK, RIDE, PEDAL_HAT, SIDE_STICK, SNARE, CLAP = 36, 51, 44, 37, 38, 39
CRASH = 49
TOMS = (45, 47, 50)
HATS = (42, 44, 46, 51, 54, 70, 49, 57)
BACKBEAT = (37, 38, 39, 40)
CHORD_WINDOW = 0.1
SUSTAINED = ("Hip-Hop", "Lo-Fi", "Jazz")
PUNCHY = ("Pop", "Rock", "EDM")


@functools.cache
def _part(style, role, seed=1, scale="Minor"):
    return compose(
        ComposeOptions(
            root="A" if scale == "Minor" else "C",
            scale=scale,
            bpm=BPM,
            minutes=2.0,
            style=style,
            seed=seed,
            roles=(role,),
            texture_batches=0,
            structure="Verse / Chorus",
        )
    )


def _notes(style, role, seed=1, scale="Minor"):
    comp = _part(style, role, seed, scale)
    return sorted(
        (n for inst in comp.instruments for n in inst.notes),
        key=lambda n: n.start,
    )


def _length(notes):
    return statistics.mean((n.end - n.start) / BEAT for n in notes)


def _on_beat(note):
    position = (note.start / BEAT) % 1.0
    return min(position, 1 - position) < 0.05


def _chords(style, seed):
    groups = []
    for n in _notes(style, "chords", seed):
        if groups and n.start - groups[-1][0].start < CHORD_WINDOW:
            groups[-1].append(n)
        else:
            groups.append([n])
    return [sorted(n.pitch for n in group) for group in groups]


def _four_on_the_floor(style, seed):
    comp = _part(style, "drums", seed)
    kicks = [n.start for n in _notes(style, "drums", seed) if n.pitch == KICK]
    full = 0
    for bar in range(comp.total_bars):
        beats = {
            round((k - bar * BAR) / BEAT)
            for k in kicks
            if bar * BAR <= k < (bar + 1) * BAR
            and abs((k - bar * BAR) / BEAT - round((k - bar * BAR) / BEAT))
            < 0.05
        }
        full += {0, 1, 2, 3} <= beats
    return full / comp.total_bars


def _count(style, seed, pitches):
    return sum(1 for n in _notes(style, "drums", seed) if n.pitch in pitches)


def _quiet(style, seed):
    return sum(1 for n in _notes(style, "drums", seed) if n.velocity < 50)


@pytest.mark.parametrize("style", SUSTAINED)
def test_sustained_styles_let_the_bass_ring(style):
    lengths = [_length(_notes(style, "bass", s)) for s in SEEDS]
    assert statistics.mean(lengths) > 0.9


@pytest.mark.parametrize("style", PUNCHY)
def test_punchy_styles_keep_the_bass_short(style):
    lengths = [_length(_notes(style, "bass", s)) for s in SEEDS]
    assert statistics.mean(lengths) < 0.55


@pytest.mark.parametrize("seed", SEEDS)
def test_edm_bass_never_lands_on_a_beat(seed):
    assert not [n for n in _notes("EDM", "bass", seed) if _on_beat(n)]


@pytest.mark.parametrize("seed", SEEDS)
def test_rock_bass_drives_in_eighths(seed):
    comp = _part("Rock", "bass", seed)
    assert len(_notes("Rock", "bass", seed)) / comp.total_bars >= 5


@pytest.mark.parametrize("style", ["Hip-Hop", "Lo-Fi"])
@pytest.mark.parametrize("seed", SEEDS)
def test_hiphop_and_lofi_bass_is_sparse(style, seed):
    comp = _part(style, "bass", seed)
    assert len(_notes(style, "bass", seed)) / comp.total_bars <= 4


def test_jazz_bass_walks_on_the_beat_while_pop_syncopates():
    def off_beat(style, seed):
        notes = _notes(style, "bass", seed)
        return sum(1 for n in notes if not _on_beat(n)) / len(notes)

    assert statistics.mean(off_beat("Jazz", s) for s in SEEDS) <= 0.2
    assert statistics.mean(off_beat("Pop", s) for s in SEEDS) >= 0.4


@pytest.mark.parametrize("style", SONG_STYLES)
def test_bass_stays_in_the_low_register(style):
    for seed in SEEDS:
        pitches = [n.pitch for n in _notes(style, "bass", seed)]
        assert 30 <= min(pitches)
        assert max(pitches) <= 60


def test_edm_chords_are_stabs_and_lofi_chords_are_pads():
    def length(style):
        return statistics.mean(
            _length(_notes(style, "chords", s)) for s in SEEDS
        )

    assert length("EDM") < 0.8
    assert length("Lo-Fi") > 1.8
    assert length("Hip-Hop") > 1.8
    assert 0.8 < length("Pop") < 1.8


@pytest.mark.parametrize("seed", SEEDS)
def test_rock_chords_are_low_power_chords(seed):
    chords = _chords("Rock", seed)
    assert chords
    for low, fifth, top in chords:
        assert fifth - low == 7
        assert top - low == 12
    assert max(c[-1] for c in chords) < 72
    pop = _chords("Pop", seed)
    assert statistics.mean(c[0] for c in pop) > statistics.mean(
        c[0] for c in chords
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_edm_chords_are_stacked_with_the_root_doubled(seed):
    for chord in _chords("EDM", seed):
        assert len(chord) == 4
        assert chord[-1] - chord[0] == 12


@pytest.mark.parametrize("style", ["Hip-Hop", "Lo-Fi"])
@pytest.mark.parametrize("seed", SEEDS)
def test_hiphop_and_lofi_chords_are_spread_wide(style, seed):
    for chord in _chords(style, seed):
        assert 3 <= len(chord) <= 4
        assert chord[-1] - chord[0] >= 14


@pytest.mark.parametrize("seed", SEEDS)
def test_jazz_chords_are_three_note_shells(seed):
    chords = _chords("Jazz", seed)
    assert {len(c) for c in chords} == {3}
    assert min(c[0] for c in chords) >= 47


def test_only_edm_plays_four_on_the_floor():
    for seed in SEEDS:
        assert _four_on_the_floor("EDM", seed) > 0.9
        for style in SONG_STYLES:
            if style != "EDM":
                assert _four_on_the_floor(style, seed) < 0.2


@pytest.mark.parametrize("seed", SEEDS)
def test_jazz_keeps_time_on_the_ride_and_others_do_not(seed):
    total = len(_notes("Jazz", "drums", seed))
    assert _count("Jazz", seed, (RIDE,)) > 0.25 * total
    for style in ("Pop", "EDM", "Hip-Hop", "Lo-Fi"):
        assert _count(style, seed, (RIDE,)) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_lofi_backbeat_is_a_side_stick_not_a_snare(seed):
    assert _count("Lo-Fi", seed, (SIDE_STICK,)) > 20
    assert _count("Lo-Fi", seed, (SNARE, CLAP)) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_jazz_closes_the_hat_with_the_foot(seed):
    assert _count("Jazz", seed, (PEDAL_HAT,)) > 20
    assert _count("Jazz", seed, (42,)) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_only_rock_fills_on_the_toms(seed):
    assert _count("Rock", seed, TOMS) >= 15
    for style in SONG_STYLES:
        if style != "Rock":
            assert _count(style, seed, TOMS) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_ghost_notes_belong_to_hiphop_lofi_and_jazz(seed):
    for style in SUSTAINED:
        assert _quiet(style, seed) >= 20
    for style in PUNCHY:
        assert _quiet(style, seed) == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_crash_cymbals_punctuate_pop_rock_and_edm(seed):
    for style in PUNCHY:
        assert _count(style, seed, (CRASH,)) >= 3
    for style in SUSTAINED:
        assert _count(style, seed, (CRASH,)) <= 2


def _slot_ratio(style, seed):
    comp = _part(style, "melody", seed)
    section = next(s for s in comp.sections if s.name == "Chorus")
    first = section.start_bar * BAR
    notes = [
        n
        for n in _notes(style, "melody", seed)
        if first <= n.start < first + BAR - 1e-6
    ]
    planned = [length * BAR / 16 for _, length, _ in comp.hook.notes]
    return statistics.mean(
        (n.end - n.start) / p for n, p in zip(notes, planned, strict=True)
    )


def test_edm_lead_is_staccato_and_lofi_lead_is_legato():
    edm = statistics.mean(_slot_ratio("EDM", s) for s in SEEDS)
    lofi = statistics.mean(_slot_ratio("Lo-Fi", s) for s in SEEDS)
    assert edm < 0.65
    assert lofi > 0.85


@pytest.mark.parametrize("minor", [False, True])
def test_every_style_has_its_own_progressions(minor):
    pools = {
        style: {
            tuple(p)
            for p in (feel.minor_progressions if minor else feel.progressions)
        }
        for style, feel in genre_feel.FEELS.items()
    }
    assert all(len(pool) >= 3 for pool in pools.values())
    for a, b in itertools.combinations(pools, 2):
        assert pools[a] != pools[b]
        assert len(pools[a] & pools[b]) <= 1


@pytest.mark.parametrize("style", SONG_STYLES)
def test_minor_keys_draw_from_the_minor_pool(style):
    feel = genre_feel.feel_for(style)
    minor = {tuple(p) for p in feel.minor_progressions}
    major = {tuple(p) for p in feel.progressions}
    for seed in range(20):
        chords = genre_feel.progression_for(
            random.Random(seed), style, "verse", minor=True
        )
        assert tuple(chords) in minor
        chords = genre_feel.progression_for(random.Random(seed), style, "verse")
        assert tuple(chords) in major


@pytest.mark.parametrize("style", SONG_STYLES)
def test_progressions_follow_the_scale_mode(style):
    comp = compose(
        ComposeOptions(
            root="A",
            scale="Minor",
            bpm=BPM,
            minutes=2.0,
            style=style,
            seed=2,
            roles=("chords",),
            texture_batches=0,
            structure="Verse / Chorus",
        )
    )
    minor = {tuple(p) for p in genre_feel.feel_for(style).minor_progressions}
    verse = next(s for s in comp.sections if s.name == "Verse")
    assert tuple(verse.progression) in minor


def _features(style, seed, scale):
    drums = _notes(style, "drums", seed, scale)
    bass = _notes(style, "bass", seed, scale)
    chords = _notes(style, "chords", seed, scale)
    melody = _notes(style, "melody", seed, scale)
    bars = _part(style, "drums", seed, scale).total_bars
    starts = Counter(round(n.start, 2) for n in chords)
    moves = [
        b.pitch - a.pitch for a, b in zip(melody, melody[1:], strict=False)
    ]
    covered = sum(n.end - n.start for n in melody)
    return [
        len(drums) / bars,
        sum(1 for n in drums if n.pitch == KICK) / bars,
        sum(1 for n in drums if n.pitch in BACKBEAT) / bars,
        sum(1 for n in drums if n.pitch in HATS) / bars,
        len({n.pitch for n in drums}),
        statistics.mean(n.velocity for n in drums),
        len(bass) / bars,
        _length(bass),
        statistics.mean(n.pitch for n in bass),
        sum(1 for n in bass if not _on_beat(n)) / len(bass),
        len(starts) / bars,
        _length(chords),
        statistics.mean(starts.values()),
        statistics.mean(n.pitch for n in chords),
        len(melody) / bars,
        1 - covered / (bars * BAR),
        _length(melody),
        statistics.mean(n.pitch for n in melody),
        statistics.pstdev(n.pitch for n in melody),
        sum(1 for m in moves if m == 0) / len(moves),
        sum(1 for m in moves if abs(m) >= 5) / len(moves),
    ]


def test_a_nearest_centroid_tells_the_styles_apart():
    rows = [
        (style, _features(style, seed, scale))
        for style in SONG_STYLES
        for seed in range(6)
        for scale in ("Minor", "Major")
    ]
    names = [style for style, _ in rows]
    matrix = np.array([vector for _, vector in rows])
    z = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-9)
    hits = Counter()
    for i, style in enumerate(names):
        centroids = {
            s: z[[j for j, n in enumerate(names) if n == s and j != i]].mean(0)
            for s in SONG_STYLES
        }
        guess = min(
            centroids, key=lambda s: np.linalg.norm(z[i] - centroids[s])
        )
        hits[style] += guess == style
    per_style = len(rows) // len(SONG_STYLES)
    assert sum(hits.values()) / len(rows) >= 0.95
    assert min(hits.values()) / per_style >= 0.8
