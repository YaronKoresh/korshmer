import random

import pytest

from korshmer.midi_generation import ComposeOptions, compose, genre_feel
from korshmer.midi_generation.composer import ROLE_KEYS, make_drum_pattern
from korshmer.midi_generation.engine import SONG_STYLES

BPM = 120.0
BEAT = 60.0 / BPM


def _song(style, seed=3, **kw):
    options = {
        "root": "A",
        "scale": "Minor",
        "bpm": BPM,
        "minutes": 1.0,
        "style": style,
        "seed": seed,
        "roles": ROLE_KEYS,
        "texture_batches": 0,
    }
    options.update(kw)
    return compose(ComposeOptions(**options))


def _drums(midi):
    return next(i for i in midi.instruments if i.is_drum)


def _offbeat_shift(inst):
    shifts = []
    for n in inst.notes:
        pos = (n.start / BEAT) % 1.0
        if 0.45 < pos < 0.8:
            shifts.append(pos)
    return shifts


def test_every_genre_has_a_feel():
    assert set(SONG_STYLES) <= set(genre_feel.FEELS)


def test_edm_kicks_on_every_quarter():
    pattern = make_drum_pattern(random.Random(1), "EDM")
    assert pattern[genre_feel.KICK] == [0, 4, 8, 12]


def test_jazz_uses_ride_and_no_closed_hat():
    pattern = make_drum_pattern(random.Random(1), "Jazz")
    assert genre_feel.RIDE in pattern
    assert genre_feel.HAT not in pattern


def test_lofi_uses_side_stick():
    pitches = {
        p
        for seed in range(10)
        for p in make_drum_pattern(random.Random(seed), "Lo-Fi")
    }
    assert genre_feel.SIDE_STICK in pitches or genre_feel.SNARE in pitches


def test_groove_pools_differ_between_genres():
    sets = {
        g: repr([sorted(g.items()) for g in feel.grooves])
        for g, feel in genre_feel.FEELS.items()
    }
    assert len(set(sets.values())) == len(sets)


@pytest.mark.parametrize("style", SONG_STYLES)
def test_progressions_come_from_genre_pool(style):
    pool = {tuple(p) for p in genre_feel.feel_for(style).progressions}
    chords = genre_feel.progression_for(random.Random(2), style, "verse")
    assert tuple(chords) in pool


def test_bridge_progression_is_rotated_pool_member():
    pool = [tuple(p) for p in genre_feel.feel_for("Pop").progressions]
    chords = genre_feel.progression_for(random.Random(2), "Pop", "bridge")
    assert any(sorted(chords) == sorted(p) for p in pool)


def test_hiphop_and_jazz_swing_but_pop_is_straight():
    pop = _offbeat_shift(_drums(_song("Pop")))
    hip = _offbeat_shift(_drums(_song("Hip-Hop")))
    jazz = _offbeat_shift(_drums(_song("Jazz")))
    assert not any(0.55 < p < 0.75 for p in pop)
    assert any(0.55 < p < 0.75 for p in hip)
    assert any(0.7 < p < 0.8 for p in jazz)


def test_apply_swing_shifts_offbeat_only():
    import pretty_midi

    inst = pretty_midi.Instrument(program=0)
    inst.notes = [
        pretty_midi.Note(80, 60, 0.0, 0.2),
        pretty_midi.Note(80, 60, BEAT / 2, BEAT / 2 + 0.2),
    ]
    genre_feel.apply_swing([inst], BPM, 1.0)
    assert inst.notes[0].start == 0.0
    assert inst.notes[1].start == pytest.approx(BEAT * (0.5 + 1 / 3))


def test_no_swing_is_noop():
    import pretty_midi

    inst = pretty_midi.Instrument(program=0)
    inst.notes = [pretty_midi.Note(80, 60, BEAT / 2, BEAT)]
    genre_feel.apply_swing([inst], BPM, 0.0)
    assert inst.notes[0].start == BEAT / 2


def test_rock_chords_are_power_chords():
    chords = next(i for i in _song("Rock").instruments if i.name == "Chords")
    starts = {}
    for n in chords.notes:
        starts.setdefault(round(n.start, 3), []).append(n.pitch)
    assert all(len(v) == 3 for v in starts.values())
    assert all(sorted(v)[2] - sorted(v)[0] == 12 for v in starts.values())


@pytest.mark.parametrize("style", SONG_STYLES)
def test_feel_is_deterministic(style):
    a = _song(style, seed=9)
    b = _song(style, seed=9)
    sig = lambda m: [
        (i.program, round(n.start, 4), n.pitch)
        for i in m.instruments
        for n in i.notes
    ]
    assert sig(a) == sig(b)


def test_genres_produce_different_drum_grooves():
    kicks = {
        style: tuple(
            round(n.start, 2)
            for n in _drums(_song(style)).notes
            if n.pitch == genre_feel.KICK
        )[:12]
        for style in ("EDM", "Jazz", "Hip-Hop")
    }
    assert len(set(kicks.values())) == 3
