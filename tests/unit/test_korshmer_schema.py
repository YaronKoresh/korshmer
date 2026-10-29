import pytest

from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.midi_generation.korshmer_schema import (
    BATCH_BARS,
    DRUM_TRACKS,
    LONG_PROGRAMS,
    LONG_TRACKS,
    MAX_MINUTES,
    MAX_TRACKS,
    MIN_ACTIVE_TRACKS,
    ROUTINE_BARS,
    SHORT_PROGRAMS,
    SHORT_TRACKS,
    TRACKS_PER_ACTIVATION,
    bars_for_minutes,
    build_schema_instruments,
    period_bars,
    plan_schema,
    plan_slots,
    track_index,
    validate_minutes,
    validate_schema_key,
)

BPM = 120.0
BAR = 60.0 / BPM * 4
BEAT = 60.0 / BPM
SLOT = ROUTINE_BARS * BAR
ALL_ROOTS = list(NOTE_NAMES)
SCALES_UNDER_TEST = ["Pentatonic Major", "Pentatonic Minor"]


def _build(scale="Pentatonic Minor", root="A", existing=0, seed=1, minutes=3.0):
    return build_schema_instruments(root, scale, BPM, existing, seed, minutes)


def _routines(inst):
    groups = {}
    for note in inst.notes:
        groups.setdefault(int(note.start / SLOT + 1e-6), []).append(note)
    return {
        k: sorted(v, key=lambda n: n.start) for k, v in sorted(groups.items())
    }


def _total_slots(minutes=3.0):
    return bars_for_minutes(minutes, BPM) // ROUTINE_BARS


def _bar_coverage(instruments, minutes=3.0):
    cover = [0] * bars_for_minutes(minutes, BPM)
    for inst in instruments:
        for bar in {int(n.start / BAR + 1e-6) for n in inst.notes}:
            cover[bar] += 1
    return cover


def _index_of(inst):
    return int(inst.name.split()[1])


@pytest.mark.parametrize(
    ("zero_based", "one_based"),
    [(0, 1), (1, 2), (8, 9), (9, 10), (10, 1), (11, 2), (25, 6)],
)
def test_track_index_is_one_based_and_wraps_after_ten(zero_based, one_based):
    assert track_index(zero_based) == one_based


@pytest.mark.parametrize(
    ("zero_based", "modulus", "index"), [(12, 20, 13), (20, 20, 1)]
)
def test_track_index_uses_the_given_modulus(zero_based, modulus, index):
    assert track_index(zero_based, modulus) == index


@pytest.mark.parametrize(
    ("index", "bars"), [(1, 8), (2, 12), (3, 16), (10, 44)]
)
def test_period_is_index_plus_one_routines(index, bars):
    assert period_bars(index) == bars


@pytest.mark.parametrize("scale", SCALES_UNDER_TEST)
def test_pentatonic_scales_accepted(scale):
    assert validate_schema_key("C", scale) == (0, scale)


@pytest.mark.parametrize("root", ["Auto", "", "H", "c"])
def test_undefined_root_rejected(root):
    with pytest.raises(ValueError, match="root"):
        validate_schema_key(root, "Pentatonic Major")


@pytest.mark.parametrize("scale", ["Auto (Detect)", "Major", "Minor", "Dorian"])
def test_other_scales_rejected(scale):
    with pytest.raises(ValueError, match="Pentatonic"):
        validate_schema_key("C", scale)


def test_tracks_come_in_batches_split_3_5_7():
    instruments = _build()
    batches = len(instruments) // TRACKS_PER_ACTIVATION
    assert len(instruments) == batches * TRACKS_PER_ACTIVATION
    drums = [i for i in instruments if i.is_drum]
    longs = [i for i in instruments if i.program in LONG_PROGRAMS.values()]
    shorts = [
        i
        for i in instruments
        if not i.is_drum and i.program in SHORT_PROGRAMS.values()
    ]
    assert len(drums) == batches * DRUM_TRACKS
    assert len(longs) == batches * LONG_TRACKS
    assert len(shorts) == batches * SHORT_TRACKS


def test_default_song_has_more_than_one_batch_of_tracks():
    assert len(_build()) >= 2 * TRACKS_PER_ACTIVATION


def test_instruments_are_distinct_within_each_batch():
    instruments = _build()
    for start in range(0, len(instruments), TRACKS_PER_ACTIVATION):
        batch = instruments[start : start + TRACKS_PER_ACTIVATION]
        assert len({i.program for i in batch if not i.is_drum}) == 12


@pytest.mark.parametrize("seed", range(6))
def test_each_routine_uses_four_different_beats_one_per_bar(seed):
    for inst in _build(seed=seed):
        for slot, notes in _routines(inst).items():
            assert len(notes) == 4
            bars = [int(n.start / BAR + 1e-6) - slot * 4 for n in notes]
            assert bars == [0, 1, 2, 3]
            beats = [
                round((n.start - (slot * 4 + b) * BAR) / BEAT)
                for b, n in enumerate(notes)
            ]
            assert sorted(beats) == [0, 1, 2, 3]


@pytest.mark.parametrize("scale", SCALES_UNDER_TEST)
@pytest.mark.parametrize("root", ALL_ROOTS)
def test_every_track_uses_four_of_the_five_scale_notes(scale, root):
    root_idx = NOTE_NAMES.index(root)
    allowed = {(root_idx + i) % 12 for i in SCALES[scale]}
    for seed in range(3):
        for inst in _build(scale, root, seed=seed):
            for notes in _routines(inst).values():
                pitches = [n.pitch for n in notes]
                classes = [p % 12 for p in pitches]
                assert len(set(classes)) == 4
                assert set(classes) <= allowed
                assert max(pitches) - min(pitches) < 12


@pytest.mark.parametrize("scale", SCALES_UNDER_TEST)
@pytest.mark.parametrize("root", ["C", "F#", "B"])
def test_each_track_has_one_octave_between_2_and_7(scale, root):
    root_idx = NOTE_NAMES.index(root)
    seen = set()
    for seed in range(15):
        for inst in _build(scale, root, seed=seed):
            octaves = {(n.pitch - root_idx) // 12 - 1 for n in inst.notes}
            assert len(octaves) == 1
            octave = octaves.pop()
            assert 2 <= octave <= 7
            seen.add(octave)
    assert seen >= {2, 3, 4, 5}


@pytest.mark.parametrize("root", ALL_ROOTS)
def test_drum_notes_are_gm_percussion_numbers_in_the_key(root):
    for seed in range(5):
        for inst in _build("Pentatonic Minor", root, seed=seed):
            if inst.is_drum:
                assert all(35 <= n.pitch <= 81 for n in inst.notes)


@pytest.mark.parametrize("root", ALL_ROOTS)
def test_all_notes_stay_in_range(root):
    for seed in range(5):
        for inst in _build("Pentatonic Major", root, seed=seed):
            assert all(0 <= n.pitch <= 127 for n in inst.notes)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("minutes", [0.5, 1, 3, 6, 12, 25])
def test_no_bar_has_fewer_than_two_tracks_playing(minutes, seed):
    cover = _bar_coverage(_build(seed=seed, minutes=minutes), minutes)
    assert min(cover) >= MIN_ACTIVE_TRACKS


@pytest.mark.parametrize("bpm", [60, 90, 140, 200])
def test_quiet_bars_are_filled_at_other_tempos(bpm):
    instruments = build_schema_instruments(
        "D", "Pentatonic Major", bpm, 0, 3, 2
    )
    bar = 240.0 / bpm
    cover = [0] * bars_for_minutes(2, bpm)
    for inst in instruments:
        for b in {int(n.start / bar + 1e-6) for n in inst.notes}:
            cover[b] += 1
    assert min(cover) >= MIN_ACTIVE_TRACKS


def test_routines_do_not_all_start_together():
    instruments = _build()
    at_first_slot = sum(1 for i in instruments if 0 in _routines(i))
    assert at_first_slot < len(instruments)
    assert at_first_slot >= MIN_ACTIVE_TRACKS


def test_frequency_follows_the_index_over_the_song():
    total = _total_slots()
    instruments = _build()
    by_index = {}
    for inst in instruments:
        by_index.setdefault(_index_of(inst), []).append(len(_routines(inst)))
    for index, plays in by_index.items():
        expected = max(1, round(total / (index + 1)))
        assert all(p >= expected for p in plays)
        assert sum(plays) / len(plays) <= expected + 3
    averages = {i: sum(p) / len(p) for i, p in by_index.items()}
    assert averages[1] > averages[5] > averages[10]


def test_slots_are_spread_over_the_whole_song():
    total = 40
    for seed in range(20):
        import random

        slots = sorted(plan_slots(random.Random(seed), total, 3))
        assert len(slots) == 10
        for k, slot in enumerate(slots):
            assert k * total // 10 <= slot < (k + 1) * total // 10


def test_placement_is_not_a_fixed_period():
    gaps = set()
    for seed in range(10):
        for inst in _build(seed=seed):
            slots = list(_routines(inst))
            gaps |= {b - a for a, b in zip(slots, slots[1:], strict=False)}
    assert len(gaps) > 4


def test_different_seeds_place_routines_differently():
    a = [list(_routines(i)) for i in _build(seed=1)]
    b = [list(_routines(i)) for i in _build(seed=2)]
    assert a != b


def test_same_seed_gives_same_schema():
    a = _build(seed=9)
    b = _build(seed=9)
    assert [(i.name, [(n.pitch, n.start) for n in i.notes]) for i in a] == [
        (i.name, [(n.pitch, n.start) for n in i.notes]) for i in b
    ]


def test_each_track_repeats_the_same_routine_notes():
    for inst in _build():
        routines = list(_routines(inst).items())
        first_slot, first = routines[0]
        for slot, notes in routines[1:]:
            assert [n.pitch for n in notes] == [n.pitch for n in first]
            assert [round((n.start - slot * SLOT) % BAR, 6) for n in notes] == [
                round((n.start - first_slot * SLOT) % BAR, 6) for n in first
            ]


def test_existing_tracks_shift_the_indexes():
    instruments = _build(existing=10)
    assert instruments[0].name.startswith("Schema 1 ")
    assert instruments[9].name.startswith("Schema 10 ")
    assert instruments[10].name.startswith("Schema 1 ")


@pytest.mark.parametrize(
    ("minutes", "bpm", "bars"),
    [
        (3.0, 120, 92),
        (1.0, 120, 32),
        (0.5, 120, 16),
        (2.0, 60, 32),
        (10, 120, 300),
    ],
)
def test_bars_follow_minutes_and_round_up_to_whole_routines(minutes, bpm, bars):
    assert bars_for_minutes(minutes, bpm) == bars
    assert bars % 4 == 0


@pytest.mark.parametrize(
    "minutes", [0, -1, 0.1, 121, float("nan"), float("inf"), "x", None]
)
def test_invalid_minutes_rejected(minutes):
    with pytest.raises(ValueError, match="minutes"):
        validate_minutes(minutes)


@pytest.mark.parametrize("minutes", [1, 3, 5, 8, 10])
def test_note_length_stays_inside_the_requested_minutes(minutes):
    instruments = _build(minutes=minutes)
    total = bars_for_minutes(minutes, BPM) * BAR
    assert max(n.end for i in instruments for n in i.notes) <= total + 1e-6


def test_track_count_is_a_multiple_of_15_that_grows_with_minutes():
    counts = [len(_build(minutes=m)) for m in (1, 3, 6, 9, 12)]
    assert all(c % 15 == 0 for c in counts)
    assert counts == sorted(counts)
    assert counts[0] >= 30
    assert counts[-1] > counts[0]


@pytest.mark.parametrize(
    ("bars", "tracks"), [(16, 30), (92, 30), (150, 45), (300, 90)]
)
def test_plan_batches_by_length(bars, tracks):
    assert plan_schema(bars)[0] == tracks


@pytest.mark.parametrize("bars", [92, 150, 220])
def test_plan_keeps_the_default_modulus_for_normal_lengths(bars):
    assert plan_schema(bars)[1] == 10


@pytest.mark.parametrize("bars", [400, 1200, 3600])
def test_very_long_songs_cap_tracks_and_raise_the_modulus(bars):
    tracks, modulus = plan_schema(bars)
    assert tracks == MAX_TRACKS
    assert modulus > 10


def test_modulus_grows_with_length_until_the_track_cap():
    mods = [plan_schema(b)[1] for b in (400, 600, 900, 3600)]
    assert mods == sorted(mods)
    assert mods[-1] == MAX_TRACKS


def test_very_long_request_builds_capped_tracks_with_longer_periods():
    instruments = _build(minutes=60)
    assert len(instruments) == MAX_TRACKS
    indexes = [_index_of(i) for i in instruments]
    assert max(indexes) > 10
    assert max(indexes) <= MAX_TRACKS


def test_long_request_has_fewer_repeats_than_the_default_modulus_would_give():
    total_slots = _total_slots(60)
    plays = [len(_routines(i)) for i in _build(minutes=60)]
    assert sum(plays) / len(plays) < total_slots / 3


def test_max_minutes_builds_the_capped_song():
    instruments = _build(minutes=MAX_MINUTES)
    assert len(instruments) == MAX_TRACKS
    assert min(_bar_coverage(instruments, MAX_MINUTES)) >= MIN_ACTIVE_TRACKS
    assert BATCH_BARS == 44
