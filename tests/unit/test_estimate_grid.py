import itertools
import random
import time
import tracemalloc
import warnings

import numpy as np
import pretty_midi
import pytest

from korshmer.midi_generation import song_match

MAJOR = (0, 2, 4, 5, 7, 9, 11)
PROGRESSION = (0, 5, 3, 4)
TOLERANCE = 0.03


def circular(value, truth, period):
    return abs((value - truth + period / 2) % period - period / 2)


def groove(
    tempo,
    phase,
    bars=24,
    swung=False,
    seed=0,
    kicks=(0, 2),
    snares=(1, 3),
    bass_shift=0,
):
    rng = random.Random(seed)
    beat = 60.0 / tempo
    bar = 4 * beat
    step = beat / 4
    drums = pretty_midi.Instrument(0, is_drum=True, name="Drums")
    bass = pretty_midi.Instrument(33, name="Bass")
    chords = pretty_midi.Instrument(0, name="Other")
    vocal = pretty_midi.Instrument(0, name="Vocals")

    def at(origin, position):
        late = 0.67 if swung and position % 4 == 2 else 0.0
        return origin + (position + late) * step

    def hit(pitch, velocity, start):
        drums.notes.append(
            pretty_midi.Note(velocity, pitch, start, start + 0.05)
        )

    for index in range(bars):
        degree = PROGRESSION[index % len(PROGRESSION)]
        triad = [MAJOR[(degree + s) % 7] for s in (0, 2, 4)]
        origin = phase + index * bar
        for position in kicks:
            hit(36, 100, at(origin, 4 * position))
        for position in snares:
            hit(38, 100, at(origin, 4 * position))
        for position in range(0, 16, 2):
            hit(42, 70, at(origin, position))
        for position in (0, 6, 8, 14):
            if rng.random() < 0.8:
                start = at(origin, position + 4 * bass_shift)
                bass.notes.append(
                    pretty_midi.Note(90, 36 + triad[0], start, start + 2 * step)
                )
        if rng.random() < 0.6:
            for pc in triad:
                chords.notes.append(
                    pretty_midi.Note(70, 60 + pc, origin, origin + 0.9 * bar)
                )
        if rng.random() < 0.5:
            position = rng.choice((0, 2, 4))
            while position < 16:
                length = rng.choice((1, 2, 2, 3, 4, 6))
                start = at(origin, position)
                pitch = 72 + rng.choice(triad + [0, 2, 4, 5, 7, 9])
                vocal.notes.append(
                    pretty_midi.Note(
                        rng.randint(70, 100),
                        pitch,
                        start,
                        start + length * step,
                    )
                )
                position += length + rng.choice((0, 0, 1, 2))
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments = [drums, bass, chords, vocal]
    return song


@pytest.mark.parametrize("swung", [False, True])
@pytest.mark.parametrize("tempo", [118.4, 82.0, 96.0, 140.0])
@pytest.mark.parametrize("phase", [0.0, 0.2, 0.37, 0.7])
def test_groove_with_hats_and_syncopated_bass_gets_its_real_beat_phase(
    phase, tempo, swung
):
    song = groove(tempo, phase, swung=swung)
    beat = 60.0 / tempo
    grid = song_match.estimate_grid(song, tempo)

    assert circular(grid.beat_phase, phase, beat) < TOLERANCE
    assert circular(grid.bar_offset, phase, 4 * beat) < TOLERANCE
    assert grid.confidence > song_match.MIN_PHASE_CONFIDENCE
    found = song_match.estimate_offset(song, tempo)
    assert circular(found, phase, 4 * beat) < TOLERANCE


def test_tempo_hint_a_little_off_is_corrected():
    song = groove(118.4, 0.37)
    grid = song_match.estimate_grid(song, 118.4 * 0.985)

    assert grid.bpm == pytest.approx(118.4, rel=0.002)
    assert circular(grid.bar_offset, 0.37, 240.0 / 118.4) < TOLERANCE


def test_downbeat_follows_the_chords_not_the_kick_pattern():
    tempo, phase = 100.0, 0.31
    beat = 60.0 / tempo
    song = groove(tempo, phase, kicks=(1,), snares=(), bass_shift=1)
    kicks = [n for n in song.instruments[0].notes if n.pitch == 36]
    grid = song_match.estimate_grid(song, tempo)

    assert all(circular(n.start - phase, beat, 4 * beat) < 1e-6 for n in kicks)
    assert circular(grid.beat_phase, phase, beat) < TOLERANCE
    assert circular(grid.bar_offset, phase, 4 * beat) < TOLERANCE


def quarter_notes(count, start=0.31, tempo=120.0, velocity=90):
    inst = pretty_midi.Instrument(0)
    for k in range(count):
        at = start + k * 60.0 / tempo
        inst.notes.append(pretty_midi.Note(velocity, 60, at, at + 0.4))
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def jittered(song, spread, seed):
    rng = random.Random(seed)
    for inst in song.instruments:
        for note in inst.notes:
            shift = max(rng.gauss(0.0, spread), -note.start)
            note.start += shift
            note.end += shift
    return song


def random_onsets(count, seed):
    rng = random.Random(seed)
    inst = pretty_midi.Instrument(0)
    for _ in range(count):
        at = rng.uniform(0.0, 30.0)
        inst.notes.append(
            pretty_midi.Note(rng.randint(50, 110), 60, at, at + 0.3)
        )
    song = pretty_midi.PrettyMIDI()
    song.instruments.append(inst)
    return song


def beats_of(count, tempo, phase, length, drum=None):
    inst = pretty_midi.Instrument(0, is_drum=drum is not None)
    beat = 60.0 / tempo
    for k in range(count):
        at = phase + k * beat
        inst.notes.append(
            pretty_midi.Note(90, drum or 60, at, at + length * beat)
        )
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def only(song, *names):
    song.instruments = [i for i in song.instruments if i.name in names]
    return song


def melody_line(tempo, phase, seed, drop=0, bars=24):
    rng = random.Random(seed)
    beat = 60.0 / tempo
    inst = pretty_midi.Instrument(0, name="Vocals")
    position = 0.0
    while position < 4 * bars:
        length = rng.choice((0.5, 0.5, 1, 1, 1.5, 2))
        at = phase + position * beat
        pitch = rng.choice((60, 62, 64, 65, 67, 69, 71, 72)) - drop
        velocity = rng.randint(70, 100)
        if at >= 0:
            inst.notes.append(
                pretty_midi.Note(velocity, pitch, at, at + 0.9 * length * beat)
            )
        position += length
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def bass_quarters(tempo, phase, seed, bars=24):
    rng = random.Random(seed)
    beat = 60.0 / tempo
    inst = pretty_midi.Instrument(33, name="Bass")
    for bar in range(bars):
        root = (36, 41, 43, 38)[bar % 4]
        for k in range(4):
            if k and rng.random() < 0.2:
                continue
            at = phase + (4 * bar + k) * beat
            inst.notes.append(
                pretty_midi.Note(
                    80 if k else 110,
                    root + 7 * bool(k),
                    at,
                    at + 0.9 * beat,
                )
            )
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def sixteenth_song(loud, soft, tempo, phase):
    beat = 60.0 / tempo
    inst = pretty_midi.Instrument(0)
    for k in range(loud):
        at = phase + k * beat
        inst.notes.append(pretty_midi.Note(100, 60, at, at + 0.1 * beat))
    for k in range(soft):
        at = phase - beat / 4 + (2 * k + 1) * beat
        inst.notes.append(pretty_midi.Note(20, 67, at, at + beat))
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


LOOPS = {
    "C Am F G": [(0, 4), (9, 3), (5, 4), (7, 4)],
    "Am G F E": [(9, 3), (7, 4), (5, 4), (4, 4)],
    "F G C C": [(5, 4), (7, 4), (0, 4), (0, 4)],
}


def chord_loop(tempo, phase, loop, beats, drums=True, bars=16):
    beat = 60.0 / tempo
    names = ("Drums", "Bass", "Pad", "Lead")
    parts = {n: pretty_midi.Instrument(0, n == "Drums", n) for n in names}

    def put(name, pitch, velocity, at, length):
        note = pretty_midi.Note(velocity, pitch, at, at + length * beat)
        parts[name].notes.append(note)

    for bar in range(bars):
        start = phase + 4 * bar * beat
        if drums:
            for k in range(4):
                put(
                    "Drums",
                    36 if k % 2 == 0 else 38,
                    105,
                    start + k * beat,
                    0.1,
                )
            for k in range(8):
                put("Drums", 42, 70, start + k * beat / 2, 0.1)
        for part in range(int(4 / beats)):
            root, third = LOOPS[loop][(bar * 4 // beats + part) % 4]
            triad = (root, root + third, root + 7)
            at = start + part * beats * beat
            put("Bass", 28 + (root - 28) % 12, 95, at, 0.9 * beats)
            for pc in triad:
                put("Pad", 55 + (pc - 55) % 12, 60, at, 0.95 * beats)
            for k in range(beats):
                pc = triad[k % 3]
                put("Lead", 64 + (pc - 64) % 12, 85, at + k * beat, 0.8)
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments = [p for p in parts.values() if p.notes]
    return song


def meter_song(beats, tempo, bars):
    beat = 60.0 / tempo
    parts = {n: pretty_midi.Instrument(0, n == "Drums") for n in "dbpl"}

    def put(name, pitch, velocity, at, length):
        note = pretty_midi.Note(velocity, pitch, at, at + length * beat)
        parts[name].notes.append(note)

    for bar in range(bars):
        start = bar * beats * beat
        root, third = LOOPS["C Am F G"][bar % 4]
        triad = (root, root + third, root + 7)
        put("d", 36, 110, start, 0.1)
        for k in range(1, beats):
            put("d", 38, 90, start + k * beat, 0.1)
        for k in range(2 * beats):
            put("d", 42, 70, start + k * beat / 2, 0.1)
        put("b", 28 + (root - 28) % 12, 95, start, 0.9 * beats)
        for pc in triad:
            put("p", 55 + (pc - 55) % 12, 60, start, 0.95 * beats)
        for k in range(beats):
            put("l", 64 + (triad[k % 3] - 64) % 12, 85, start + k * beat, 0.8)
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments = [p for p in parts.values() if p.notes]
    return song


PHRASES = ((0, 1, 2, 3), (0, 1.5, 2, 3.5), (0, 0.5, 1, 2, 3), (1, 3), (0, 2))


def phrase_line(tempo, phase, seed, bars):
    rng = random.Random(seed)
    beat = 60.0 / tempo
    inst = pretty_midi.Instrument(80)
    for bar in range(bars):
        root, third = LOOPS["C Am F G"][bar % 4]
        for position in rng.choice(PHRASES):
            at = phase + (4 * bar + position) * beat
            pc = (root, root + third, root + 7)[rng.randrange(3)]
            if at >= 0:
                pitch = 64 + (pc - 64) % 12 + 12 * rng.randrange(2)
                inst.notes.append(
                    pretty_midi.Note(90, pitch, at, at + 0.7 * beat)
                )
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def arp_song(tempo, phase, bars):
    beat = 60.0 / tempo
    inst = pretty_midi.Instrument(81)
    for bar in range(bars):
        root, third = LOOPS["C Am F G"][bar % 4]
        low = [48 + (pc - 48) % 12 for pc in (root, root + third, root + 7)]
        pitches = low + [n + 12 for n in low] + [n + 24 for n in low]
        for k in range(16):
            at = phase + (16 * bar + k) * beat / 4
            if at >= 0:
                pitch = pitches[k % 8 if k % 8 < 6 else 10 - k % 8]
                inst.notes.append(
                    pretty_midi.Note(
                        100 if k % 4 == 0 else 75, pitch, at, at + 0.22 * beat
                    )
                )
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def test_empty_and_short_songs_give_no_grid():
    short = quarter_notes(song_match.MIN_GRID_ONSETS - 1, start=0.0)
    for song in (None, pretty_midi.PrettyMIDI()):
        assert song_match.estimate_grid(song, 120.0) is None
        assert song_match.estimate_offset(song, 120.0) == 0.0
    assert song_match.estimate_grid(short, 120.0) is None
    short.instruments[0].notes.append(pretty_midi.Note(90, 60, 7.5, 7.9))
    assert song_match.estimate_grid(short, 120.0) is not None


@pytest.mark.parametrize("count", [4, 8, 12, 15])
def test_a_short_song_on_the_beat_still_gets_its_phase(count):
    song = quarter_notes(count)

    assert song_match.estimate_grid(song, 120.0) is None
    assert song_match.estimate_offset(song, 120.0) == pytest.approx(
        0.31, abs=0.02
    )


@pytest.mark.parametrize("phase", [0.31, 0.77])
@pytest.mark.parametrize("loud, soft", [(8, 4), (10, 5)])
def test_a_short_song_on_a_sixteenth_lattice_gets_its_quarter_phase(
    loud, soft, phase
):
    song = sixteenth_song(loud, soft, 120.0, phase)

    assert song_match.MIN_GRID_ONSETS > loud + soft
    assert circular(song_match.estimate_offset(song, 120.0), phase, 0.5) < 0.02


def test_a_song_of_a_few_notes_or_no_velocity_gives_no_offset():
    assert song_match.estimate_offset(quarter_notes(3), 120.0) == 0.0
    silent = quarter_notes(20, velocity=0)
    assert song_match.estimate_grid(silent, 120.0) is None
    assert song_match.estimate_offset(silent, 120.0) == 0.0


@pytest.mark.parametrize("spread", [0.03, 0.045])
@pytest.mark.parametrize("phase", [0.13, 0.31, 0.77, 1.42])
def test_timing_jitter_still_gives_the_real_bar_line(phase, spread):
    for seed in range(3):
        song = jittered(groove(120.0, phase, seed=seed), spread, seed)
        found = song_match.estimate_offset(song, 120.0)
        assert circular(found, phase, 2.0) < TOLERANCE


@pytest.mark.parametrize("error", [0.1, 0.3])
def test_a_tempo_hint_a_little_off_gives_the_offset_on_its_own_grid(error):
    hint, phase, bars = 120.0, 0.31, 90
    song = groove(hint + error, phase, bars=bars)
    bar, true_bar = 240.0 / hint, 240.0 / (hint + error)
    middle = phase - bars / 2 * (bar - true_bar)

    assert (
        circular(song_match.estimate_offset(song, hint), middle, bar)
        < TOLERANCE
    )


def test_onsets_without_a_beat_rarely_give_an_offset():
    found = sum(
        song_match.estimate_offset(random_onsets(40, seed), 120.0) != 0.0
        for seed in range(200)
    )

    assert found < 20


@pytest.mark.parametrize("count", [6, 12])
def test_a_few_onsets_without_a_beat_rarely_give_an_offset(count):
    found = sum(
        song_match.estimate_offset(random_onsets(count, seed), 120.0) != 0.0
        for seed in range(200)
    )

    assert found < 40


@pytest.mark.parametrize("drum", [None, 42, 36])
@pytest.mark.parametrize("length", [0.0, 0.1, 0.25])
def test_short_notes_on_the_beat_give_the_beat_phase(drum, length):
    for count, tempo, phase in itertools.product(
        (16, 40), (95.0, 120.0), (0.13, 0.31, 0.77)
    ):
        beat = 60.0 / tempo
        song = beats_of(count, tempo, phase, length, drum)
        found = song_match.estimate_offset(song, tempo)
        grid = song_match.estimate_grid(song, tempo)

        assert circular(found, phase, beat) < TOLERANCE
        assert circular(grid.beat_phase, phase, beat) < TOLERANCE


@pytest.mark.parametrize("tempo", [70.0, 95.0, 120.0, 150.0])
def test_a_vocal_line_that_starts_on_a_bar_line_gets_no_offset(tempo):
    found = [
        song_match.estimate_offset(
            only(groove(tempo, 0.0, seed=seed), "Vocals"), tempo
        )
        for seed in range(8)
    ]

    assert found[0] == 0.0
    assert sum(offset != 0.0 for offset in found) <= 1


@pytest.mark.parametrize("drop", [0, 24])
def test_a_random_melody_on_a_bar_line_gets_no_offset(drop):
    found = [
        song_match.estimate_offset(
            jittered(melody_line(tempo, 0.0, seed, drop), spread, seed),
            tempo,
        )
        for tempo in (70.0, 95.0, 120.0, 150.0)
        for spread in (0.0, 0.02)
        for seed in range(8)
    ]

    assert not any(found)


@pytest.mark.parametrize("loop", list(LOOPS))
def test_chords_that_change_faster_than_a_bar_do_not_move_the_bar_line(loop):
    for tempo, phase, beats, drums in itertools.product(
        (90.0, 120.0), (0.0, 0.31), (1, 2), (True, False)
    ):
        song = chord_loop(tempo, phase, loop, beats, drums)
        found = song_match.estimate_offset(song, tempo)

        assert circular(found, phase, 240.0 / tempo) < TOLERANCE


@pytest.mark.parametrize("beats", [2, 3, 5])
def test_a_song_in_another_meter_that_starts_on_its_downbeat_gets_no_offset(
    beats,
):
    for tempo, bars in itertools.product((80.0, 100.0, 128.0), (8, 16, 24)):
        song = meter_song(beats, tempo, bars)

        assert song_match.estimate_offset(song, tempo) == 0.0


@pytest.mark.parametrize("tempo", [80.0, 100.0, 128.0, 160.0])
def test_notes_a_little_early_do_not_make_a_pickup_of_nearly_a_beat(tempo):
    for early, seed in itertools.product((0.025, 0.04, 0.06), range(3)):
        songs = (
            arp_song(tempo, -early, 8),
            phrase_line(tempo, -early, seed, 8),
        )
        for song in songs:
            found = song_match.estimate_offset(song, tempo)

            assert circular(found, 0.0, 240.0 / tempo) < 0.06


@pytest.mark.parametrize("tempo", [80.0, 100.0, 128.0])
def test_a_tempo_hint_a_little_slow_does_not_make_a_pickup_of_nearly_a_beat(
    tempo,
):
    for error, seed in itertools.product((0.001, 0.002), range(3)):
        songs = (
            arp_song(tempo * (1 + error), 0.0, 16),
            phrase_line(tempo * (1 + error), 0.0, seed, 16),
        )
        for song in songs:
            found = song_match.estimate_offset(song, tempo)

            assert circular(found, 0.0, 240.0 / tempo) < 0.06


def test_fold_offset_zeroes_the_window_before_a_bar_line_it_is_given():
    assert song_match.fold_offset(1.95, 120.0) == pytest.approx(1.95)
    assert song_match.fold_offset(1.95, 120.0, early=0.08) == 0.0
    assert song_match.fold_offset(0.3, 120.0, early=0.08) == pytest.approx(0.3)


PICKUP_SONGS = {
    "band": lambda tempo, phase, seed: groove(tempo, phase, seed=seed),
    "chords and vocal": lambda tempo, phase, seed: only(
        groove(tempo, phase, seed=seed), "Other", "Vocals"
    ),
    "chords and bass": lambda tempo, phase, seed: only(
        groove(tempo, phase, seed=seed), "Other", "Bass"
    ),
    "kick on the bar": lambda tempo, phase, seed: only(
        groove(tempo, phase, seed=seed, kicks=(0,), snares=(2,)), "Drums"
    ),
    "accented bass": lambda tempo, phase, seed: bass_quarters(
        tempo, phase, seed
    ),
}


@pytest.mark.parametrize("name", list(PICKUP_SONGS))
def test_a_pickup_still_gives_the_bar_line(name):
    for tempo, phase, seed in itertools.product(
        (95.0, 120.0), (0.13, 0.77, 1.1, 1.7), range(3)
    ):
        song = PICKUP_SONGS[name](tempo, phase, seed)
        found = song_match.estimate_offset(song, tempo)

        assert circular(found, phase, 240.0 / tempo) < TOLERANCE


@pytest.mark.parametrize("changes", [{"search": -0.01}, {"div": 0}])
def test_estimate_grid_rejects_a_bad_search_or_division(changes):
    with pytest.raises(ValueError, match="search"):
        song_match.estimate_grid(groove(120.0, 0.0), 120.0, **changes)


def test_estimate_offset_wrapper_keeps_the_pinned_values():
    tempo = 100.0
    bar = 240.0 / tempo
    for seconds in (0.15, 0.3, 0.45, 1.2):
        found = song_match.estimate_offset(groove(tempo, seconds), tempo)
        assert found == pytest.approx(seconds, abs=0.02)
    assert song_match.estimate_offset(groove(tempo, 0.0), tempo) == 0.0
    assert song_match.estimate_offset(groove(tempo, 0.01), tempo) == 0.0
    assert song_match.estimate_offset(groove(tempo, bar - 0.01), tempo) == 0.0
    song = groove(tempo, 0.45)
    grid = song_match.estimate_grid(song, tempo, search=0)
    assert song_match.estimate_offset(song, tempo) == grid.bar_offset
    with pytest.raises(ValueError, match="bpm"):
        song_match.estimate_grid(song, 5.0)


def _dense_song(notes, seconds, tempo):
    rng = np.random.default_rng(3)
    step = 60.0 / tempo / 4
    starts = rng.integers(0, int(seconds / step), size=notes) * step + 0.11
    pitches = rng.integers(36, 84, size=notes)
    inst = pretty_midi.Instrument(0)
    inst.notes = [
        pretty_midi.Note(80, int(p), float(s), float(s) + 0.2)
        for s, p in zip(starts, pitches, strict=True)
    ]
    song = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(inst)
    return song


def test_large_song_is_memory_bounded_and_fast():
    song = _dense_song(20000, 300.0, 120.0)
    tracemalloc.start()
    started = time.perf_counter()
    grid = song_match.estimate_grid(song, 120.0)
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < 100 * 1024 * 1024
    assert elapsed < 10.0
    assert circular(grid.beat_phase, 0.11, 0.125) < TOLERANCE


def timed_offset(song):
    song_match.estimate_offset(quarter_notes(20), 120.0)
    started = time.perf_counter()
    found = song_match.estimate_offset(song, 120.0)
    return found, time.perf_counter() - started


def test_a_stray_far_future_note_gives_no_offset_without_the_cost():
    song = _dense_song(20000, 300.0, 120.0)
    song.instruments[0].notes.append(pretty_midi.Note(80, 60, 1e7, 1e7 + 1))

    found, elapsed = timed_offset(song)

    assert found == 0.0
    assert elapsed < 1.0


def test_a_note_held_for_hours_costs_little():
    song = _dense_song(20000, 300.0, 120.0)
    song.instruments[0].notes.append(pretty_midi.Note(80, 40, 0.0, 20000.0))
    song.instruments[0].notes.append(pretty_midi.Note(80, 60, 5.0, 1e9))

    found, elapsed = timed_offset(song)

    assert circular(found, 0.11, 0.125) < TOLERANCE
    assert elapsed < 1.0


@pytest.mark.parametrize("bad", [(float("nan"), 2.0), (1.0, float("nan"))])
@pytest.mark.parametrize("stop", [float("inf"), 5.0])
def test_a_note_that_is_not_finite_is_ignored(bad, stop):
    song = quarter_notes(40)
    expected = song_match.estimate_offset(song, 120.0)
    song.instruments[0].notes.append(pretty_midi.Note(90, 60, *bad))
    song.instruments[0].notes.append(
        pretty_midi.Note(90, 60, stop, float("inf"))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = song_match.estimate_offset(song, 120.0)
        grid = song_match.estimate_grid(song, 120.0)

    assert found == pytest.approx(expected)
    assert grid is not None


def test_large_song_grid_and_offset_take_seconds_not_minutes():
    song = _dense_song(20000, 300.0, 120.0)
    song_match.estimate_grid(groove(120.0, 0.0), 120.0)
    started = time.perf_counter()
    song_match.estimate_grid(song, 120.0)
    offset = song_match.estimate_offset(song, 120.0)
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert circular(offset, 0.11, 0.125) < TOLERANCE
