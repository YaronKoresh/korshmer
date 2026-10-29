import pretty_midi
import pytest

from korshmer.midi_algorithms import smart_quantize

BPM = 120.0
STEP = 0.25


def _song(first, count=16, jitter=0.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    inst = pretty_midi.Instrument(0)
    for i in range(count):
        start = first + i * STEP + (jitter if i % 2 else -jitter)
        inst.notes.append(pretty_midi.Note(80, 60 + i % 5, start, start + 0.2))
    pm.instruments.append(inst)
    return pm


def _starts(pm):
    return [n.start for n in pm.instruments[0].notes]


@pytest.mark.parametrize("first", [0.35, 0.1, 0.2])
def test_notes_stay_on_the_phase_of_the_song(first):
    pm = smart_quantize(_song(first), BPM, None, grid_divisions=2)

    assert _starts(pm) == pytest.approx(
        [first + i * STEP for i in range(16)], abs=1e-6
    )


def test_slightly_loose_notes_are_pulled_to_the_song_phase():
    pm = smart_quantize(_song(0.35, jitter=0.02), BPM, None, grid_divisions=2)

    assert _starts(pm) == pytest.approx(
        [0.35 + i * STEP for i in range(16)], abs=0.03
    )


def test_song_on_the_grid_is_unchanged():
    pm = smart_quantize(_song(0.0), BPM, None, grid_divisions=2)

    assert _starts(pm) == pytest.approx([i * STEP for i in range(16)])


def test_given_beat_times_still_win():
    pm = smart_quantize(_song(0.35), BPM, [0.0], grid_divisions=2)

    assert _starts(pm)[0] == pytest.approx(0.25)


def test_no_notes_is_fine():
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    pm.instruments.append(pretty_midi.Instrument(0))

    assert smart_quantize(pm, BPM, None) is pm


def test_scattered_notes_fall_back_to_the_zero_grid():
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    inst = pretty_midi.Instrument(0)
    for i, start in enumerate([0.0, 0.05, 0.11, 0.17, 0.2]):
        inst.notes.append(pretty_midi.Note(80, 60 + i, start, start + 0.1))
    pm.instruments.append(inst)

    smart_quantize(pm, BPM, None, grid_divisions=2)

    assert all(
        n.start % STEP == pytest.approx(0.0, abs=1e-6)
        or n.start % STEP == pytest.approx(STEP, abs=1e-6)
        for n in inst.notes
    )
