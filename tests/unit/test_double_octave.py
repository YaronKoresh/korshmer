import pretty_midi
import pytest

from korshmer.midi.transforms import double_octave


def _midi(pitches, is_drum=False):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=is_drum)
    for i, pitch in enumerate(pitches):
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=i, end=i + 0.5)
        )
    midi.instruments.append(inst)
    return midi


def _pitches(midi):
    return sorted(n.pitch for n in midi.instruments[0].notes)


def test_doubles_every_note_one_octave_down():
    result = double_octave(_midi([60, 72]), -1)
    assert _pitches(result) == [48, 60, 60, 72]


def test_doubles_every_note_one_octave_up():
    result = double_octave(_midi([60, 72]), 1)
    assert _pitches(result) == [60, 72, 72, 84]


def test_notes_without_room_below_are_not_duplicated():
    result = double_octave(_midi([0, 5, 11, 12]), -1)
    assert _pitches(result) == [0, 0, 5, 11, 12]


def test_notes_without_room_above_are_not_duplicated():
    result = double_octave(_midi([100, 116, 127]), 1)
    assert _pitches(result) == [100, 112, 116, 127]


@pytest.mark.parametrize("shift", [-1, 1])
def test_no_exact_duplicate_note_is_created(shift):
    result = double_octave(_midi([0, 127, 60]), shift)
    keys = [(n.pitch, n.start, n.end) for n in result.instruments[0].notes]
    assert len(keys) == len(set(keys))


def test_zero_shift_adds_nothing():
    result = double_octave(_midi([60, 64]), 0)
    assert _pitches(result) == [60, 64]


def test_drum_tracks_are_untouched():
    result = double_octave(_midi([36, 38], is_drum=True), -1)
    assert _pitches(result) == [36, 38]
