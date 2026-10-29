import pretty_midi
import pytest

from korshmer.midi_algorithms import generate_euclidean_rhythm


def _midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, 60, 0.0, 2.0))
    midi.instruments.append(inst)
    return midi


@pytest.mark.parametrize("steps", [0, -1, 0.5, 257, float("nan"), True])
def test_invalid_steps_are_rejected_instead_of_hanging(steps):
    with pytest.raises(ValueError):
        generate_euclidean_rhythm(_midi(), steps=steps)


@pytest.mark.parametrize("hits", [-1, 17, 1.5])
def test_invalid_hits_are_rejected(hits):
    with pytest.raises(ValueError):
        generate_euclidean_rhythm(_midi(), steps=16, hits=hits)


@pytest.mark.parametrize("note", [-1, 128, 1000])
def test_note_outside_midi_range_is_rejected(note):
    with pytest.raises(ValueError):
        generate_euclidean_rhythm(_midi(), note_num=note)


def test_valid_arguments_still_build_the_pattern():
    result = generate_euclidean_rhythm(_midi(), steps=8, hits=3, note_num=38)
    drums = result.instruments[-1]
    assert drums.is_drum
    assert {n.pitch for n in drums.notes} == {38}
    assert len(drums.notes) >= 3
