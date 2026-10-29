import numpy as np
import pretty_midi

from korshmer.midi_algorithms import generate_melody_continuation


def _midi(spec):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for pitch, start, end in spec:
        inst.notes.append(pretty_midi.Note(90, pitch, start, end))
    midi.instruments.append(inst)
    return midi


def test_two_notes_do_not_crash_for_any_seed():
    for seed in range(20):
        np.random.seed(seed)
        midi = _midi([(60, 0.0, 0.5), (62, 0.5, 1.0)])
        generate_melody_continuation(midi)


def test_source_without_usable_transitions_is_returned_unchanged():
    midi = _midi([(60, 0.0, 0.5), (62, 0.5, 1.0)])
    result = generate_melody_continuation(midi)
    assert [i.name for i in result.instruments] == [""]


def test_notes_starting_together_do_not_hang():
    midi = _midi([(60, 0.0, 0.5), (64, 0.0, 0.5), (67, 0.0, 0.5)])
    result = generate_melody_continuation(midi, bars=1)
    notes = result.instruments[-1].notes
    assert len(notes) <= 200
