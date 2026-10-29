import pretty_midi

from korshmer.midi_algorithms import apply_legato_staccato


def _midi(spec):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for pitch, start, end in spec:
        inst.notes.append(pretty_midi.Note(90, pitch, start, end))
    midi.instruments.append(inst)
    return midi


def test_legato_keeps_note_whose_next_start_is_within_20_ms():
    midi = _midi([(60, 0.0, 0.5), (64, 0.01, 0.5), (67, 1.0, 1.5)])
    apply_legato_staccato(midi, "Legato")
    for note in midi.instruments[0].notes:
        assert note.end > note.start
    first = midi.instruments[0].notes[0]
    assert (first.start, first.end) == (0.0, 0.5)


def test_legato_still_extends_to_next_start():
    midi = _midi([(60, 0.0, 0.2), (64, 1.0, 1.5)])
    apply_legato_staccato(midi, "Legato")
    assert midi.instruments[0].notes[0].end == 0.98
