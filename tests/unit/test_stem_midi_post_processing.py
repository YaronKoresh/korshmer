import pretty_midi

from korshmer.midi_algorithms import (
    apply_scale_filter,
    fix_legato_and_flutter,
)


def _midi(notes):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(90, pitch, start, end))
    midi.instruments.append(inst)
    return midi


def test_scale_filter_keeps_every_note_and_snaps_to_scale():
    midi = _midi([(60, 0.0, 0.4), (61, 0.5, 0.9), (62, 1.0, 1.4)])
    apply_scale_filter(midi, 0, "Major")
    pitches = [n.pitch for n in midi.instruments[0].notes]
    assert len(pitches) == 3
    assert {p % 12 for p in pitches} <= {0, 2, 4, 5, 7, 9, 11}


def test_scale_filter_stays_inside_midi_range():
    midi = _midi([(127, 0.0, 0.4), (0, 0.5, 0.9)])
    apply_scale_filter(midi, 0, "Major")
    assert all(0 <= n.pitch <= 127 for n in midi.instruments[0].notes)


def test_legato_keeps_separately_played_repeated_notes():
    midi = _midi([(60, k * 0.25, k * 0.25 + 0.22) for k in range(4)])
    fix_legato_and_flutter(midi)
    assert len(midi.instruments[0].notes) == 4


def test_legato_still_merges_overlapping_and_flutter_fragments():
    midi = _midi([(60, 0.0, 0.5), (60, 0.4, 0.8), (60, 0.82, 0.86)])
    fix_legato_and_flutter(midi)
    notes = midi.instruments[0].notes
    assert len(notes) == 1
    assert notes[0].end == 0.86
