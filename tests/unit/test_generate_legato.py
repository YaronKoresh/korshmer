import pretty_midi

from korshmer.midi.transforms import generate_legato


def _midi(notes, is_drum=False):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=is_drum)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(80, pitch, start, end))
    midi.instruments.append(inst)
    return midi


def _spans(midi):
    return sorted((n.start, n.pitch, n.end) for n in midi.instruments[0].notes)


def test_every_chord_note_is_extended_to_next_onset():
    midi = _midi(
        [(60, 0, 0.5), (64, 0, 0.5), (67, 0, 0.5), (62, 2, 2.5), (65, 2, 2.5)]
    )
    generate_legato(midi)
    assert _spans(midi) == [
        (0, 60, 2),
        (0, 64, 2),
        (0, 67, 2),
        (2, 62, 2.5),
        (2, 65, 2.5),
    ]


def test_chord_notes_with_different_ends_all_reach_next_onset():
    midi = _midi([(60, 0, 0.5), (64, 0, 1.0), (67, 3, 3.5)])
    generate_legato(midi)
    assert _spans(midi) == [(0, 60, 3), (0, 64, 3), (3, 67, 3.5)]


def test_notes_ending_after_next_onset_are_not_shortened():
    midi = _midi([(60, 0, 2.0), (64, 1, 1.5)])
    generate_legato(midi)
    assert _spans(midi) == [(0, 60, 2.0), (1, 64, 1.5)]


def test_single_notes_and_last_note_behave_as_before():
    midi = _midi([(60, 0, 0.5), (62, 1, 1.25), (64, 2, 2.25)])
    generate_legato(midi)
    assert _spans(midi) == [(0, 60, 1), (1, 62, 2), (2, 64, 2.25)]


def test_drums_are_untouched_and_empty_input_is_fine():
    midi = _midi([(36, 0, 0.1), (38, 1, 1.1)], is_drum=True)
    generate_legato(midi)
    assert _spans(midi) == [(0, 36, 0.1), (1, 38, 1.1)]
    generate_legato(_midi([]))
