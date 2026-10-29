import pretty_midi

from korshmer.midi.transforms import force_monophonic


def _midi(notes):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        )
    midi.instruments.append(inst)
    return midi


def _spans(midi):
    return [(n.pitch, n.start, n.end) for n in midi.instruments[0].notes]


def test_kept_notes_never_overlap():
    midi = _midi([(60, 0, 2), (64, 0, 2), (67, 1, 3), (62, 1.02, 1.5)])
    assert _spans(force_monophonic(midi)) == [(64, 0, 1), (67, 1, 3)]


def test_notes_across_old_rounding_boundary_form_one_chord():
    midi = _midi([(60, 0.024, 1), (72, 0.026, 1)])
    spans = _spans(force_monophonic(midi))
    assert [sp[0] for sp in spans] == [72]


def test_low_mode_keeps_lowest_note():
    midi = _midi([(60, 0, 1), (64, 0, 1), (55, 2, 3)])
    result = force_monophonic(midi, "Low")
    assert [n.pitch for n in result.instruments[0].notes] == [60, 55]


def test_drums_untouched():
    midi = _midi([(36, 0, 1), (38, 0, 1)])
    midi.instruments[0].is_drum = True
    assert len(force_monophonic(midi).instruments[0].notes) == 2
