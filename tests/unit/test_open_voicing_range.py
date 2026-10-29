import pretty_midi

from korshmer.midi_algorithms import apply_open_voicing


def _chord(pitches):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch in pitches:
        inst.notes.append(pretty_midi.Note(80, pitch, 0.0, 1.0))
    midi.instruments.append(inst)
    return midi, inst


def test_open_voicing_raises_second_note_an_octave():
    midi, inst = _chord([48, 52, 55])
    apply_open_voicing(midi)
    assert sorted(n.pitch for n in inst.notes) == [48, 55, 64]


def test_open_voicing_keeps_pitches_within_midi_range(tmp_path):
    midi, inst = _chord([100, 120, 125])
    apply_open_voicing(midi)
    assert all(0 <= n.pitch <= 127 for n in inst.notes)
    midi.write(str(tmp_path / "out.mid"))


def test_open_voicing_still_shifts_when_top_is_exactly_127():
    midi, inst = _chord([60, 115, 120])
    apply_open_voicing(midi)
    assert sorted(n.pitch for n in inst.notes) == [60, 120, 127]
