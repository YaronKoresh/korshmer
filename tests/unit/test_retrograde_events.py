import pretty_midi

from korshmer.midi.transforms import apply_retrograde


def _midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0.0, 1.0))
    inst.notes.append(pretty_midi.Note(100, 64, 3.0, 4.0))
    inst.control_changes.append(pretty_midi.ControlChange(64, 127, 0.5))
    inst.control_changes.append(pretty_midi.ControlChange(64, 0, 1.0))
    inst.pitch_bends.append(pretty_midi.PitchBend(2000, 3.5))
    midi.instruments.append(inst)
    return midi


def test_retrograde_mirrors_control_changes_and_pitch_bends():
    inst = apply_retrograde(_midi()).instruments[0]
    assert [c.time for c in inst.control_changes] == [3.0, 3.5]
    assert [c.value for c in inst.control_changes] == [0, 127]
    assert [b.time for b in inst.pitch_bends] == [0.5]


def test_retrograde_keeps_note_mirroring():
    inst = apply_retrograde(_midi()).instruments[0]
    spans = sorted((n.pitch, n.start, n.end) for n in inst.notes)
    assert spans == [(60, 3.0, 4.0), (64, 0.0, 1.0)]


def test_retrograde_events_after_last_note_stay_non_negative():
    midi = _midi()
    midi.instruments[0].pitch_bends[0].time = 6.0
    inst = apply_retrograde(midi).instruments[0]
    assert all(c.time >= 0 for c in inst.control_changes)
    assert all(b.time >= 0 for b in inst.pitch_bends)
