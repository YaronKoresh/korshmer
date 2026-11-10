import pretty_midi

from korshmer.midi.adaptive_snap import snap_adaptive


def _midi(start, drum):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, is_drum=drum)
    inst.notes.append(
        pretty_midi.Note(velocity=100, pitch=38, start=start, end=start + 0.1)
    )
    pm.instruments.append(inst)
    return pm


def test_snap_drums_false_keeps_drum_notes_at_their_original_position():
    off_grid = 0.19
    drum = _midi(off_grid, drum=True)
    melodic = _midi(off_grid, drum=False)
    snap_adaptive(drum, 120.0, 0.0, snap_drums=False)
    snap_adaptive(melodic, 120.0, 0.0, snap_drums=True)
    assert drum.instruments[0].notes[0].start == off_grid
    assert melodic.instruments[0].notes[0].start != off_grid


def test_snap_drums_true_still_snaps_drums_by_default():
    off_grid = 0.19
    drum = _midi(off_grid, drum=True)
    snap_adaptive(drum, 120.0, 0.0)
    assert drum.instruments[0].notes[0].start != off_grid


def test_snap_drums_false_still_trims_overlapping_same_pitch_drums():

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(
        pretty_midi.Note(velocity=100, pitch=42, start=0.190, end=0.190 + 0.06)
    )
    inst.notes.append(
        pretty_midi.Note(velocity=100, pitch=42, start=0.225, end=0.225 + 0.06)
    )
    pm.instruments.append(inst)

    snap_adaptive(pm, 120.0, 0.0, snap_drums=False)

    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    assert [round(n.start, 3) for n in notes] == [0.190, 0.225]
    assert notes[0].end <= notes[1].start
