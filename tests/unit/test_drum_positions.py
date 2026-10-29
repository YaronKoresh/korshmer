"""Drums must keep the positions they were transcribed at.

Drum hits are detected as the onsets actually heard in the source. The grid
snap at the end of a conversion must leave those onsets where they are, so the
drum part matches the original timing instead of being pulled onto a mechanical
every-eighth grid, while melodic notes are still snapped. Overlapping same-pitch
hits are still trimmed, exactly as they were when drums were snapped.
"""

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
    off_grid = 0.19  # between the 1/8 (0.25) grid lines at 120 BPM
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
    # Two closed-hat hits closer together than the first note's length overlap.
    # Their onsets must be left untouched, but the overlap must still be trimmed
    # just as it was when drums were snapped onto the grid.
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
