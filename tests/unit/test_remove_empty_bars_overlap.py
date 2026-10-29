import pretty_midi
import pytest

from korshmer.midi import remove_empty_bars


def _midi(spans):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in spans:
        inst.notes.append(pretty_midi.Note(100, pitch, start, end))
    midi.instruments.append(inst)
    return midi


def test_long_note_covering_gap_keeps_later_notes_in_place():
    midi = _midi([(48, 0.0, 10.0), (60, 1.0, 2.0), (62, 5.0, 6.0)])
    notes = remove_empty_bars(midi).instruments[0].notes
    assert [(n.start, n.end) for n in notes] == [
        (0.0, 10.0),
        (1.0, 2.0),
        (5.0, 6.0),
    ]


def test_real_silence_is_still_shortened():
    midi = _midi([(60, 0.0, 0.5), (62, 4.0, 4.5)])
    notes = remove_empty_bars(midi).instruments[0].notes
    assert notes[1].start == 0.5 + 0.5


def _with_events():
    midi = _midi([(60, 0.0, 1.0), (62, 10.0, 11.0), (64, 20.0, 21.0)])
    inst = midi.instruments[0]
    inst.control_changes.append(pretty_midi.ControlChange(64, 127, 10.5))
    inst.pitch_bends.append(pretty_midi.PitchBend(100, 10.2))
    inst.control_changes.append(pretty_midi.ControlChange(64, 0, 20.5))
    return midi


def test_control_changes_and_pitch_bends_follow_their_notes():
    result = remove_empty_bars(_with_events())
    inst = result.instruments[0]
    assert [n.start for n in inst.notes] == [0.0, 1.5, 3.0]
    assert [c.time for c in inst.control_changes] == [2.0, 3.5]
    assert [b.time for b in inst.pitch_bends] == pytest.approx([1.7])


def test_events_inside_removed_silence_land_at_the_cut():
    midi = _midi([(60, 0.0, 1.0), (62, 10.0, 11.0)])
    midi.instruments[0].control_changes.append(
        pretty_midi.ControlChange(1, 50, 5.0)
    )
    midi.instruments[0].control_changes.append(
        pretty_midi.ControlChange(1, 60, 0.5)
    )
    result = remove_empty_bars(midi)
    times = sorted(c.time for c in result.instruments[0].control_changes)
    assert times == [0.5, 1.5]


def test_time_signature_key_signature_and_lyrics_follow_their_notes():
    midi = _midi([(60, 0.0, 1.0), (62, 10.0, 11.0)])
    midi.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 10.0))
    midi.key_signature_changes.append(pretty_midi.KeySignature(2, 10.0))
    midi.lyrics.append(pretty_midi.Lyric("la", 10.0))
    result = remove_empty_bars(midi)
    assert result.time_signature_changes[-1].time == 1.5
    assert result.key_signature_changes[-1].time == 1.5
    assert result.lyrics[0].time == 1.5


def test_midi_without_events_is_unchanged_apart_from_notes():
    result = remove_empty_bars(_midi([(60, 0.0, 1.0)]))
    assert result.instruments[0].control_changes == []
    assert result.instruments[0].pitch_bends == []
