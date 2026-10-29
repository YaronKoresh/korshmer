import pretty_midi
import pytest

from korshmer.midi.transforms import time_stretch


def _build_midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 1.0, 2.0))
    inst.control_changes.append(pretty_midi.ControlChange(1, 64, 1.0))
    inst.pitch_bends.append(pretty_midi.PitchBend(2000, 1.5))
    midi.instruments.append(inst)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 2.0))
    midi.key_signature_changes.append(pretty_midi.KeySignature(2, 4.0))
    midi.lyrics.append(pretty_midi.Lyric("la", 3.0))
    return midi


def test_time_stretch_scales_notes_and_control_changes():
    result = time_stretch(_build_midi(), factor=2.0)
    inst = result.instruments[0]
    assert inst.notes[0].start == pytest.approx(2.0)
    assert inst.notes[0].end == pytest.approx(4.0)
    assert inst.control_changes[0].time == pytest.approx(2.0)


def test_time_stretch_scales_pitch_bends():
    result = time_stretch(_build_midi(), factor=2.0)
    assert result.instruments[0].pitch_bends[0].time == pytest.approx(3.0)


def test_time_stretch_scales_meta_events():
    result = time_stretch(_build_midi(), factor=0.5)
    assert result.time_signature_changes[0].time == pytest.approx(1.0)
    assert result.key_signature_changes[0].time == pytest.approx(2.0)
    assert result.lyrics[0].time == pytest.approx(1.5)


def test_time_stretch_does_not_modify_input():
    midi = _build_midi()
    time_stretch(midi, factor=2.0)
    assert midi.instruments[0].pitch_bends[0].time == pytest.approx(1.5)


@pytest.mark.parametrize(
    "bad", [0, -1, -0.5, float("nan"), float("inf"), "2", True]
)
def test_time_stretch_rejects_invalid_factor(bad):
    with pytest.raises(ValueError):
        time_stretch(_build_midi(), factor=bad)


def test_time_stretch_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        time_stretch(_build_midi(), ratio=0)


def test_time_stretch_invalid_factor_leaves_input_unchanged():
    midi = _build_midi()
    with pytest.raises(ValueError):
        time_stretch(midi, factor=0)
    assert midi.instruments[0].notes[0].end == pytest.approx(2.0)
