import pretty_midi
import pytest

from korshmer.midi.transforms import apply_strumming


def _chord():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for pitch in (60, 64, 67):
        inst.notes.append(pretty_midi.Note(80, pitch, 0.0, 1.0))
    midi.instruments.append(inst)
    return midi


@pytest.mark.parametrize(
    "delay",
    [-0.1, float("nan"), float("inf"), -float("inf"), "0.1", None, True],
)
def test_invalid_delay_is_rejected(delay):
    with pytest.raises(ValueError):
        apply_strumming(_chord(), delay_sec=delay)


def test_invalid_delay_leaves_notes_untouched():
    midi = _chord()
    with pytest.raises(ValueError):
        apply_strumming(midi, delay_sec=-0.1)
    assert [n.start for n in midi.instruments[0].notes] == [0.0, 0.0, 0.0]


def test_valid_delay_staggers_notes_after_time_zero():
    result = apply_strumming(_chord(), delay_sec=0.1)
    starts = [n.start for n in result.instruments[0].notes]
    assert starts == pytest.approx([0.0, 0.1, 0.2])
    assert min(starts) >= 0.0


def test_zero_delay_is_allowed():
    result = apply_strumming(_chord(), delay_sec=0)
    assert [n.start for n in result.instruments[0].notes] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("direction", ["Sideways", "up", "", None, 5])
def test_unknown_direction_is_rejected_and_notes_are_untouched(direction):
    midi = _chord()
    with pytest.raises(ValueError, match="direction"):
        apply_strumming(midi, direction=direction)
    assert [n.start for n in midi.instruments[0].notes] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("curve", ["Bogus", "linear", "", None, 5])
def test_unknown_curve_is_rejected_and_notes_are_untouched(curve):
    midi = _chord()
    with pytest.raises(ValueError, match="curve"):
        apply_strumming(midi, curve=curve)
    assert [n.start for n in midi.instruments[0].notes] == [0.0, 0.0, 0.0]


def test_known_directions_and_curves_still_strum():
    up = apply_strumming(_chord(), direction="Up")
    assert [n.pitch for n in up.instruments[0].notes] == [67, 64, 60]
    down = apply_strumming(_chord(), direction="Down", curve="Accelerate")
    assert [round(n.start, 3) for n in down.instruments[0].notes] == [
        0.0,
        0.015,
        0.06,
    ]
