import pretty_midi
import pytest

from korshmer.midi_algorithms import apply_lfo_modulation


def _midi():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(80, 60, 0.0, 1.0))
    inst.notes.append(pretty_midi.Note(80, 62, 0.3, 1.3))
    pm.instruments.append(inst)
    return pm


@pytest.mark.parametrize("shape", ["Sine", "Square", "Triangle", "Saw"])
@pytest.mark.parametrize("target", ["Velocity", "CC11 (Expression)"])
def test_every_shape_works_for_every_target(shape, target):
    pm = apply_lfo_modulation(_midi(), 1.0, 20, target, shape)
    inst = pm.instruments[0]
    if target == "Velocity":
        assert all(1 <= n.velocity <= 127 for n in inst.notes)
    else:
        assert inst.control_changes
        assert all(0 <= c.value <= 127 for c in inst.control_changes)


def test_square_cc11_alternates_between_extremes():
    pm = apply_lfo_modulation(_midi(), 1.0, 20, "CC11 (Expression)", "Square")
    values = {c.value for c in pm.instruments[0].control_changes}
    assert values <= {1, 127}


def test_triangle_and_saw_ranges():
    from korshmer.midi_algorithms import _lfo_value

    assert _lfo_value("Triangle", 1.0, 0.0) == pytest.approx(-1.0)
    assert _lfo_value("Triangle", 1.0, 0.5) == pytest.approx(1.0)
    assert _lfo_value("Saw", 1.0, 0.0) == pytest.approx(-1.0)
    assert _lfo_value("Saw", 1.0, 0.75) == pytest.approx(0.5)


def test_unknown_shape_or_target_is_rejected():
    with pytest.raises(ValueError):
        apply_lfo_modulation(_midi(), 1.0, 20, "Velocity", "Wobble")
    with pytest.raises(ValueError):
        apply_lfo_modulation(_midi(), 1.0, 20, "Pan", "Sine")
