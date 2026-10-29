import pretty_midi
import pytest

from korshmer.midi.transforms import apply_crescendo, limit_pitch_range


def _midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i in range(4):
        inst.notes.append(pretty_midi.Note(80, 50 + i * 10, i, i + 0.5))
    midi.instruments.append(inst)
    return midi


BAD = [float("nan"), float("inf"), "x", True]


@pytest.mark.parametrize("param", ["start_vel", "end_vel"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), "x", True])
def test_crescendo_rejects_invalid_velocity(param, value):
    with pytest.raises(ValueError):
        apply_crescendo(_midi(), **{param: value})


def test_crescendo_defaults_and_valid_values():
    out = apply_crescendo(_midi())
    velocities = [n.velocity for n in out.instruments[0].notes]
    assert velocities == sorted(velocities)
    assert velocities[0] == 40
    out = apply_crescendo(_midi(), 20, 100)
    assert out.instruments[0].notes[0].velocity == 20


@pytest.mark.parametrize("param", ["min_pitch", "max_pitch", "min_p", "max_p"])
@pytest.mark.parametrize("value", BAD + [60.5])
def test_pitch_limit_rejects_invalid_pitch(param, value):
    with pytest.raises(ValueError):
        limit_pitch_range(_midi(), **{param: value})


def test_pitch_limit_rejects_inverted_range():
    with pytest.raises(ValueError, match="above"):
        limit_pitch_range(_midi(), min_pitch=80, max_pitch=40)


def test_pitch_limit_valid_range_still_filters():
    out = limit_pitch_range(_midi(), min_pitch=55, max_pitch=75)
    assert [n.pitch for n in out.instruments[0].notes] == [60, 70]
