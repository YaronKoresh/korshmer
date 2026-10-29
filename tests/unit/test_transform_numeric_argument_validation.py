import pretty_midi
import pytest

from korshmer.midi.transforms import (
    apply_mirror,
    compress_velocity,
    double_octave,
    randomize_velocity,
    transpose_midi,
)


def _midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(80, 60, 0.0, 1.0))
    inst.notes.append(pretty_midi.Note(80, 64, 1.0, 2.0))
    midi.instruments.append(inst)
    return midi


def _pitches(midi):
    return [n.pitch for n in midi.instruments[0].notes]


BAD = [1.5, float("nan"), float("inf"), -float("inf"), "2", None, True]


@pytest.mark.parametrize("value", BAD)
def test_transpose_rejects_non_whole_semitones(value):
    midi = _midi()
    with pytest.raises(ValueError):
        transpose_midi(midi, value)
    assert _pitches(midi) == [60, 64]


@pytest.mark.parametrize("value", BAD)
def test_mirror_rejects_non_whole_center(value):
    midi = _midi()
    with pytest.raises(ValueError):
        apply_mirror(midi, value)
    assert _pitches(midi) == [60, 64]


@pytest.mark.parametrize(
    "value", [-5, 5.5, float("nan"), float("inf"), "5", True]
)
def test_randomize_rejects_invalid_amount(value):
    midi = _midi()
    with pytest.raises(ValueError):
        randomize_velocity(midi, amount=value)
    assert [n.velocity for n in midi.instruments[0].notes] == [80, 80]


def test_whole_valued_float_is_accepted_and_saves():
    midi = transpose_midi(_midi(), 2.0)
    assert _pitches(midi) == [62, 66]
    assert all(type(p) is int for p in _pitches(midi))
    midi = apply_mirror(_midi(), 60.0)
    assert _pitches(midi) == [60, 56]
    midi = randomize_velocity(_midi(), amount=0)
    assert [n.velocity for n in midi.instruments[0].notes] == [80, 80]


def _velocities(midi):
    return [n.velocity for n in midi.instruments[0].notes]


@pytest.mark.parametrize(
    "value", [0.5, float("nan"), float("inf"), "1", None, True]
)
def test_double_octave_rejects_non_whole_shift(value):
    midi = _midi()
    with pytest.raises(ValueError):
        double_octave(midi, value)
    assert _pitches(midi) == [60, 64]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), "2", True]
)
def test_compress_velocity_rejects_bad_ratio(value):
    midi = _midi()
    with pytest.raises(ValueError):
        compress_velocity(midi, ratio=value)
    assert _velocities(midi) == [80, 80]


def test_compress_velocity_accepts_valid_ratio():
    midi = _midi()
    compress_velocity(midi, ratio=0.5)
    assert _velocities(midi) == [72, 72]
