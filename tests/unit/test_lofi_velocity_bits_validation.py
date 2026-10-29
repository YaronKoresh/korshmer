import pretty_midi
import pytest

from korshmer.midi.transforms import lofi_velocity


def _midi(velocity=100):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity, 60, 0.0, 1.0))
    midi.instruments.append(inst)
    return midi


@pytest.mark.parametrize(
    "bits",
    [2.5, float("nan"), float("inf"), "4", None, True, 0, -1, 8, 1e18],
)
def test_lofi_velocity_rejects_invalid_bits(bits):
    midi = _midi()
    with pytest.raises(ValueError, match="Bits"):
        lofi_velocity(midi, bits)
    assert midi.instruments[0].notes[0].velocity == 100


def test_lofi_velocity_default_bits_unchanged():
    assert lofi_velocity(_midi()).instruments[0].notes[0].velocity == 103


@pytest.mark.parametrize("bits", [1, 4, 7, 4.0])
def test_lofi_velocity_accepts_valid_bits(bits):
    velocity = lofi_velocity(_midi(), bits).instruments[0].notes[0].velocity
    assert 1 <= velocity <= 127
