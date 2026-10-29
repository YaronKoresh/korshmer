import io

import pretty_midi
import pytest

from korshmer.midi.transforms import extract_rhythm


def _midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(80, 60, 0.0, 1.0))
    inst.notes.append(pretty_midi.Note(80, 64, 1.0, 2.0))
    midi.instruments.append(inst)
    return midi


def _pitches(midi):
    return [n.pitch for n in midi.instruments[0].notes]


BAD = [
    128,
    -1,
    200,
    1.5,
    float("nan"),
    float("inf"),
    "60",
    None,
    True,
]


@pytest.mark.parametrize("value", BAD)
def test_extract_rhythm_rejects_invalid_pitch(value):
    midi = _midi()
    with pytest.raises(ValueError):
        extract_rhythm(midi, value)
    assert _pitches(midi) == [60, 64]


@pytest.mark.parametrize("value", [0, 36, 60, 127, 60.0])
def test_extract_rhythm_accepts_valid_pitch_and_saves(value):
    midi = extract_rhythm(_midi(), value)
    assert _pitches(midi) == [int(value)] * 2
    assert all(isinstance(p, int) for p in _pitches(midi))
    midi.write(io.BytesIO())


def test_extract_rhythm_skips_drums():
    midi = _midi()
    drum = pretty_midi.Instrument(0, is_drum=True)
    drum.notes.append(pretty_midi.Note(80, 38, 0.0, 0.5))
    midi.instruments.append(drum)
    extract_rhythm(midi, 72)
    assert [n.pitch for n in drum.notes] == [38]
