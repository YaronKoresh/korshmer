import pretty_midi
import pytest

from korshmer import midi_algorithms
from korshmer.midi_processors import (
    process_advanced_quantize,
)


@pytest.fixture(autouse=True)
def processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", str(tmp_path / "data"))


def _write(path, notes, tempo=120.0, program=0, is_drum=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program, is_drum=is_drum)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(90, pitch, start, end))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def test_advanced_quantize_snaps_notes_to_the_grid(tmp_path):
    notes = [(60, 0.03, 0.2), (62, 0.27, 0.45), (64, 0.52, 0.7)]
    path = _write(tmp_path / "loose.mid", notes, tempo=120.0)
    out = process_advanced_quantize(path, 1.0, 4)
    starts = [n.start for n in pretty_midi.PrettyMIDI(out).instruments[0].notes]
    assert starts == pytest.approx([0.0, 0.25, 0.5], abs=1e-3)


def test_advanced_quantize_strength_blends_between_original_and_grid(tmp_path):
    path = _write(tmp_path / "loose.mid", [(60, 0.1, 0.2)], tempo=120.0)
    out = process_advanced_quantize(path, 0.5, 4)
    note = pretty_midi.PrettyMIDI(out).instruments[0].notes[0]
    assert note.start == pytest.approx(0.1125, abs=5e-3)


def test_advanced_quantize_rejects_invalid_strength(tmp_path):
    path = _write(tmp_path / "loose.mid", [(60, 0.1, 0.2)])
    with pytest.raises(ValueError):
        process_advanced_quantize(path, float("nan"), 4)
