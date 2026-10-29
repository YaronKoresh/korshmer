import pretty_midi
import pytest

from korshmer import midi_algorithms
from korshmer.midi_processors import (
    process_scale_detection,
)

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]


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


def _scale_song(tmp_path):
    notes = [(p, i * 0.5, i * 0.5 + 0.4) for i, p in enumerate(C_MAJOR)]
    return _write(tmp_path / "song.mid", notes)


def test_scale_detection_writes_a_report_with_the_scale_name(tmp_path):
    out = process_scale_detection(_scale_song(tmp_path))
    text = open(out).read()
    assert out.endswith("song_ScaleAnalysis.txt")
    assert text.startswith("Scale Analysis for: song")
    assert "C Major" in text


def test_scale_detection_ignores_drums(tmp_path):
    notes = [(36 + i, i * 0.25, i * 0.25 + 0.2) for i in range(12)]
    path = _write(tmp_path / "drums.mid", notes, is_drum=True)
    text = open(process_scale_detection(path)).read()
    assert "No matching scale found." in text


def test_scale_detection_of_a_nearly_empty_file(tmp_path):
    path = _write(tmp_path / "two.mid", [(60, 0.0, 0.5), (62, 0.5, 1.0)])
    assert (
        "No matching scale found." in open(process_scale_detection(path)).read()
    )
