import json

import pretty_midi
import pytest

from korshmer.midi import export_to_json
from korshmer.midi_processors import process_json_export


def _write(path, name="Lead", program=5, is_drum=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=110)
    inst = pretty_midi.Instrument(program, is_drum=is_drum, name=name)
    inst.notes.append(pretty_midi.Note(90, 60, 0.0, 1.0))
    inst.notes.append(pretty_midi.Note(70, 64, 1.0, 1.5))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def test_export_of_a_loaded_midi_file_is_valid_json(tmp_path):
    path = _write(tmp_path / "song.mid")
    data = json.loads(export_to_json(path))
    track = data["tracks"][0]
    assert track["name"] == "Lead"
    assert track["program"] == 5
    assert track["is_drum"] is False
    assert [n["pitch"] for n in track["notes"]] == [60, 64]
    assert [n["velocity"] for n in track["notes"]] == [90, 70]
    assert track["notes"][1]["start"] == pytest.approx(1.0, abs=0.01)


def test_export_values_are_plain_python_types(tmp_path):
    pm = pretty_midi.PrettyMIDI(_write(tmp_path / "song.mid"))
    data = json.loads(export_to_json(pm))
    track = data["tracks"][0]
    assert type(track["program"]) is int
    assert type(track["notes"][0]["start"]) is float


def test_export_writes_the_file_when_a_path_is_given(tmp_path):
    path = _write(tmp_path / "song.mid", is_drum=True)
    out = tmp_path / "out.json"
    assert export_to_json(path, str(out)) == str(out)
    assert json.loads(out.read_text())["tracks"][0]["is_drum"] is True


def test_tool_writes_a_json_file_next_to_a_mid_file(tmp_path):
    path = _write(tmp_path / "song.mid")
    out = process_json_export(path)
    assert out == str(tmp_path / "song.json")
    assert json.loads(open(out).read())["bpm"] == 120.0


def test_tool_names_the_json_after_a_midi_extension(tmp_path):
    path = _write(tmp_path / "song.midi")
    assert process_json_export(path) == str(tmp_path / "song.json")


def test_tool_keeps_mid_in_a_folder_name(tmp_path):
    folder = tmp_path / "my.mid.files"
    folder.mkdir()
    path = _write(folder / "song.mid")
    assert process_json_export(path) == str(folder / "song.json")
