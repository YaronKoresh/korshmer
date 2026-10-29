import sys

import pretty_midi
import pytest
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.utils.tempo import midi_file_tempo


def _song(path, bpm):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(0)
    for k in range(4):
        inst.notes.append(pretty_midi.Note(80, 60 + k, k * 0.5, k * 0.5 + 0.4))
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def test_file_tempo_is_read_from_the_file(tmp_path):
    assert midi_file_tempo(_song(tmp_path / "a.mid", 90)) == pytest.approx(
        90, abs=0.01
    )


def test_file_without_tempo_or_unreadable_gives_none(tmp_path):
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not midi")
    assert midi_file_tempo(bad) is None
    assert midi_file_tempo(tmp_path / "missing.mid") is None


def test_loading_keeps_the_tempo_of_the_midi_file(tmp_path):
    arranger = ArrangerTimeline()
    heard = []
    arranger.tempo_changed.connect(heard.append)
    arranger.load_midi_file(_song(tmp_path / "s.mid", 90), tempo=120.0)
    assert arranger.piano_roll.tempo == pytest.approx(90, abs=0.01)
    assert heard[-1] == pytest.approx(90, abs=0.01)
    exported = arranger.get_midi_from_editor()
    assert exported.get_tempo_changes()[1][0] == pytest.approx(90, abs=0.01)


def test_session_loading_can_keep_the_given_tempo(tmp_path):
    arranger = ArrangerTimeline()
    arranger.load_midi_file(
        _song(tmp_path / "s.mid", 90), tempo=98.0, file_tempo=False
    )
    assert arranger.piano_roll.tempo == pytest.approx(98.0)
