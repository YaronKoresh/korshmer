import sys
from unittest import mock

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.audio_processing import vocal_melody
from korshmer.audio_processing.stem_session import StemSession
from korshmer.midi import transcription_cleanup as cleanup
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.mixins.actions.stems import StemsActionsMixin
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline


class _Studio(ProjectManagementActionsMixin, StemsActionsMixin):
    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.current_session = None


def _session(base):
    folder = base / "saved_data" / "papit" / "take"
    (folder / "Stems").mkdir(parents=True)
    vocals = folder / "Stems" / "vocals.wav"
    sf.write(vocals, np.zeros(2205, dtype=np.float32), 22050)
    session = StemSession(
        str(base / "papit.mp3"), str(folder), stems={"vocals": str(vocals)}
    )
    session.save()
    return session


def _add_note(studio, pitch=60):
    model = studio.arranger.model
    track = model.first_track_id()
    if track is None:
        track = model.add_track("Lead").id
    model.add_note(track, pitch, 0.0, 0.5, 90)
    studio.arranger.piano_roll.load_model()


def _ask(answer):
    return mock.patch.object(QMessageBox, "question", return_value=answer)


def test_an_empty_editor_closes_without_a_question(tmp_path):
    studio = _Studio(tmp_path)
    with _ask(QMessageBox.Cancel) as question:
        assert studio.confirm_unsaved_changes() is True
    question.assert_not_called()


def test_unsaved_notes_ask_before_closing(tmp_path):
    studio = _Studio(tmp_path)
    _add_note(studio)
    assert studio.has_unsaved_changes()
    with _ask(QMessageBox.Cancel):
        assert studio.confirm_unsaved_changes() is False
    with _ask(QMessageBox.Discard):
        assert studio.confirm_unsaved_changes() is True


def test_choosing_save_writes_the_song_and_then_allows_closing(tmp_path):
    studio = _Studio(tmp_path)
    studio.current_session = _session(tmp_path)
    _add_note(studio)
    with (
        _ask(QMessageBox.Save),
        mock.patch.object(QFileDialog, "getSaveFileName") as dialog,
    ):
        assert studio.confirm_unsaved_changes() is True
    dialog.assert_not_called()
    assert (
        tmp_path / "saved_data" / "papit" / "take" / "papit.kproj"
    ).is_file()
    assert not studio.has_unsaved_changes()
    _add_note(studio, 64)
    assert studio.has_unsaved_changes()


def test_a_cancelled_save_dialog_keeps_the_window_open(tmp_path):
    studio = _Studio(tmp_path)
    _add_note(studio)
    with (
        _ask(QMessageBox.Save),
        mock.patch.object(
            QFileDialog, "getSaveFileName", return_value=("", "")
        ),
    ):
        assert studio.confirm_unsaved_changes() is False


def test_a_freshly_loaded_conversion_is_not_called_unsaved(tmp_path):
    studio = _Studio(tmp_path)
    studio.load_session_into_editor(_session(tmp_path))
    assert studio.arranger.model.clips
    assert not studio.has_unsaved_changes()
    _add_note(studio)
    assert studio.has_unsaved_changes()


def _ringing_level(pitch, sounding_until, seconds=4.0, fps=50.0):
    level = np.full((cleanup.PITCH_FLUX_BINS, int(seconds * fps)), -70.0)
    row = pitch - cleanup.PITCH_FLUX_LOWEST
    level[row, : int(sounding_until * fps)] = -8.0
    return level, fps


def _one_voice(*notes):
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(80, p, a, b) for p, a, b in notes]
    return inst


def test_a_note_rings_for_as_long_as_its_pitch_keeps_sounding():
    level, fps = _ringing_level(60, 1.5)
    inst = _one_voice((60, 0.0, 0.2), (64, 0.0, 0.2))
    assert cleanup.extend_ringing_notes(inst, level, fps) == 1
    ends = {n.pitch: n.end for n in inst.notes}
    assert ends[60] == pytest.approx(1.5, abs=0.03)
    assert ends[64] == 0.2


def test_a_ringing_note_stops_before_the_next_note_of_the_same_pitch():
    level, fps = _ringing_level(60, 3.0)
    inst = _one_voice((60, 0.0, 0.2), (60, 1.0, 1.2))
    cleanup.extend_ringing_notes(inst, level, fps)
    first, second = sorted(inst.notes, key=lambda n: n.start)
    assert first.end == pytest.approx(1.0 - cleanup.RING_GAP_S)
    assert second.end <= 1.2 + cleanup.RING_MAX_EXTRA_S
    drums = pretty_midi.Instrument(0, is_drum=True)
    drums.notes = [pretty_midi.Note(80, 60, 0.0, 0.2)]
    assert cleanup.extend_ringing_notes(drums, level, fps) == 0


def test_a_very_short_blip_two_semitones_away_joins_its_neighbour():
    hop = 256 / 22050
    spans = [(0, 8, 58), (8, 70, 60), (70, 110, 65)]
    assert vocal_melody._absorb_scoops(spans, hop) == [
        (0, 70, 60),
        (70, 110, 65),
    ]
    far = [(0, 8, 56), (8, 70, 60)]
    assert vocal_melody._absorb_scoops(far, hop) == far
