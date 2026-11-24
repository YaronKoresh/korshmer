import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

try:
    import korshmer.__main__ as main_module
    from korshmer.__main__ import MainWindow
except Exception as import_error:
    MainWindow = None
    MAIN_IMPORT_ERROR = str(import_error)
else:
    MAIN_IMPORT_ERROR = ""

from korshmer.services import file_management
from korshmer.ui.widgets.arranger.track_panel import TrackRow
from korshmer.utils import audio_host


@pytest.fixture
def window(tmp_path, monkeypatch):
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    folder = tmp_path / "saved_data"
    folder.mkdir()
    monkeypatch.setattr(main_module, "DATA_DIR", folder)
    monkeypatch.setattr(
        file_management, "INPUT_LIST_FILE", folder / "inputs.json"
    )
    monkeypatch.setattr(
        audio_host, "AUDIO_HOST_FILE", folder / "audio_host.json"
    )
    audio_host.reset_host_preference_cache()
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, name, staticmethod(lambda *a, **k: None)
        )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.Discard),
    )
    win = MainWindow()
    win.show()
    win.activateWindow()
    QTest.qWaitForWindowActive(win)
    arranger = win.arranger
    win.tabs.setCurrentWidget(arranger)
    model = arranger.model
    for number in range(4):
        track = model.add_track(f"Part {number + 1}")
        model.add_note(track.id, 60 + number, float(number), 0.5)
    arranger.track_panel.refresh()
    arranger.piano_roll.load_model()
    QApplication.processEvents()
    yield win
    win.close()
    QApplication.processEvents()


def rows(win):
    return win.arranger.track_panel.findChildren(TrackRow)


def click(row, modifiers=Qt.NoModifier):
    QTest.mouseClick(
        row, Qt.LeftButton, modifiers, QPoint(2, row.height() // 2)
    )


def note_count(win):
    return len(win.arranger.model.notes)


def test_delete_in_the_track_list_removes_the_chosen_tracks_not_the_selected_notes(
    window,
):
    window.arranger.piano_roll.select_all()
    click(rows(window)[0])
    click(rows(window)[1], Qt.ControlModifier)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Delete)

    assert [t.name for t in window.arranger.model.tracks.values()] == [
        "Part 3",
        "Part 4",
    ]
    assert note_count(window) == 2


def test_control_a_in_the_track_list_selects_tracks_not_notes(window):
    click(rows(window)[0])

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_A, Qt.ControlModifier)

    assert len(window.arranger.track_panel.selected_track_ids()) == 4
    assert window.arranger.piano_roll.scene.selectedItems() == []


def test_one_undo_brings_back_tracks_deleted_together_from_the_list(window):
    click(rows(window)[0])
    click(rows(window)[2], Qt.ShiftModifier)
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Delete)
    assert len(window.arranger.model.tracks) == 1

    window.arranger.undo()

    assert len(window.arranger.model.tracks) == 4
    assert note_count(window) == 4


def test_delete_in_the_editor_still_deletes_the_selected_notes(window):
    roll = window.arranger.piano_roll
    roll.setFocus()
    roll.select_all()

    QTest.keyClick(roll, Qt.Key_Delete)

    assert len(window.arranger.model.tracks) == 4
    assert note_count(window) == 0


def test_escape_in_the_track_list_clears_the_chosen_tracks_and_keeps_playing(
    window, monkeypatch
):
    stops = []
    monkeypatch.setattr(window, "_handle_escape", lambda: stops.append(1))
    window._setup_keyboard_shortcuts()
    click(rows(window)[0])
    click(rows(window)[1], Qt.ControlModifier)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Escape)

    assert window.arranger.track_panel.selected_track_ids() == []
    assert stops == []


def test_m_in_the_track_list_mutes_every_chosen_track(window):
    click(rows(window)[0])
    click(rows(window)[3], Qt.ShiftModifier)

    QTest.keyClick(QApplication.focusWidget(), Qt.Key_M)

    assert all(t.muted for t in window.arranger.model.tracks.values())
