import json
import os
import sys

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl, Signal
from PySide6.QtGui import QDropEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

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

from korshmer.mixins import ui as ui_module
from korshmer.services import file_management
from korshmer.utils import audio_host

LAYOUT_FILE = "window_layout.json"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
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
    return folder


@pytest.fixture
def open_window(data_dir):
    opened = []

    def build():
        win = MainWindow()
        opened.append(win)
        return win

    yield build
    for win in opened:
        win.close()
    QApplication.processEvents()


@pytest.fixture
def window(open_window):
    return open_window()


def save_layout(data_dir, layout):
    (data_dir / LAYOUT_FILE).write_text(json.dumps(layout))


def saved_layout(data_dir):
    return json.loads((data_dir / LAYOUT_FILE).read_text())


def remembering_window(open_window):
    window = open_window()
    window.restore_window_layout()
    window.show()
    QApplication.processEvents()
    return window


def record_calls(monkeypatch, target, name):
    calls = []
    monkeypatch.setattr(target, name, lambda *a, **k: calls.append(a))
    return calls


def write_song(folder):
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    keys = pretty_midi.Instrument(program=0, name="Keys")
    for i in range(4):
        keys.notes.append(pretty_midi.Note(90, 60 + i, i * 0.5, i * 0.5 + 0.4))
    pm.instruments.append(keys)
    path = folder / "song.mid"
    pm.write(str(path))
    return str(path)


def write_tone(folder, name="tone.wav"):
    path = folder / name
    t = np.arange(4410) / 44100
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 440 * t)), 44100)
    return str(path)


class BusyThread:
    def __init__(self):
        self.running = True
        self.stopped = False
        self.waited = []

    def isRunning(self):
        return self.running

    def stop(self):
        self.stopped = True
        self.running = False

    def requestInterruption(self):
        self.running = False

    def wait(self, timeout_ms=None):
        self.waited.append(timeout_ms)
        return True

    def terminate(self):
        self.running = False


def test_closing_the_window_saves_tab_splitter_and_size(data_dir, open_window):
    window = remembering_window(open_window)
    window.setGeometry(40, 30, 1000, 720)
    window.tabs.setCurrentIndex(2)
    QApplication.processEvents()
    window.splitter.setSizes([400, 590])
    sizes = window.splitter.sizes()
    window.close()
    saved = saved_layout(data_dir)
    assert (saved["geometry"], saved["tab"], saved["splitter"]) == (
        [40, 30, 1000, 720],
        2,
        sizes,
    )


def test_a_collapsed_side_panel_is_saved_as_collapsed(data_dir, open_window):
    window = remembering_window(open_window)
    window.splitter.setCollapsible(0, True)
    window.splitter.setSizes([0, 900])
    window.close()
    assert saved_layout(data_dir)["splitter"][0] == 0


def test_a_collapsed_side_panel_is_restored_as_collapsed(data_dir, open_window):
    save_layout(data_dir, {"splitter": [0, 900]})
    window = open_window()
    window.splitter.setCollapsible(0, True)
    window.restore_window_layout()
    window.show()
    QApplication.processEvents()
    assert window.splitter.sizes()[0] == 0


def test_a_fullscreen_window_is_saved_as_fullscreen_not_maximized(
    data_dir, open_window
):
    window = remembering_window(open_window)
    window.showFullScreen()
    QApplication.processEvents()
    window.close()
    saved = saved_layout(data_dir)
    assert (saved["fullscreen"], saved["maximized"]) == (True, False)


def test_a_maximized_window_is_saved_as_maximized_not_fullscreen(
    data_dir, open_window
):
    window = remembering_window(open_window)
    window.showMaximized()
    QApplication.processEvents()
    window.close()
    saved = saved_layout(data_dir)
    assert (saved["fullscreen"], saved["maximized"]) == (False, True)


def test_a_window_closed_in_fullscreen_opens_in_fullscreen(
    data_dir, open_window
):
    save_layout(data_dir, {"fullscreen": True, "maximized": False})
    window = remembering_window(open_window)
    assert window.isFullScreen() and not window.isMaximized()


def test_a_window_closed_maximized_opens_maximized(data_dir, open_window):
    save_layout(data_dir, {"fullscreen": False, "maximized": True})
    window = remembering_window(open_window)
    assert window.isMaximized() and not window.isFullScreen()


def test_the_last_active_tab_is_restored_on_start(data_dir, open_window):
    save_layout(data_dir, {"tab": 3})
    window = open_window()
    window.restore_window_layout()
    assert window.tabs.currentIndex() == 3


def test_the_window_size_and_place_are_restored_on_start(data_dir, open_window):
    screen = QApplication.primaryScreen().availableGeometry()
    wanted = [screen.x() + 5, screen.y() + 5, 720, 520]
    save_layout(data_dir, {"geometry": wanted})
    window = open_window()
    window.setMinimumSize(100, 100)
    window.restore_window_layout()
    assert list(window.geometry().getRect()) == wanted


def test_a_window_saved_on_a_missing_monitor_opens_on_the_screen(
    data_dir, open_window
):
    save_layout(data_dir, {"geometry": [90000, 90000, 700, 500]})
    window = open_window()
    window.restore_window_layout()
    screen = QApplication.primaryScreen().availableGeometry()
    assert screen.intersects(window.geometry())


def test_the_splitter_position_is_restored_on_start(data_dir, open_window):
    first = remembering_window(open_window)
    total = sum(first.splitter.sizes())
    first.splitter.setSizes([total // 2, total - total // 2])
    remembered = first.splitter.sizes()
    first.close()
    second = remembering_window(open_window)
    assert second.splitter.sizes() == remembered


def test_a_damaged_layout_file_leaves_the_default_window(data_dir, open_window):
    (data_dir / LAYOUT_FILE).write_text("{broken")
    window = open_window()
    before = window.geometry()
    window.restore_window_layout()
    assert (window.geometry(), window.tabs.currentIndex()) == (before, 0)


def test_a_layout_with_wrong_value_types_leaves_the_default_window(
    data_dir, open_window
):
    save_layout(
        data_dir,
        {
            "geometry": "wide",
            "tab": "EXPORT",
            "splitter": [0, -5],
            "time_display": 7,
            "fullscreen": "yes",
            "maximized": 1,
        },
    )
    window = open_window()
    before = window.geometry()
    window.restore_window_layout()
    assert (
        window.geometry(),
        window.tabs.currentIndex(),
        window.isFullScreen(),
        window.isMaximized(),
    ) == (before, 0, False, False)


def test_the_chosen_time_format_is_saved_as_soon_as_it_is_clicked(
    data_dir, open_window
):
    window = remembering_window(open_window)
    QTest.mouseClick(window.control_bar.lbl_time, Qt.LeftButton)
    assert saved_layout(data_dir)["time_display"] == "millis"


def test_the_chosen_time_format_is_restored_on_start(data_dir, open_window):
    save_layout(data_dir, {"time_display": "bars"})
    window = open_window()
    window.restore_window_layout()
    assert window.control_bar.time_display_mode() == "bars"


def test_safeguard_a_window_that_was_not_asked_to_remember_leaves_the_saved_layout_alone(
    data_dir, open_window
):
    window = open_window()
    window.tabs.setCurrentIndex(2)
    QTest.mouseClick(window.control_bar.lbl_time, Qt.LeftButton)
    window.close()
    assert not (data_dir / LAYOUT_FILE).exists()


def test_starting_the_app_restores_the_layout_saved_by_the_last_session(
    data_dir, monkeypatch
):
    save_layout(data_dir, {"tab": 3, "time_display": "bars"})
    for name in (
        "install_crash_handler",
        "register_exit_cleanup",
        "load_plugins",
    ):
        monkeypatch.setattr(main_module, name, lambda: None)
    monkeypatch.setattr(QApplication, "exec", staticmethod(lambda: 0))
    font = QApplication.font()
    try:
        with pytest.raises(SystemExit):
            main_module.main()
        started = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, MainWindow) and widget.isVisible()
        ]
        state = [
            (win.tabs.currentIndex(), win.control_bar.time_display_mode())
            for win in started
        ]
        for win in started:
            win.close()
    finally:
        QApplication.setFont(font)
    assert state == [(3, "bars")]
    assert saved_layout(data_dir)["tab"] == 3


def test_the_bars_display_uses_the_song_tempo(window):
    window.control_bar.set_time_display_mode("bars")
    window.spin_bpm.setValue(60.0)
    window.bpm_update_timer.timeout.emit()
    window.control_bar.update_time_label(2000, 4000)
    assert window.control_bar.lbl_time.text() == "1.3.000 / 2.1.000"


def test_the_bars_display_counts_from_the_first_beat_of_the_song(window):
    window.control_bar.set_time_display_mode("bars")
    window.arranger.model.grid_offset = 0.5
    window.control_bar.update_time_label(1500, 2500)
    assert window.control_bar.lbl_time.text() == "1.3.000 / 2.1.000"


def test_the_bars_display_follows_the_tempo_changes_of_the_song(window):
    class Doubling:
        def beats_at(self, seconds):
            return seconds * 4.0

    window.control_bar.set_time_display_mode("bars")
    window.arranger.model.tempo_map = Doubling()
    window.control_bar.update_time_label(1000, 2000)
    shown = window.control_bar.lbl_time.text()
    window.arranger.model.tempo_map = None
    assert shown == "2.1.000 / 3.1.000"


def test_the_bars_display_follows_the_time_signature_of_the_editor(
    data_dir, open_window, monkeypatch
):
    timeline = ui_module.ArrangerTimeline
    if not hasattr(timeline, "time_signature_changed"):

        class SignedTimeline(timeline):
            time_signature_changed = Signal(int)

        monkeypatch.setattr(ui_module, "ArrangerTimeline", SignedTimeline)
    window = open_window()
    window.control_bar.set_time_display_mode("bars")
    window.arranger.time_signature_changed.emit(3)
    window.control_bar.update_time_label(1500, 3000)
    assert window.control_bar.lbl_time.text() == "2.1.000 / 3.1.000"


def test_the_automatic_backup_holds_the_notes_on_screen(
    window, monkeypatch, tmp_path
):
    held = []
    manager = window.project_manager
    monkeypatch.setattr(
        manager,
        "create_backup",
        lambda: (
            held.append(len(manager.current_project_data["editor"]["notes"]))
            or True
        ),
    )
    window.arranger.load_midi_file(write_song(tmp_path), replace=True)
    window.backup_timer.timeout.emit()
    assert held == [4]


def test_safeguard_no_backup_is_written_while_backups_are_switched_off(
    window, monkeypatch
):
    calls = record_calls(monkeypatch, window.project_manager, "create_backup")
    window.project_manager.backup_enabled = False
    window.backup_timer.timeout.emit()
    assert calls == []


@pytest.mark.parametrize("holder_name", ["clip_thread", "_action_worker"])
def test_closing_the_window_stops_a_running_background_job(window, holder_name):
    job = BusyThread()
    setattr(window, holder_name, job)
    window.close()
    assert job.stopped and job.waited


def test_safeguard_closing_the_window_still_stops_a_running_export(window):
    job = BusyThread()
    window.export_thread = job
    window.close()
    assert job.stopped and job.waited


def test_closing_the_window_stops_a_running_recording(window):
    recording = BusyThread()
    window.arranger.recorder = recording
    window.close()
    assert recording.stopped and window.arranger.recorder is None


def test_closing_the_window_cancels_running_clip_repairs(window, monkeypatch):
    cancelled = record_calls(monkeypatch, window.arranger, "cancel_clip_edits")
    waited = record_calls(monkeypatch, window.arranger, "wait_for_clip_edits")
    window.close()
    assert (len(cancelled), len(waited)) == (1, 1)


def drop_files(window, paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    event = QDropEvent(
        QPointF(QPoint(20, 20)),
        Qt.CopyAction,
        mime,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    window.dropEvent(event)
    analysis = window.visualizer.worker
    if analysis is not None:
        analysis.wait(5000)
    return event.isAccepted()


def test_an_audio_file_dropped_on_the_window_joins_the_song_list(
    window, tmp_path
):
    tone = write_tone(tmp_path)
    drop_files(window, [tone])
    assert window.file_list_paths == [os.path.realpath(tone)]


def test_a_midi_file_dropped_on_the_window_joins_the_song_list(
    window, tmp_path
):
    song = write_song(tmp_path)
    drop_files(window, [write_tone(tmp_path), song])
    assert os.path.realpath(song) in window.file_list_paths


def test_a_drop_with_music_files_is_accepted(window, tmp_path):
    assert drop_files(window, [write_tone(tmp_path)])


def test_safeguard_a_dropped_file_that_is_not_music_is_left_out(
    window, tmp_path
):
    notes = tmp_path / "notes.txt"
    notes.write_text("lyrics")
    drop_files(window, [str(notes)])
    assert window.file_list_paths == []


def count_cursor_lookups(monkeypatch, global_point):
    lookups = []

    def pos():
        lookups.append(1)
        return global_point

    monkeypatch.setattr(main_module.QCursor, "pos", staticmethod(pos))
    return lookups


def send_mouse_move(widget, local_point):
    event = QMouseEvent(
        QEvent.MouseMove,
        QPointF(local_point),
        QPointF(widget.mapToGlobal(local_point)),
        Qt.NoButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(widget, event)


@pytest.fixture
def other_window():
    other = QWidget()
    other.setMouseTracking(True)
    other.resize(300, 200)
    other.show()
    yield other
    other.close()


def test_mouse_moves_over_another_window_do_no_resize_work(
    window, other_window, monkeypatch
):
    window.show()
    lookups = count_cursor_lookups(
        monkeypatch, window.mapToGlobal(QPoint(2, 300))
    )
    for x in range(20):
        send_mouse_move(other_window, QPoint(10 + x, 50))
    assert lookups == []


def test_mouse_moves_over_another_window_do_not_change_the_main_cursor(
    window, other_window, monkeypatch
):
    window.show()
    count_cursor_lookups(monkeypatch, window.mapToGlobal(QPoint(2, 300)))
    send_mouse_move(other_window, QPoint(10, 50))
    assert window.cursor().shape() == Qt.ArrowCursor


def test_safeguard_the_left_window_edge_still_shows_the_resize_cursor(
    window, monkeypatch
):
    window.show()
    edge = QPoint(2, 300)
    count_cursor_lookups(monkeypatch, window.mapToGlobal(edge))
    send_mouse_move(window.centralWidget(), edge)
    assert window.cursor().shape() == Qt.SizeHorCursor


def test_safeguard_leaving_the_window_edge_restores_the_normal_cursor(
    window, monkeypatch
):
    window.show()
    count_cursor_lookups(monkeypatch, window.mapToGlobal(QPoint(2, 300)))
    send_mouse_move(window.centralWidget(), QPoint(2, 300))
    inside = QPoint(window.width() // 2, 300)
    count_cursor_lookups(monkeypatch, window.mapToGlobal(inside))
    send_mouse_move(window.centralWidget(), inside)
    assert window.cursor().shape() == Qt.ArrowCursor


def test_safeguard_dragging_the_right_edge_still_resizes_the_window(
    window, monkeypatch
):
    window.show()
    window.setGeometry(50, 50, 1000, 720)
    QApplication.processEvents()
    grip = QPoint(window.width() - 2, 300)
    QTest.mousePress(window.centralWidget(), Qt.LeftButton, Qt.NoModifier, grip)
    count_cursor_lookups(monkeypatch, window.mapToGlobal(grip) + QPoint(60, 0))
    send_mouse_move(window.centralWidget(), grip + QPoint(60, 0))
    QTest.mouseRelease(
        window.centralWidget(), Qt.LeftButton, Qt.NoModifier, grip
    )
    assert window.width() == 1060
