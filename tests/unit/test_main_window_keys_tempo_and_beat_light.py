import json
import os
import sys

import pretty_midi
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QKeySequenceEdit, QMessageBox

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
from korshmer.ui.theming.color_utils import BEAT_INDICATOR_OFF
from korshmer.utils import audio_host


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
        win.show()
        win.activateWindow()
        QTest.qWaitForWindowActive(win)
        win.setFocus()
        return win

    yield build
    for win in opened:
        win.close()
    QApplication.processEvents()


@pytest.fixture
def window(open_window):
    return open_window()


def press(key, modifiers=Qt.NoModifier):
    QTest.keyClick(QApplication.focusWidget(), key, modifiers)


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


def focus_category_list(window, row):
    window.tabs.setCurrentWidget(window.page_tools)
    window.list_categories.setFocus()
    window.list_categories.setCurrentRow(row)


def help_keys(window):
    own = window.shortcut_help_text().split("Inside the piano roll")[0]
    return {
        line.strip().split(" - ")[0]
        for line in own.splitlines()
        if line.startswith("  ") and " - " in line
    }


def bound_keys(window):
    return {
        action.shortcut().toString(QKeySequence.NativeText)
        for action in window.actions()
        if not action.shortcut().isEmpty()
    }


def count_beat_light_restyles(window):
    calls = []
    original = window.beat_indicator.setStyleSheet

    def record(sheet):
        calls.append(sheet)
        original(sheet)

    window.beat_indicator.setStyleSheet = record
    return calls


class FakeLink:
    enabled = True

    def __init__(self):
        self.tempos = []

    def publish_tempo(self, bpm):
        self.tempos.append(bpm)

    def stop(self):
        self.enabled = False


def test_pressing_ctrl_s_saves_the_song_like_the_menu_entry(
    window, monkeypatch
):
    saves = record_calls(monkeypatch, window, "save_project_action")
    press(Qt.Key_S, Qt.ControlModifier)
    assert len(saves) == 1


def test_changing_the_tempo_box_announces_the_new_tempo_once(window):
    link = FakeLink()
    window.link_sync = link
    window.spin_bpm.setValue(133.0)
    assert link.tempos == [133.0]


def test_the_beat_light_rests_until_a_song_plays(window):
    assert not window.beat_timer.isActive()


def test_safeguard_the_beat_light_blinks_while_a_song_plays(window):
    window.control_bar.player.playbackStateChanged.emit(1)
    assert window.beat_timer.isActive()


def test_the_beat_light_stops_blinking_when_the_song_stops(window):
    window.control_bar.player.playbackStateChanged.emit(1)
    window.control_bar.player.playbackStateChanged.emit(0)
    assert not window.beat_timer.isActive()


def test_changing_the_tempo_while_stopped_leaves_the_beat_light_resting(window):
    window.spin_bpm.setValue(98.0)
    assert not window.beat_timer.isActive()


def test_safeguard_the_beat_light_follows_a_tempo_change_during_playback(
    window,
):
    window.control_bar.player.playbackStateChanged.emit(1)
    window.spin_bpm.setValue(60.0)
    assert window.beat_timer.interval() == 1000


def test_the_beat_light_is_not_restyled_while_it_is_already_lit(window):
    window.control_bar.player.playbackStateChanged.emit(1)
    window.beat_timer.timeout.emit()
    restyles = count_beat_light_restyles(window)
    window.beat_timer.timeout.emit()
    assert restyles == []


def test_the_beat_light_goes_dark_when_the_song_stops_mid_flash(window):
    window.control_bar.player.playbackStateChanged.emit(1)
    window.beat_timer.timeout.emit()
    window.control_bar.player.playbackStateChanged.emit(0)
    assert window.beat_indicator.styleSheet() == BEAT_INDICATOR_OFF


def test_escape_stops_playback(window, monkeypatch):
    stops = record_calls(monkeypatch, window.control_bar.player, "stop")
    press(Qt.Key_Escape)
    assert len(stops) == 1


def test_escape_clears_selected_notes_before_it_stops_playback(
    window, monkeypatch, tmp_path
):
    window.arranger.load_midi_file(write_song(tmp_path), replace=True)
    window.arranger.select_all()
    stops = record_calls(monkeypatch, window.control_bar.player, "stop")
    press(Qt.Key_Escape)
    selected = window.arranger.piano_roll.scene.selectedItems()
    assert (selected, stops) == ([], [])


def test_home_moves_the_playhead_to_the_start(window, monkeypatch):
    moves = record_calls(monkeypatch, window.control_bar.player, "setPosition")
    press(Qt.Key_Home)
    assert moves == [(0,)]


def test_ctrl_o_opens_a_song(window, monkeypatch):
    calls = record_calls(monkeypatch, window, "load_project_action")
    press(Qt.Key_O, Qt.ControlModifier)
    assert len(calls) == 1


def test_ctrl_n_starts_a_new_song(window, monkeypatch):
    calls = record_calls(monkeypatch, window, "new_project_action")
    press(Qt.Key_N, Qt.ControlModifier)
    assert len(calls) == 1


def test_ctrl_e_exports_the_editor_mix(window, monkeypatch):
    calls = record_calls(monkeypatch, window, "export_editor_mix_action")
    press(Qt.Key_E, Qt.ControlModifier)
    assert len(calls) == 1


def test_a_letter_typed_in_a_focused_list_does_not_tap_the_tempo(
    window, monkeypatch
):
    taps = record_calls(monkeypatch, window, "on_tap_tempo")
    focus_category_list(window, 0)
    press(Qt.Key_T)
    assert taps == []


def test_safeguard_home_in_a_focused_list_moves_the_list_not_the_playhead(
    window, monkeypatch
):
    moves = record_calls(monkeypatch, window.control_bar.player, "setPosition")
    focus_category_list(window, 2)
    press(Qt.Key_Home)
    assert (window.list_categories.currentRow(), moves) == (0, [])


def test_safeguard_home_on_a_focused_slider_moves_the_slider_not_the_playhead(
    window, monkeypatch
):
    moves = record_calls(monkeypatch, window.control_bar.player, "setPosition")
    slider = window.control_bar.slider_seek
    slider.setRange(0, 1000)
    slider.setValue(500)
    slider.setFocus()
    press(Qt.Key_Home)
    assert (slider.value(), moves) == (0, [])


def test_safeguard_home_in_a_focused_combo_box_picks_its_first_entry(
    window, monkeypatch
):
    moves = record_calls(monkeypatch, window.control_bar.player, "setPosition")
    window.tabs.setCurrentIndex(1)
    window.combo_scale.setCurrentIndex(3)
    window.combo_scale.setFocus()
    press(Qt.Key_Home)
    assert (window.combo_scale.currentIndex(), moves) == (0, [])


def test_safeguard_escape_while_typing_in_the_tool_finder_keeps_playing(
    window, monkeypatch
):
    stops = record_calls(monkeypatch, window.control_bar.player, "stop")
    window.focus_tool_finder()
    press(Qt.Key_Escape)
    assert stops == []


def test_safeguard_ctrl_e_while_typing_in_the_tool_finder_exports_nothing(
    window, monkeypatch
):
    exports = record_calls(monkeypatch, window, "export_editor_mix_action")
    window.focus_tool_finder()
    press(Qt.Key_E, Qt.ControlModifier)
    assert exports == []


def test_safeguard_letters_typed_in_the_tool_finder_end_up_in_the_text(
    window, monkeypatch
):
    taps = record_calls(monkeypatch, window, "on_tap_tempo")
    window.focus_tool_finder()
    press(Qt.Key_T)
    assert (window.edit_tool_search.text(), taps) == ("t", [])


def test_ctrl_s_still_saves_while_typing_in_the_tool_finder(
    window, monkeypatch
):
    saves = record_calls(monkeypatch, window, "save_project_action")
    window.focus_tool_finder()
    press(Qt.Key_S, Qt.ControlModifier)
    assert len(saves) == 1


def test_safeguard_space_still_plays_while_a_list_has_focus(
    window, monkeypatch
):
    plays = record_calls(monkeypatch, window, "_handle_play_pause")
    focus_category_list(window, 1)
    press(Qt.Key_Space)
    assert len(plays) == 1


def test_every_key_listed_in_the_help_is_really_bound(window):
    listed = help_keys(window)
    assert listed and listed <= bound_keys(window)


def test_every_bound_key_is_listed_in_the_help(window):
    assert bound_keys(window) <= help_keys(window)


def test_the_help_does_not_promise_keys_for_features_that_do_not_exist(window):
    text = window.shortcut_help_text()
    assert "Record" not in text and "Zoom Horizontal" not in text


def test_f1_shows_the_generated_shortcut_list(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda parent, title, text: shown.append(text)),
    )
    press(Qt.Key_F1)
    assert shown == [window.shortcut_help_text()]


def test_a_customised_key_is_still_in_use_after_a_restart(
    data_dir, open_window, monkeypatch
):
    (data_dir / "shortcuts_user.json").write_text(
        json.dumps({"save": "Ctrl+Alt+S"})
    )
    window = open_window()
    saves = record_calls(monkeypatch, window, "save_project_action")
    press(Qt.Key_S, Qt.ControlModifier | Qt.AltModifier)
    assert len(saves) == 1


def test_a_customised_key_appears_in_the_help(data_dir, open_window):
    (data_dir / "shortcuts_user.json").write_text(json.dumps({"stop": "F4"}))
    assert "  F4 - Stop" in open_window().shortcut_help_text()


def use_shortcut_dialog(monkeypatch, typed, reset=False):
    from korshmer.ui.widgets.dialogs.shortcuts import ShortcutsDialog

    seen = []

    def answer(dialog):
        seen.append(dialog)
        if reset:
            dialog.btn_reset.click()
        for action, keys in typed.items():
            dialog.findChild(QKeySequenceEdit, action).setKeySequence(
                QKeySequence(keys)
            )
        dialog.btn_save.click()
        return dialog.result()

    monkeypatch.setattr(ShortcutsDialog, "exec", answer)
    return seen


def test_the_help_menu_offers_to_customise_the_shortcuts(window):
    help_menu = window.title_bar.menu_bar.actions()[1].menu()
    assert "Customise Shortcuts…" in [a.text() for a in help_menu.actions()]


def test_a_key_changed_in_the_shortcut_dialog_works_at_once(
    window, monkeypatch
):
    use_shortcut_dialog(monkeypatch, {"stop": "F4"})
    window.customise_shortcuts_action()
    stops = record_calls(monkeypatch, window.control_bar.player, "stop")
    window.setFocus()
    press(Qt.Key_F4)
    assert len(stops) == 1


def test_a_key_changed_in_the_shortcut_dialog_frees_the_old_key(
    window, monkeypatch
):
    use_shortcut_dialog(monkeypatch, {"stop": "F4"})
    window.customise_shortcuts_action()
    stops = record_calls(monkeypatch, window.control_bar.player, "stop")
    window.setFocus()
    press(Qt.Key_Escape)
    assert stops == []


def test_a_key_changed_in_the_shortcut_dialog_is_saved_for_the_next_start(
    data_dir, window, monkeypatch
):
    use_shortcut_dialog(monkeypatch, {"stop": "F4"})
    window.customise_shortcuts_action()
    saved = json.loads((data_dir / "shortcuts_user.json").read_text())
    assert saved["stop"] == "F4"


def test_the_shortcut_dialog_refuses_a_key_that_another_action_uses(
    data_dir, window, monkeypatch
):
    seen = use_shortcut_dialog(monkeypatch, {"stop": "Space"})
    window.customise_shortcuts_action()
    assert "Play" in seen[0].lbl_problem.text()
    assert not (data_dir / "shortcuts_user.json").exists()


def test_the_shortcut_dialog_can_put_the_default_keys_back(
    data_dir, open_window, monkeypatch
):
    (data_dir / "shortcuts_user.json").write_text(json.dumps({"stop": "F4"}))
    window = open_window()
    use_shortcut_dialog(monkeypatch, {}, reset=True)
    window.customise_shortcuts_action()
    assert "  Esc - Stop" in window.shortcut_help_text()
