import os
import struct
import sys
import time
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QWidget,
)

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.audio_processing.stem_session import StemSession
from korshmer.mixins.actions import project_management as song_actions
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
    saved_song_in,
    with_found_clips,
)
from korshmer.mixins.actions.stems import StemsActionsMixin
from korshmer.mixins.pages.main import (
    SEPARATION_HINTS,
    VELOCITY_AMOUNT_NAMES,
    MainPagesMixin,
)
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.material.choice import ChoiceBar, SectionCard
from korshmer.ui.widgets.material.mode_tool import ModeToolCard, ToolMode


def _combo(*labels):
    combo = QComboBox()
    combo.addItems(list(labels))
    return combo


def test_a_choice_bar_shows_every_option_and_follows_the_combo():
    combo = _combo("Fast", "Balanced", "Best")
    bar = ChoiceBar(combo, SEPARATION_HINTS)
    assert bar.labels() == ["Fast", "Balanced", "Best"]
    assert bar.checked_label() == "Fast"
    combo.setCurrentText("Best")
    assert bar.checked_label() == "Best"
    assert bar.hint_label.text() == SEPARATION_HINTS["Best"]


def test_clicking_a_choice_changes_the_combo():
    combo = _combo("Fast", "Balanced", "Best")
    bar = ChoiceBar(combo)
    bar.buttons[1].click()
    assert combo.currentText() == "Balanced"
    assert bar.hint_label.isHidden()


def test_a_tool_mode_is_one_click_away():
    calls = []
    card = ModeToolCard(
        "Edit",
        [
            ToolMode("Transpose", lambda: calls.append("transpose")),
            ToolMode("Snap to scale", lambda: calls.append("snap")),
        ],
    )
    assert card.combo_mode.isHidden()
    card.mode_bar.buttons[1].click()
    card.button.click()
    assert calls == ["snap"]
    assert card.current_mode().label == "Snap to scale"


class _Pages(QWidget, MainPagesMixin):
    presets = {"Default": {}}

    def create_label_help(self, text, _tooltip):
        return QWidget()

    def create_checkbox_with_help(self, checkbox, _tooltip):
        return checkbox

    def update_editor_tempo(self, *_args):
        pass

    def on_tap_tempo(self):
        pass

    def apply_preset(self, *_args):
        pass

    def save_preset_dialog(self):
        pass

    def apply_preset_default(self):
        pass

    def run_json_export(self):
        pass


def _titles(page):
    return [card.title for card in page.findChildren(SectionCard)]


def test_the_theory_tab_is_grouped_the_way_musicians_think():
    host = _Pages()
    page = host._create_page_music()
    assert _titles(page) == [
        "Tempo",
        "Key",
        "Timing",
        "Dynamics",
        "Sync and remote control",
    ]
    assert host.root_bar.labels()[:3] == ["Auto", "C", "C#"]
    assert host.grid_bar.checked_label() == "1/16"
    assert host.spin_bpm.objectName() == "tempoBox"


def test_the_velocity_amount_explains_itself():
    host = _Pages()
    page = host._create_page_music()
    assert _titles(page)
    assert not host.slider_vel_val.isEnabled()
    host.velocity_bar.buttons[1].click()
    assert host.combo_velocity.currentText() == "Fixed"
    assert host.slider_vel_val.isEnabled()
    assert host.slider_vel_val.lbl_name.text() == VELOCITY_AMOUNT_NAMES["Fixed"]


def test_the_export_tab_has_sections_and_plain_choices():
    host = _Pages()
    page = host._create_page_export()
    assert _titles(page)[:3] == ["Preset", "MIDI files", "Stems"]
    assert host.separation_bar.checked_label() == "Balanced"
    assert host.rad_separate.isChecked() and not host.rad_merged.isChecked()
    assert not host.chk_wav32.isChecked()
    host.separation_bar.buttons[0].click()
    assert host.combo_separation.currentText() == "Fast"


class _Studio(ProjectManagementActionsMixin, StemsActionsMixin):
    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.current_session = None


def _converted_song(base):
    folder = base / "saved_data" / "papit" / "2026-11-17_10-00"
    (folder / "Stems").mkdir(parents=True)
    vocals = folder / "Stems" / "vocals.wav"
    sf.write(vocals, np.zeros(2205, dtype=np.float32), 22050)
    session = StemSession(
        str(base / "papit.mp3"), str(folder), stems={"vocals": str(vocals)}
    )
    session.save()
    return session


def _write_two_notes(studio):
    track = studio.arranger.model.add_track("Lead")
    studio.arranger.model.add_note(track.id, 60, 0.0, 0.5, 90)
    studio.arranger.model.add_note(track.id, 64, 0.5, 0.5, 70)
    studio.arranger.piano_roll.load_model()


def test_save_song_keeps_the_edit_inside_the_converted_song_folder(tmp_path):
    studio = _Studio(tmp_path)
    studio.current_session = _converted_song(tmp_path)
    _write_two_notes(studio)
    with mock.patch.object(QFileDialog, "getSaveFileName") as dialog:
        assert studio.save_project_action() is True
    dialog.assert_not_called()
    saved = saved_song_in(studio.current_session.output_dir)
    assert saved is not None and saved.name == "papit.kproj"
    assert not any((tmp_path / "saved_data" / "projects").iterdir())


def test_open_song_brings_back_the_notes_and_the_stems(tmp_path):
    studio = _Studio(tmp_path)
    studio.current_session = _converted_song(tmp_path)
    _write_two_notes(studio)
    studio.save_project_action()
    saved = saved_song_in(studio.current_session.output_dir)

    later = _Studio(tmp_path)
    assert later.open_song_file(saved) is True
    pitches = sorted(n.pitch for n in later.arranger.model.notes)
    assert pitches == [60, 64]
    assert later.current_session.output_dir == os.path.dirname(str(saved))
    assert later.project_manager.current_project_path == saved


def test_a_new_song_gets_its_own_folder_beside_converted_songs(tmp_path):
    studio = _Studio(tmp_path)
    with mock.patch.object(
        song_actions, "get_limited_text", return_value=("Night Drive", True)
    ):
        studio.new_project_action()
    expected = tmp_path / "saved_data" / "Night Drive" / "Night Drive.kproj"
    assert studio.project_manager.current_project_path == expected
    assert expected.parent.is_dir()
    assert not (tmp_path / "saved_data" / "projects" / "Night Drive").exists()
    assert studio.save_project_action() is True
    assert expected.is_file()


def test_a_song_cannot_take_the_name_of_an_internal_folder(tmp_path):
    manager = ProjectManager(tmp_path)
    assert manager.song_path_for("Backups") is None
    assert manager.song_path_for("../escape") is None
    assert manager.start_song("projects") is None


def test_loading_a_song_folder_offers_the_saved_edit(tmp_path):
    studio = _Studio(tmp_path)
    studio.current_session = _converted_song(tmp_path)
    _write_two_notes(studio)
    studio.save_project_action()
    folder = studio.current_session.output_dir

    later = _Studio(tmp_path)
    with mock.patch.object(
        QMessageBox, "question", return_value=QMessageBox.Yes
    ):
        assert later.load_output_folder_session(folder) is True
    assert len(later.arranger.model.notes) == 2


def test_a_moved_song_folder_still_finds_its_audio(tmp_path):
    (tmp_path / "Stems").mkdir()
    moved = tmp_path / "Stems" / "vocals.wav"
    moved.write_bytes(b"x")
    editor = {"clips": [{"path": "/old/place/Stems/vocals.wav", "name": "V"}]}
    found = with_found_clips(editor, tmp_path)
    assert found["clips"][0]["path"] == str(moved)
    assert editor["clips"][0]["path"] == "/old/place/Stems/vocals.wav"


def test_editor_buttons_wrap_instead_of_overlapping():
    arranger = ArrangerTimeline()
    arranger.resize(520, 600)
    arranger.show()
    QApplication.processEvents()
    names = [name for name in vars(arranger) if name.startswith("btn_")]
    shown = [
        getattr(arranger, name)
        for name in names
        if getattr(arranger, name).isVisible()
    ]
    assert len(shown) >= 15
    for index, first in enumerate(shown):
        assert first.geometry().right() <= arranger.width()
        for second in shown[index + 1 :]:
            if first.parent() is second.parent():
                assert not first.geometry().intersects(second.geometry())
    arranger.close()


def _toolbar_rows(arranger):
    names = [name for name in vars(arranger) if name.startswith("btn_")]
    tops = {
        getattr(arranger, name).geometry().top()
        for name in names
        if getattr(arranger, name).isVisible()
    }
    return len(tops)


def test_editor_buttons_share_rows_without_dead_space():
    arranger = ArrangerTimeline()
    arranger.resize(1000, 600)
    arranger.show()
    QApplication.processEvents()
    assert _toolbar_rows(arranger) <= 2
    arranger.resize(520, 600)
    QApplication.processEvents()
    assert _toolbar_rows(arranger) <= 4
    arranger.close()


def test_a_level_box_always_shows_its_percentage():
    from PySide6.QtWidgets import QAbstractSpinBox

    from korshmer.ui.widgets.arranger.track_panel import make_level_box

    box = make_level_box(85, 200, lambda _value: None)
    assert box.buttonSymbols() == QAbstractSpinBox.NoButtons
    assert box.text() == "85%"
    needed = box.fontMetrics().horizontalAdvance("200%")
    assert box.width() >= needed + 12


RANGES = {0: (12, 108), 33: (0, 84)}


@pytest.fixture
def instrument_ranges(monkeypatch):
    from korshmer.ui.widgets.arranger import track_panel
    from korshmer.ui.widgets.midi import piano_roll

    def fake(program, is_drum=False, path=None):
        return (27, 87) if is_drum else RANGES.get(program, (0, 127))

    monkeypatch.setattr(piano_roll, "playable_range", fake)
    monkeypatch.setattr(track_panel, "playable_range", fake)
    return fake


def test_the_editor_opens_in_the_middle_of_the_keyboard():
    arranger = ArrangerTimeline()
    arranger.resize(900, 600)
    arranger.show()
    QApplication.processEvents()
    roll = arranger.piano_roll
    centre = roll.mapToScene(roll.viewport().rect().center())
    assert 48 <= roll.vm.pitch_from_y(centre.y()) <= 84
    arranger.close()


def test_the_shaded_rows_follow_the_instrument_of_the_track(instrument_ranges):
    arranger = ArrangerTimeline()
    roll = arranger.piano_roll
    piano = arranger.model.add_track("Lead", program=0)
    bass = arranger.model.add_track("Bass", program=33)
    roll.current_track_id = piano.id
    (_, top_to), (low_from, _) = roll.silent_row_spans()
    assert roll.vm.pitch_from_y(top_to - 1) == 109
    assert roll.vm.pitch_from_y(low_from) == 11
    roll.current_track_id = bass.id
    (_, top_to), (low_from, _) = roll.silent_row_spans()
    assert roll.vm.pitch_from_y(top_to - 1) == 85
    assert low_from == 128 * roll.vm.grid_size


def test_a_note_the_instrument_cannot_play_says_so(instrument_ranges):
    from PySide6.QtCore import QPointF

    arranger = ArrangerTimeline()
    roll = arranger.piano_roll
    bass = arranger.model.add_track("Bass", program=33)
    roll.current_track_id = bass.id
    heard = []
    roll.range_warning.connect(heard.append)
    roll._create_note_at(QPointF(100.0, roll.vm.y_from_pitch(60) + 1))
    assert heard == []
    roll._create_note_at(QPointF(200.0, roll.vm.y_from_pitch(96) + 1))
    assert len(heard) == 1
    assert "C-1 to C6" in heard[0] and "silent" in heard[0]


def test_silent_notes_are_listed_per_track_with_its_range(instrument_ranges):
    arranger = ArrangerTimeline()
    model = arranger.model
    lead = model.add_track("Lead", program=0)
    bass = model.add_track("Bass", program=33)
    drums = model.add_track("Drums", is_drum=True)
    model.add_note(lead.id, 100, 0.0, 0.5)
    model.add_note(lead.id, 118, 1.0, 0.5)
    model.add_note(bass.id, 100, 2.0, 0.5)
    model.add_note(drums.id, 40, 0.0, 0.5)
    assert model.silent_notes(instrument_ranges) == {lead.id: 1, bass.id: 1}
    arranger.track_panel.update_status()
    shown = arranger.track_panel.lbl_status.text()
    assert "1 notes on Lead" in shown and "(C0 to C8)" in shown
    assert "1 notes on Bass" in shown and "(C-1 to C6)" in shown
    model.clear()
    arranger.track_panel.update_status()
    assert "outside" not in arranger.track_panel.lbl_status.text()


def _chunk(name, data):
    return name + struct.pack("<I", len(data)) + data + b"\0" * (len(data) & 1)


def _tiny_soundfont(path):
    def gens(*items):
        return b"".join(struct.pack("<HBB", *item) for item in items)

    name = b"x".ljust(20, b"\0")
    phdr = name + struct.pack("<HHHIII", 5, 0, 0, 0, 0, 0)
    phdr += name + struct.pack("<HHHIII", 0, 0, 1, 0, 0, 0)
    pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0)
    pgen = gens((41, 0, 0), (0, 0, 0))
    inst = name + struct.pack("<H", 0) + name + struct.pack("<H", 2)
    ibag = b"".join(struct.pack("<HH", n, 0) for n in (0, 2, 4))
    igen = gens((43, 40, 59), (53, 0, 0), (43, 60, 96), (53, 1, 0), (0, 0, 0))
    hydra = b"pdta" + b"".join(
        _chunk(tag, data)
        for tag, data in (
            (b"phdr", phdr),
            (b"pbag", pbag),
            (b"pgen", pgen),
            (b"inst", inst),
            (b"ibag", ibag),
            (b"igen", igen),
        )
    )
    body = b"sfbk" + _chunk(b"LIST", hydra)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_each_instrument_range_is_read_from_the_soundfont(tmp_path):
    from korshmer.utils.soundfont_ranges import (
        FULL_RANGE,
        playable_range,
        range_text,
    )

    font = tmp_path / "tiny.sf2"
    _tiny_soundfont(font)
    assert playable_range(5, path=font) == (40, 96)
    assert playable_range(6, path=font) == FULL_RANGE
    assert playable_range(5, path=tmp_path / "missing.sf2") == FULL_RANGE
    assert range_text((40, 96)) == "E2 to C7"


def test_versions_are_listed_by_what_changed_not_by_file_name(tmp_path):
    studio = _Studio(tmp_path)
    studio.current_session = _converted_song(tmp_path)
    _write_two_notes(studio)
    studio.save_project_action()
    for words in ("Louder chorus drums", "", "Louder chorus drums"):
        with mock.patch.object(
            song_actions, "get_limited_text", return_value=(words, True)
        ):
            studio.create_snapshot_action()
        time.sleep(1.1)
    with mock.patch.object(
        song_actions.QInputDialog, "getItem", return_value=("", False)
    ) as dialog:
        studio.restore_snapshot_action()
    shown = dialog.call_args.args[3]
    assert len(shown) == len(set(shown)) == 3
    assert (
        sum(label.startswith("Louder chorus drums — ") for label in shown) == 2
    )
    assert sum(label.startswith("Version saved — ") for label in shown) == 1
    assert not any(
        ".kproj" in label or "_snapshot_" in label for label in shown
    )


def test_choosing_a_version_by_its_description_restores_it(tmp_path):
    studio = _Studio(tmp_path)
    studio.current_session = _converted_song(tmp_path)
    _write_two_notes(studio)
    studio.save_project_action()
    with mock.patch.object(
        song_actions, "get_limited_text", return_value=("Two notes", True)
    ):
        studio.create_snapshot_action()
    studio.arranger.clear_editor()

    def pick(_parent, _title, _label, items, *_rest):
        return items[0], True

    with mock.patch.object(song_actions.QInputDialog, "getItem", pick):
        studio.restore_snapshot_action()
    assert len(studio.arranger.model.notes) == 2


def test_each_level_number_sits_right_after_its_own_label():
    from korshmer.ui.widgets.audio.live_meters import PAIR_GAP, LiveMeters

    meters = LiveMeters()
    meters.resize(256, 40)
    meters.show()
    QApplication.processEvents()
    spots = {}
    for name, value in meters.values.items():
        title = meters.titles[name]
        assert title.parent() is value.parent()
        assert value.x() - (title.x() + title.width()) == PAIR_GAP
        cell = value.parent().geometry()
        spots[name] = (cell.x(), cell.x() + value.x() + value.width())
    row = sorted(
        spots[n] for n in ("momentary_lufs", "short_term_lufs", "rms_db")
    )
    for (_, value_end), (next_start, _) in zip(row, row[1:], strict=False):
        assert next_start - value_end > PAIR_GAP
    meters.close()


def test_the_tutorial_opens_the_tab_each_step_needs():
    from korshmer.ui.managers.tutorial import _intro_steps

    tabs = {step.title: step.tab for step in _intro_steps()}
    assert tabs["The editor"] == "EDITOR"
    assert tabs["Set the tempo"] == "THEORY"
    assert tabs["Find any tool"] == "EFFECTS"
    targets = {step.title: step.target_widget_name for step in _intro_steps()}
    assert targets["Tempo and key"] == "tabs.tabBar"
    assert targets["Effects and tools"] == "tabs.tabBar"
