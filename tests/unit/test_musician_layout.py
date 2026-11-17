import os
import sys
from unittest import mock

import numpy as np
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
