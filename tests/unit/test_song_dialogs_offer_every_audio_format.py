import os
import sys
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.event_bus import Event
from korshmer.mixins import files as files_module
from korshmer.mixins.actions import audio as audio_module, stems as stems_module
from korshmer.mixins.actions.audio import AudioActionsMixin
from korshmer.mixins.actions.core import CoreActionsMixin
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.mixins.actions.stems import StemsActionsMixin
from korshmer.mixins.files import FilesMixin
from korshmer.services.file_management import FileManagementService

ACCEPTED = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".au")
NEWLY_ACCEPTED = ["song.m4a", "song.aac", "song.au", "SONG.M4A"]


@pytest.fixture(autouse=True)
def accepted_formats(monkeypatch):
    monkeypatch.setattr(
        FileManagementService, "VALID_AUDIO_EXTENSIONS", ACCEPTED
    )
    monkeypatch.setattr(
        FileManagementService,
        "VALID_EXTENSIONS",
        ACCEPTED + FileManagementService.VALID_MIDI_EXTENSIONS,
    )


@pytest.fixture
def filters(monkeypatch):
    seen = []

    def open_names(parent, title, folder, file_filter):
        seen.append(file_filter)
        return [], ""

    def open_name(parent, title, folder, file_filter):
        seen.append(file_filter)
        return "", ""

    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", staticmethod(open_names)
    )
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(open_name))
    return seen


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2]))
    )
    return shown


def offered(file_filter):
    return set(file_filter.split("(")[1].rstrip(")").split())


def wanted():
    return {f"*{extension}" for extension in ACCEPTED}


class ImportHost(ProjectManagementActionsMixin):
    import_manager = mock.Mock(imports_dir="/imports")


class PlayHost(FilesMixin):
    def __init__(self):
        self.control_bar = mock.Mock()
        self.arranger = mock.Mock()


class FolderHost(CoreActionsMixin):
    def __init__(self):
        self.added = []

    def add_file(self, path):
        self.added.append(os.path.basename(path))


class SelectionHost:
    def __init__(self, path):
        self.path = path
        self.started = []
        self.is_processing = True

    def _get_selected_file(self, is_input=None):
        return self.path

    def _stem_thread_is_running(self):
        return False

    def _run_action(self, *args, **kwargs):
        self.started.append(args)


class StemHost(SelectionHost, StemsActionsMixin):
    on_stem_progress = None
    on_stems_session_ready = None
    on_stem_separation_failed = None


class ConvertHost(SelectionHost, AudioActionsMixin):
    conversions = 0


def test_the_add_files_dialog_offers_every_accepted_audio_format(filters):
    FilesMixin().browse_files()
    assert wanted() <= offered(filters[0])


def test_safeguard_the_add_files_dialog_still_offers_midi_files(filters):
    FilesMixin().browse_files()
    assert {"*.mid", "*.midi"} <= offered(filters[0])


def test_the_add_audio_clip_dialog_offers_every_accepted_audio_format(filters):
    EditorActionsMixin().editor_add_audio_clip_dialog()
    assert wanted() <= offered(filters[0])


def test_the_import_audio_dialog_offers_every_accepted_audio_format(filters):
    ImportHost().import_audio_files_action()
    assert wanted() <= offered(filters[0])


@pytest.mark.parametrize("name", NEWLY_ACCEPTED)
def test_double_clicking_any_accepted_audio_file_loads_it(
    name, tmp_path, monkeypatch
):
    loaded = []
    monkeypatch.setattr(
        files_module,
        "publish",
        lambda event, **data: (
            loaded.append(data.get("path"))
            if event == Event.AUDIO_LOAD_REQUESTED
            else None
        ),
    )
    path = str(tmp_path / name)
    PlayHost().on_play_click(path, is_output=True)
    assert loaded == [path]


def test_adding_a_folder_takes_every_accepted_audio_format(
    tmp_path, monkeypatch
):
    for name in ("a.m4a", "b.au", "c.aac", "d.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: str(tmp_path)),
    )
    host = FolderHost()
    with mock.patch("korshmer.mixins.actions.core.publish"):
        host.run_batch_folder_action()
    assert sorted(host.added) == ["a.m4a", "b.au", "c.aac", "d.wav"]


@pytest.mark.parametrize("name", NEWLY_ACCEPTED)
def test_stem_separation_accepts_every_accepted_audio_format(
    name, warnings, monkeypatch
):
    monkeypatch.setattr(stems_module, "publish", lambda *a, **k: None)
    separation = mock.MagicMock()
    monkeypatch.setattr(
        "korshmer.workers.stem_separation.StemSeparationThread", separation
    )
    StemHost(f"/music/{name}").run_stem_separation()
    assert (warnings, separation.call_args.args) == ([], (f"/music/{name}",))


@pytest.mark.parametrize("name", NEWLY_ACCEPTED)
def test_audio_to_midi_accepts_every_accepted_audio_format(
    name, warnings, monkeypatch
):
    monkeypatch.setattr(
        audio_module.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    ConvertHost(f"/music/{name}").run_audio_to_midi_conversion()
    assert warnings == []


def test_safeguard_a_text_file_is_still_refused_for_audio_to_midi(warnings):
    ConvertHost("/music/notes.txt").run_audio_to_midi_conversion()
    assert len(warnings) == 1
