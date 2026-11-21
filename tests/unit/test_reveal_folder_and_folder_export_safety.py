import os
from unittest import mock

import pytest

from korshmer.mixins import files as files_module
from korshmer.mixins.actions import core as core_module
from korshmer.mixins.actions.core import CoreActionsMixin
from korshmer.mixins.files import FilesMixin


class ExportHost(CoreActionsMixin):
    def __init__(self):
        self.spin_bpm = mock.Mock(value=mock.Mock(return_value=120))

    def get_selected_midi(self):
        return "/some/song.mid"


@pytest.fixture
def opener(monkeypatch):
    commands = []

    def call(command, *args, **kwargs):
        commands.append(list(command))
        return 0

    monkeypatch.setattr(files_module.subprocess, "call", call)
    return commands


@pytest.fixture
def no_opener(monkeypatch):
    def call(command, *args, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(files_module.sys, "platform", "linux")
    monkeypatch.setattr(files_module.subprocess, "call", call)


@pytest.fixture
def dialogs(monkeypatch):
    shown = {"warning": [], "critical": [], "information": []}
    for name, texts in shown.items():
        monkeypatch.setattr(
            files_module.QMessageBox,
            name,
            staticmethod(lambda *a, _texts=texts, **k: _texts.append(a[2])),
        )
    monkeypatch.setattr(
        files_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: files_module.QMessageBox.Yes),
    )
    return shown


@pytest.fixture
def status(monkeypatch):
    texts = []
    monkeypatch.setattr(
        files_module,
        "publish",
        lambda event, **data: texts.append(data.get("text", "")),
    )
    return texts


def choose_folder(monkeypatch, folder):
    monkeypatch.setattr(
        files_module.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: str(folder)),
    )


def song_folder(parent, name="My Song"):
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / "take.mid").write_bytes(b"MThd")
    return folder


def test_open_location_uses_xdg_open_on_linux(tmp_path, monkeypatch, opener):
    monkeypatch.setattr(files_module.sys, "platform", "linux")
    FilesMixin().open_file_location_action(str(tmp_path))
    assert opener == [["xdg-open", str(tmp_path)]]


def test_safeguard_open_location_uses_open_on_macos(
    tmp_path, monkeypatch, opener
):
    monkeypatch.setattr(files_module.sys, "platform", "darwin")
    FilesMixin().open_file_location_action(str(tmp_path))
    assert opener == [["open", str(tmp_path)]]


def test_safeguard_open_location_uses_the_shell_on_windows(
    tmp_path, monkeypatch, opener
):
    opened = []
    monkeypatch.setattr(files_module.sys, "platform", "win32")
    monkeypatch.setattr(
        files_module.os, "startfile", opened.append, raising=False
    )
    FilesMixin().open_file_location_action(str(tmp_path))
    assert (opened, opener) == ([str(tmp_path)], [])


def test_open_location_of_a_file_reveals_its_folder(
    tmp_path, monkeypatch, opener
):
    monkeypatch.setattr(files_module.sys, "platform", "linux")
    song = tmp_path / "song.mid"
    song.write_bytes(b"MThd")
    FilesMixin().open_file_location_action(str(song))
    assert opener == [["xdg-open", str(tmp_path)]]


def test_open_location_without_a_file_manager_tells_the_user_instead_of_failing(
    tmp_path, no_opener, status
):
    FilesMixin().open_file_location_action(str(tmp_path))
    assert len(status) == 1 and str(tmp_path) in status[0]


def test_a_finished_wav_export_is_not_reported_as_failed_when_the_folder_cannot_open(
    tmp_path, monkeypatch, no_opener
):
    saved = tmp_path / "saved_data"
    saved.mkdir()
    monkeypatch.setattr(core_module, "DATA_DIR", saved)
    monkeypatch.setattr(core_module, "SF3_PATH", str(tmp_path / "sf.sf3"))
    monkeypatch.setattr(core_module.os.path, "exists", lambda p: True)
    critical = mock.Mock()
    information = mock.Mock()
    with (
        mock.patch.object(
            core_module.QFileDialog,
            "getSaveFileName",
            return_value=(str(saved / "out.wav"), "WAV Audio (*.wav)"),
        ),
        mock.patch.object(core_module.QMessageBox, "information", information),
        mock.patch.object(core_module.QMessageBox, "critical", critical),
        mock.patch("korshmer.audio_processors.render_midi_to_wav"),
        mock.patch.object(core_module, "publish"),
    ):
        ExportHost().export_audio_action()
    assert (information.call_count, critical.call_count) == (1, 0)


def test_exporting_a_folder_onto_itself_keeps_the_folder_and_tells_the_user(
    tmp_path, monkeypatch, dialogs
):
    source = song_folder(tmp_path / "library")
    choose_folder(monkeypatch, source.parent)
    FilesMixin().export_file_action(str(source))
    assert (source / "take.mid").read_bytes() == b"MThd"
    assert (len(dialogs["warning"]), dialogs["information"]) == (1, [])


def test_exporting_a_folder_into_itself_is_refused(
    tmp_path, monkeypatch, dialogs
):
    source = song_folder(tmp_path)
    choose_folder(monkeypatch, source)
    FilesMixin().export_file_action(str(source))
    assert os.listdir(source) == ["take.mid"]
    assert len(dialogs["warning"]) == 1


def test_exporting_a_folder_over_a_folder_that_holds_it_is_refused(
    tmp_path, monkeypatch, dialogs
):
    source = song_folder(tmp_path / "target" / "Songs", "Songs")
    choose_folder(monkeypatch, tmp_path / "target")
    FilesMixin().export_file_action(str(source))
    assert (source / "take.mid").exists()
    assert len(dialogs["warning"]) == 1


def test_safeguard_exporting_a_folder_to_another_place_still_copies_it(
    tmp_path, monkeypatch, dialogs
):
    source = song_folder(tmp_path)
    target = tmp_path / "usb"
    target.mkdir()
    choose_folder(monkeypatch, target)
    FilesMixin().export_file_action(str(source))
    assert (target / "My Song" / "take.mid").exists()
    assert dialogs["warning"] == []
