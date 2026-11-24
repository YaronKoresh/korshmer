import os
import sys
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QWidget,
)

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.mixins.actions import project_management as pm_module
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)


class FakeArranger:
    def __init__(self):
        self.model = SimpleNamespace(notes=[1], clips=[])
        self.state = {
            "tempo": 120,
            "tracks": [],
            "notes": [1],
            "clips": [],
            "time_signature": "4/4",
        }

    def to_project_dict(self):
        return dict(self.state)


class FakeProjects:
    def __init__(self, error=None):
        self.last_load_backup = None
        if error is not None:
            self.last_load_error = error

    def load_project_with_fallback(self, path):
        return None


class FakeExports:
    def __init__(self, failures):
        self.failures = failures
        self.export_queue = deque()
        self.worker = None

    def get_preset_names(self):
        return ["CD Quality"]

    def batch_export(self, source_files, preset, output_dir):
        for source in source_files:
            self.export_queue.append(
                {"source": str(source), "callback": None, "status": "queued"}
            )
        return len(source_files)

    def process_queue(self):
        self.worker = threading.Thread(target=self.run)
        self.worker.start()

    def run(self):
        while self.export_queue:
            job = self.export_queue.popleft()
            problem = self.failures.get(Path(job["source"]).name)
            job["status"] = "failed" if problem else "completed"
            if problem:
                job["error"] = problem
            if job["callback"]:
                job["callback"](job, not problem)


class SongHost(QWidget, ProjectManagementActionsMixin):
    def __init__(self, project_manager=None, export_manager=None):
        super().__init__()
        self.arranger = FakeArranger()
        self.project_manager = project_manager
        self.export_manager = export_manager
        self.output_list = QListWidget()


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda parent, title, text: shown.append(text)),
    )
    return shown


@pytest.fixture
def status(monkeypatch):
    texts = []
    monkeypatch.setattr(
        pm_module, "publish", lambda event, **data: texts.append(data["text"])
    )
    return texts


def export_all(host, tmp_path, monkeypatch, names):
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, str(path))
        host.output_list.addItem(item)
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        staticmethod(lambda *a, **k: ("CD Quality", True)),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: str(tmp_path / "out")),
    )
    host.batch_export_action()
    host.export_manager.worker.join()
    QApplication.processEvents()


def test_changing_only_the_time_signature_marks_the_song_as_changed():
    host = SongHost()
    host._mark_song_saved()
    host.arranger.state["time_signature"] = "3/4"
    assert host.has_unsaved_changes()


def test_safeguard_an_untouched_song_is_not_marked_as_changed():
    host = SongHost()
    host._mark_song_saved()
    assert not host.has_unsaved_changes()


def test_a_song_that_cannot_be_opened_shows_why(warnings):
    host = SongHost(FakeProjects("The file is not a KORSHMER song"))
    host.open_song_file("/songs/broken.kproj")
    assert warnings == [
        "Failed to load project:\nThe file is not a KORSHMER song"
    ]


def test_safeguard_without_a_known_reason_the_plain_message_is_shown(warnings):
    host = SongHost(FakeProjects())
    host.open_song_file("/songs/broken.kproj")
    assert warnings == ["Failed to load project"]


def test_a_failed_export_is_shown_to_the_user_with_its_reason(
    tmp_path, monkeypatch, warnings, status
):
    host = SongHost(export_manager=FakeExports({"b.wav": "Disk is full"}))
    export_all(host, tmp_path, monkeypatch, ["a.wav", "b.wav"])
    assert len(warnings) == 1
    assert "b.wav" in warnings[0] and "Disk is full" in warnings[0]


def test_the_export_problem_is_shown_on_the_window_thread(
    tmp_path, monkeypatch, status
):
    threads = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a: threads.append(threading.current_thread())),
    )
    host = SongHost(export_manager=FakeExports({"a.wav": "Disk is full"}))
    export_all(host, tmp_path, monkeypatch, ["a.wav"])
    assert threads == [threading.main_thread()]


def test_a_finished_export_reports_how_many_files_were_written(
    tmp_path, monkeypatch, warnings, status
):
    host = SongHost(export_manager=FakeExports({"b.wav": "Disk is full"}))
    export_all(host, tmp_path, monkeypatch, ["a.wav", "b.wav", "c.wav"])
    assert status[-1] == "Exported 2 of 3 files"


def test_an_export_without_problems_shows_no_warning(
    tmp_path, monkeypatch, warnings, status
):
    host = SongHost(export_manager=FakeExports({}))
    export_all(host, tmp_path, monkeypatch, ["a.wav"])
    assert (warnings, status[-1]) == ([], "Exported 1 of 1 files")


def test_safeguard_nothing_to_export_is_still_reported(
    tmp_path, monkeypatch, warnings
):
    host = SongHost(export_manager=mock.Mock())
    host.batch_export_action()
    assert warnings == ["No files available for export"]
