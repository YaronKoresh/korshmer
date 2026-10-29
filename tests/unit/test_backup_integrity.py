from __future__ import annotations

import time

import pytest

from korshmer.project_management.project_backup import create_backup_file
from korshmer.project_management.project_manager import ProjectManager


@pytest.fixture
def manager(tmp_path):
    pm = ProjectManager(tmp_path)
    pm.create_project("demo")
    return pm


def _base(tmp_path):
    return tmp_path / "saved_data" / "projects" / "demo" / "demo"


def test_backup_of_uncompressed_project_keeps_project_path(manager, tmp_path):
    assert manager.save_project(_base(tmp_path), compress=False)
    original = manager.current_project_path
    assert original.suffix == ".json"

    manager.last_backup_time = 0
    assert manager.create_backup() is True

    assert manager.current_project_path == original
    assert len(manager.list_backups("demo")) == 1


def test_backup_of_compressed_project_keeps_project_path(manager, tmp_path):
    assert manager.save_project(_base(tmp_path))
    original = manager.current_project_path

    manager.last_backup_time = 0
    assert manager.create_backup() is True

    assert manager.current_project_path == original
    assert len(manager.list_backups("demo")) == 1


def test_failed_backup_save_keeps_project_path(manager, tmp_path, monkeypatch):
    assert manager.save_project(_base(tmp_path), compress=False)
    original = manager.current_project_path

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "korshmer.project_management.project_manager.save_compressed", broken
    )
    manager.last_backup_time = 0
    assert manager.create_backup() is False
    assert manager.current_project_path == original
    assert manager.list_backups("demo") == []


def test_failed_backup_leaves_no_partial_file_and_keeps_old_backups(tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    project = tmp_path / "song.json"
    project.write_text("{}")
    calls = []

    def save(path):
        calls.append(path)
        path.write_text("{}")
        if len(calls) == 3:
            path.write_text("{partial")
            raise OSError("disk full")

    for index in range(3):
        try:
            create_backup_file(project, backups, 2, save)
        except OSError:
            assert index == 2
        time.sleep(1.05)

    files = sorted(p for p in backups.glob("song_backup_*.kproj"))
    assert len(files) == 2
    assert all(p.read_text() == "{}" for p in files)

    create_backup_file(project, backups, 2, lambda path: path.write_text("{}"))
    files = sorted(p for p in backups.glob("song_backup_*.kproj"))
    assert len(files) == 2
    assert all(p.read_text() == "{}" for p in files)
    assert not any(p.is_file() for p in (backups / ".partial").glob("*"))
