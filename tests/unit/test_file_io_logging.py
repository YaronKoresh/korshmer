import logging
import os

import pytest

from korshmer.project_management.file_io_log import LOGGER_NAME, logged_file_op
from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.project_recovery import (
    clear_recovery_file,
    load_recovery_file,
)


def _messages(caplog, level=None):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER_NAME and (level is None or r.levelno == level)
    ]


@pytest.fixture
def manager(tmp_path):
    pm = ProjectManager(tmp_path)
    pm.create_project("demo")
    return pm


@pytest.fixture
def project_base(manager, tmp_path):
    return tmp_path / "saved_data" / "projects" / "demo" / "demo"


def test_successful_save_logs_path_and_size(manager, project_base, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    assert manager.save_project(project_base)

    saved = project_base.with_suffix(".kproj")
    done = _messages(caplog, logging.INFO)
    assert any(
        m.startswith(f"Save project done: {saved}")
        and f"{saved.stat().st_size} bytes" in m
        for m in done
    )
    assert any(
        m == f"Save project started: {saved}"
        for m in _messages(caplog, logging.DEBUG)
    )


def test_uncompressed_save_and_load_are_logged(manager, project_base, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    assert manager.save_project(project_base, compress=False)
    saved = project_base.with_suffix(".json")
    assert manager.load_project(saved) is not None

    done = _messages(caplog, logging.INFO)
    assert any(m.startswith(f"Save project done: {saved}") for m in done)
    assert any(m.startswith(f"Load project done: {saved}") for m in done)


def test_failed_load_logs_the_file_path(manager, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    bad = tmp_path / "saved_data" / "projects" / "bad.json"
    bad.write_text("{oops", encoding="utf-8")

    assert manager.load_project(bad) is None

    errors = _messages(caplog, logging.ERROR)
    assert len(errors) == 1
    assert errors[0].startswith(f"Load project failed: {bad} (JSONDecodeError")


def test_missing_compressed_project_logs_the_file_path(
    manager, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    missing = tmp_path / "saved_data" / "projects" / "missing.kproj"

    assert manager.load_project(missing) is None

    errors = _messages(caplog, logging.ERROR)
    assert errors and errors[0].startswith(f"Load project failed: {missing}")


def test_backup_is_logged(manager, project_base, caplog):
    assert manager.save_project(project_base)
    manager.last_backup_time = 0
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    assert manager.create_backup()

    backups = manager.list_backups("demo")
    assert len(backups) == 1
    assert any(
        m.startswith(f"Create backup done: {backups[0]}")
        for m in _messages(caplog, logging.INFO)
    )


def test_old_backup_deletion_is_logged(manager, project_base, caplog):
    assert manager.save_project(project_base)
    backups_dir = manager.backups_dir
    old = backups_dir / "demo_backup_20000101_000000.kproj"
    old.write_bytes(b"x")
    manager.max_backups = 1
    manager.last_backup_time = 0
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    os.utime(old, (1, 1))
    assert manager.create_backup()

    assert not old.exists()
    assert f"Delete old backup done: {old}" in _messages(caplog, logging.INFO)


def test_recovery_save_load_and_clear_are_logged(manager, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    recovery = manager._recovery_file()

    assert manager.save_recovery_state()
    assert load_recovery_file(recovery) is not None
    assert clear_recovery_file(recovery)

    done = _messages(caplog, logging.INFO)
    assert any(
        m.startswith(f"Save recovery file done: {recovery}") for m in done
    )
    assert any(
        m.startswith(f"Load recovery file done: {recovery}") for m in done
    )
    assert any(
        m.startswith(f"Clear recovery file done: {recovery} (") for m in done
    )


def test_corrupt_recovery_file_logs_read_failure_and_move(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    recovery = tmp_path / ".recovery.gz"
    recovery.write_bytes(b"not gzip")

    assert load_recovery_file(recovery) is None

    errors = _messages(caplog, logging.ERROR)
    assert errors and errors[0].startswith(
        f"Load recovery file failed: {recovery}"
    )
    assert any(
        m.startswith(f"Move corrupt recovery file done: {recovery} -> ")
        for m in _messages(caplog, logging.INFO)
    )


def test_logged_file_op_reraises_and_skips_done_line(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    target = tmp_path / "x.bin"

    with pytest.raises(OSError):
        with logged_file_op("Write thing", target):
            raise OSError("disk full")

    assert _messages(caplog, logging.ERROR) == [
        f"Write thing failed: {target} (OSError: disk full)"
    ]
    assert _messages(caplog, logging.INFO) == []


def test_logged_file_op_without_existing_file_omits_size(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    target = tmp_path / "gone.bin"

    with logged_file_op("Remove thing", target):
        pass

    done = _messages(caplog, logging.INFO)
    assert len(done) == 1
    assert "bytes" not in done[0]
    assert done[0].startswith(f"Remove thing done: {target} (")
