import pytest

from korshmer.project_management.project_manager import ProjectManager


@pytest.fixture
def manager(tmp_path):
    pm = ProjectManager(tmp_path)
    pm.create_project("demo")
    base = tmp_path / "saved_data" / "projects" / "demo" / "demo"
    assert pm.save_project(base, compress=False)
    return pm


def _break_save(manager):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    manager._save_compressed = broken


def test_failed_snapshot_keeps_project_data(manager):
    path = manager.current_project_path
    data = manager.current_project_data
    snapshot = dict(data)
    _break_save(manager)

    assert manager.create_snapshot("before change") is False

    assert manager.current_project_path == path
    assert manager.current_project_data is data
    assert manager.current_project_data == snapshot
    assert "snapshot_info" not in manager.current_project_data


def test_failed_template_save_keeps_project_data(manager):
    path = manager.current_project_path
    data = manager.current_project_data
    snapshot = dict(data)
    _break_save(manager)

    assert manager.save_as_template("tpl") is False

    assert manager.current_project_path == path
    assert manager.current_project_data is data
    assert manager.current_project_data == snapshot
    assert manager.current_project_data.get("name") == "demo"
    assert not manager.current_project_data.get("is_template")


def test_failed_template_load_keeps_project_data(manager):
    assert manager.save_as_template("tpl") is True
    path = manager.current_project_path
    data = manager.current_project_data

    def broken(*args, **kwargs):
        raise OSError("unreadable")

    manager._load_compressed = broken
    assert manager.load_template("tpl") is None

    assert manager.current_project_path == path
    assert manager.current_project_data is data


def test_snapshot_and_template_still_work_when_saving_succeeds(manager):
    path = manager.current_project_path
    data = manager.current_project_data

    assert manager.create_snapshot("ok") is True
    assert manager.save_as_template("tpl") is True

    assert manager.current_project_path == path
    assert manager.current_project_data is data
    assert len(manager.list_snapshots("demo")) == 1
    assert manager.list_templates() == ["tpl"]
