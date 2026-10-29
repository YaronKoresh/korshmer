import json

import pytest

from korshmer.ui.managers.workspace import WorkspaceManager

BUILTIN = {"mixing", "editing", "recording", "mastering"}


def _custom(manager):
    return set(manager.get_all_workspaces()) - BUILTIN


@pytest.mark.parametrize(
    "name",
    ["my_workspace_1", "workspace_workspace_", "live_set", "mix.v2", "A B"],
)
def test_saved_workspace_name_survives_a_reload(tmp_path, name):
    first = WorkspaceManager(tmp_path)
    assert first.save_workspace(name, {"panel": 1}) is True

    second = WorkspaceManager(tmp_path)
    second._load_from_disk()

    assert _custom(second) == {name}
    assert second.load_workspace(name) == {"panel": 1}


@pytest.mark.parametrize(
    "name", ["", " ", ".", "..", "a/b", "a\\b", "bad\x00name", None, 5, "x."]
)
def test_invalid_workspace_names_are_rejected(tmp_path, name):
    manager = WorkspaceManager(tmp_path)

    assert manager.save_workspace(name, {"panel": 1}) is False

    assert _custom(manager) == set()
    assert list(tmp_path.iterdir()) == []


def test_failed_disk_write_is_reported(tmp_path):
    manager = WorkspaceManager(tmp_path / "missing" / "dir")

    assert manager.save_workspace("live", {"panel": 1}) is False


def test_workspace_without_data_dir_is_kept_in_memory():
    manager = WorkspaceManager()

    assert manager.save_workspace("live", {"panel": 1}) is True
    assert manager.load_workspace("live") == {"panel": 1}


def test_auto_save_off_keeps_the_workspace_out_of_the_disk(tmp_path):
    manager = WorkspaceManager(tmp_path)
    manager.auto_save = False

    assert manager.save_workspace("live", {"panel": 1}) is True
    assert list(tmp_path.iterdir()) == []


def test_files_with_unusable_names_are_skipped_on_load(tmp_path):
    (tmp_path / "workspace_.json").write_text(json.dumps({"layout": {}}))
    (tmp_path / "workspace_...json").write_text(json.dumps({"layout": {}}))
    (tmp_path / "workspace_ok.json").write_text(json.dumps({"layout": {}}))

    manager = WorkspaceManager(tmp_path)
    manager._load_from_disk()

    assert _custom(manager) == {"ok"}


def test_delete_removes_a_reloaded_custom_workspace(tmp_path):
    first = WorkspaceManager(tmp_path)
    first.save_workspace("my_workspace_1", {})
    second = WorkspaceManager(tmp_path)
    second._load_from_disk()

    assert second.delete_workspace("my_workspace_1") is True
    assert not (tmp_path / "workspace_my_workspace_1.json").exists()
