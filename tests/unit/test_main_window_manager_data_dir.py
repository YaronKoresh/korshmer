from korshmer.ui.managers import KeyboardShortcutManager, WorkspaceManager


def test_keyboard_shortcut_manager_without_data_dir_cannot_save():
    mgr = KeyboardShortcutManager()
    assert mgr.data_dir is None
    assert mgr.save_preset("test") is False


def test_keyboard_shortcut_manager_with_data_dir_can_save(tmp_path):
    mgr = KeyboardShortcutManager(data_dir=tmp_path)
    assert mgr.data_dir is not None
    assert mgr.save_preset("test") is True
    assert (tmp_path / "shortcuts_test.json").exists()


def test_workspace_manager_without_data_dir_cannot_save():
    mgr = WorkspaceManager()
    assert mgr.data_dir is None
    mgr.save_workspace("test_ws", {"panel": "left"})
    assert "test_ws" in mgr.workspaces


def test_workspace_manager_with_data_dir_saves_to_disk(tmp_path):
    mgr = WorkspaceManager(data_dir=tmp_path)
    assert mgr.data_dir is not None
    mgr.save_workspace("test_ws", {"panel": "left"})
    assert (tmp_path / "workspace_test_ws.json").exists()


def test_workspace_manager_load_from_disk_with_data_dir(tmp_path):
    mgr1 = WorkspaceManager(data_dir=tmp_path)
    mgr1.save_workspace("custom", {"layout": "wide"})

    mgr2 = WorkspaceManager(data_dir=tmp_path)
    mgr2._load_from_disk()
    assert "custom" in mgr2.workspaces
    assert mgr2.workspaces["custom"]["layout"] == {"layout": "wide"}
