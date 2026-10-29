from korshmer.ui.managers.workspace import WorkspaceManager


def test_workspace_manager_accepts_data_dir(tmp_path):
    wm = WorkspaceManager(data_dir=str(tmp_path / "test_ws"))
    assert wm.data_dir is not None
    assert wm.data_dir == tmp_path / "test_ws"


def test_workspace_manager_default_none():
    wm = WorkspaceManager()
    assert wm.data_dir is None
