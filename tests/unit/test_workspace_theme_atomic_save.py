from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

from korshmer.services.file_management import FileManagementService
from korshmer.ui.managers.theme import ThemeManager
from korshmer.ui.managers.workspace import WorkspaceManager


def test_workspace_save_uses_replace():
    source = inspect.getsource(WorkspaceManager._save_to_disk)
    assert "os.replace" in source


def test_theme_save_uses_replace():
    source = inspect.getsource(ThemeManager.save_custom_theme)
    assert "os.replace" in source


def test_file_management_save_uses_replace():
    source = inspect.getsource(FileManagementService.save_file_list)
    assert "os.replace" in source


def test_workspace_save_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        mgr = WorkspaceManager(data_dir=td)
        mgr.save_workspace("test", {"panels": [1, 2, 3]})

        path = Path(td) / "workspace_test.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["layout"] == {"panels": [1, 2, 3]}

        tmp_path = Path(td) / ".workspace_test.json.tmp"
        assert not tmp_path.exists()


def test_theme_save_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        mgr = ThemeManager(data_dir=td)
        assert mgr.save_custom_theme("myblue", {"accent": "#0000ff"})

        path = Path(td) / "theme_myblue.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["accent"] == "#0000ff"
