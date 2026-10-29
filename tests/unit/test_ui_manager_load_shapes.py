import json

import pytest

from korshmer.ui.managers.keyboard_shortcuts import KeyboardShortcutManager
from korshmer.ui.managers.theme import ThemeManager
from korshmer.ui.managers.workspace import WorkspaceManager
from korshmer.ui.theming.theme_manager import ThemeManager as ThemingManager

BUILTIN_WORKSPACES = {"mixing", "editing", "recording", "mastering"}


@pytest.mark.parametrize("payload", [[1, 2], "text", 5, None])
def test_shortcut_preset_that_is_not_an_object_keeps_current_shortcuts(
    tmp_path, payload
):
    (tmp_path / "shortcuts_bad.json").write_text(json.dumps(payload))
    manager = KeyboardShortcutManager(tmp_path)
    before = manager.get_all()

    assert manager.load_preset("bad") is False

    assert manager.get_all() == before
    assert manager.preset_name == "default"
    assert manager.get("save") == "Ctrl+S"


def test_shortcut_preset_drops_entries_that_are_not_text(tmp_path):
    (tmp_path / "shortcuts_mine.json").write_text(
        json.dumps({"save": "Ctrl+Alt+S", "open": 5, "new": ["x"]})
    )
    manager = KeyboardShortcutManager(tmp_path)

    assert manager.load_preset("mine") is True

    assert manager.get_all() == {"save": "Ctrl+Alt+S"}


def test_workspace_loader_skips_wrong_shapes_and_keeps_good_files(tmp_path):
    (tmp_path / "workspace_a.json").write_text("[]")
    (tmp_path / "workspace_b.json").write_text(
        json.dumps({"name": "b", "layout": {"x": 1}})
    )
    (tmp_path / "workspace_c.json").write_text("{bad")
    (tmp_path / "workspace_d.json").write_text(json.dumps({"name": "d"}))
    (tmp_path / "workspace_e.json").write_text(
        json.dumps({"name": "e", "layout": [1]})
    )
    (tmp_path / "workspace_f.json").write_bytes(b"\xff\xfe")
    (tmp_path / "workspace_g.json").write_text(
        json.dumps({"name": "g", "layout": {}})
    )
    manager = WorkspaceManager(tmp_path)

    manager._load_from_disk()

    custom = set(manager.workspaces) - BUILTIN_WORKSPACES
    assert custom == {"b", "g"}
    assert manager.load_workspace("b") == {"x": 1}


@pytest.mark.parametrize("cls", [ThemeManager, ThemingManager])
def test_theme_loader_skips_wrong_shapes_and_keeps_good_files(tmp_path, cls):
    (tmp_path / "theme_a.json").write_text("[]")
    (tmp_path / "theme_b.json").write_text(json.dumps({"bg": "#001"}))
    (tmp_path / "theme_c.json").write_text("{bad")
    (tmp_path / "theme_d.json").write_text('"text"')
    (tmp_path / "theme_e.json").write_bytes(b"\xff\xfe")
    (tmp_path / "theme_f.json").write_text(json.dumps({"bg": "#010"}))
    manager = cls(tmp_path)

    manager.load_custom_themes()

    assert manager.custom_themes == {"b": {"bg": "#001"}, "f": {"bg": "#010"}}


@pytest.mark.parametrize("cls", [ThemeManager, ThemingManager])
def test_import_theme_rejects_file_that_is_not_an_object(tmp_path, cls):
    source = tmp_path / "list.json"
    source.write_text("[1, 2]")
    manager = cls(tmp_path / "data")

    assert manager.import_theme(source) is None

    assert manager.custom_themes == {}
