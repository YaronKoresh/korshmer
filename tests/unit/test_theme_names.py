import json

import pytest

from korshmer.ui.managers.theme import ThemeManager
from korshmer.ui.theming.theme_manager import (
    ThemeManager as ThemingThemeManager,
)

MANAGERS = [ThemeManager, ThemingThemeManager]


@pytest.fixture(params=MANAGERS)
def manager_cls(request):
    return request.param


def test_name_containing_theme_prefix_round_trips(manager_cls, tmp_path):
    manager = manager_cls(tmp_path)
    assert manager.save_custom_theme("my_theme_1", {"bg_primary": "#111111"})
    reloaded = manager_cls(tmp_path)
    reloaded.load_custom_themes()
    assert list(reloaded.custom_themes) == ["my_theme_1"]


@pytest.mark.parametrize(
    "name", ["", "..", ".", "a/b", "a\\b", "../x", "x\x00y", 5, None, "a" * 101]
)
def test_save_rejects_unsafe_names(manager_cls, tmp_path, name):
    manager = manager_cls(tmp_path)
    assert manager.save_custom_theme(name, {"bg_primary": "#111111"}) is False
    assert manager.custom_themes == {}
    assert list(tmp_path.iterdir()) == []


def test_import_cleans_path_characters_in_name(manager_cls, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = tmp_path / "in.json"
    source.write_text(json.dumps({"name": "../evil", "bg_primary": "#222222"}))
    manager = manager_cls(data_dir)
    assert manager.import_theme(source) == ".._evil"
    assert [p.name for p in data_dir.iterdir()] == ["theme_.._evil.json"]
    assert [p.name for p in tmp_path.iterdir()] == ["data", "in.json"]


def test_import_uses_file_name_when_theme_name_is_missing(
    manager_cls, tmp_path
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = tmp_path / "ocean.json"
    source.write_text(json.dumps({"bg_primary": "#222222"}))
    manager = manager_cls(data_dir)
    assert manager.import_theme(source) == "ocean"
    assert "ocean" in manager.custom_themes


def test_import_falls_back_to_file_name_for_dot_only_theme_name(
    manager_cls, tmp_path
):
    source = tmp_path / "plain.json"
    source.write_text(json.dumps({"name": "..", "bg_primary": "#222222"}))
    manager = manager_cls(tmp_path)
    assert manager.import_theme(source) == "plain"
    assert (tmp_path / "theme_plain.json").exists()


def test_import_that_cannot_be_saved_returns_none(manager_cls, tmp_path):
    source = tmp_path / "in.json"
    source.write_text(json.dumps({"name": "ok", "bg_primary": "#222222"}))
    manager = manager_cls(tmp_path / "missing")
    assert manager.import_theme(source) is None
