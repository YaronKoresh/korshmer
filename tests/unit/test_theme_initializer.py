import json

from korshmer.ui.managers.theme import ThemeManager
from korshmer.ui.managers.theme_initializer import init_theme_manager


def test_loads_custom_themes_from_data_dir(tmp_path):
    (tmp_path / "theme_ocean.json").write_text(json.dumps({"bg": "#001"}))
    (tmp_path / "theme_forest.json").write_text(json.dumps({"bg": "#010"}))

    theme_manager = init_theme_manager(tmp_path)

    assert isinstance(theme_manager, ThemeManager)
    assert theme_manager.data_dir == tmp_path
    assert theme_manager.custom_themes == {
        "ocean": {"bg": "#001"},
        "forest": {"bg": "#010"},
    }


def test_matches_manual_construction_and_load(tmp_path):
    (tmp_path / "theme_ocean.json").write_text(json.dumps({"bg": "#001"}))

    expected = ThemeManager(tmp_path)
    expected.load_custom_themes()

    assert init_theme_manager(tmp_path).custom_themes == expected.custom_themes


def test_empty_data_dir_has_no_custom_themes(tmp_path):
    assert init_theme_manager(tmp_path).custom_themes == {}
