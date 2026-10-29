import json
import tempfile
from pathlib import Path

from korshmer.ui.managers.theme import ThemeManager


class TestThemeManagerInitialization:
    def test_init_with_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            assert theme_manager.data_dir == Path(tmpdir)
            assert theme_manager.current_theme == "dark"

    def test_init_creates_themes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            assert "dark" in theme_manager.themes
            assert "light" in theme_manager.themes
            assert "neon" in theme_manager.themes
            assert "classic" in theme_manager.themes
            assert "high_contrast" in theme_manager.themes

    def test_init_empty_custom_themes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            assert theme_manager.custom_themes == {}


class TestGetTheme:
    def test_get_current_theme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            theme = theme_manager.get_theme()
            assert theme == theme_manager.themes["dark"]

    def test_get_specific_theme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            theme = theme_manager.get_theme("light")
            assert theme == theme_manager.themes["light"]

    def test_get_nonexistent_theme_returns_dark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            theme = theme_manager.get_theme("nonexistent")
            assert theme == theme_manager.themes["dark"]


class TestSetTheme:
    def test_set_valid_theme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            result = theme_manager.set_theme("light")
            assert result is True
            assert theme_manager.current_theme == "light"

    def test_set_invalid_theme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            result = theme_manager.set_theme("nonexistent")
            assert result is False
            assert theme_manager.current_theme == "dark"


class TestSaveCustomTheme:
    def test_save_custom_theme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)
            custom_theme = {
                "name": "custom",
                "bg_primary": "#123456",
                "text_primary": "#FFFFFF",
            }
            result = theme_manager.save_custom_theme("custom", custom_theme)
            assert result is True
            assert "custom" in theme_manager.custom_themes

            theme_file = Path(tmpdir) / "theme_custom.json"
            assert theme_file.exists()


class TestLoadCustomThemes:
    def test_load_custom_themes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_data = {
                "name": "custom",
                "bg_primary": "#123456",
                "text_primary": "#FFFFFF",
            }
            theme_file = Path(tmpdir) / "theme_custom.json"
            with open(theme_file, "w") as f:
                json.dump(theme_data, f)

            theme_manager = ThemeManager(tmpdir)
            theme_manager.load_custom_themes()

            assert "custom" in theme_manager.custom_themes
            assert (
                theme_manager.custom_themes["custom"]["bg_primary"] == "#123456"
            )


class TestCorrectUsagePattern:
    def test_theme_manager_with_data_dir_parameter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            theme_manager = ThemeManager(tmpdir)

            assert theme_manager.data_dir == Path(tmpdir)
            assert theme_manager.current_theme == "dark"
            assert len(theme_manager.themes) > 0

            theme = theme_manager.get_theme()
            assert "bg_primary" in theme
            assert "text_primary" in theme
