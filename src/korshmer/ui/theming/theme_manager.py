import json
import logging
import os
import re
from pathlib import Path

THEME_FILE_PREFIX = "theme_"
MAX_THEME_NAME_LENGTH = 100

_UNSAFE_NAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')


class ThemeManager:
    def __init__(self, data_dir=None):
        if data_dir is None:
            import tempfile

            data_dir = tempfile.mkdtemp()
        self.data_dir = Path(data_dir)
        self.current_theme = "dark"
        self.themes = self._init_themes()
        self.custom_themes = {}

    def _init_themes(self):
        return {
            "dark": {
                "name": "Dark",
                "bg_primary": "#101010",
                "bg_secondary": "#1a1a1a",
                "text_primary": "#ffffff",
                "text_secondary": "#b0b0b0",
                "accent": "#00FFFF",
                "border": "#333333",
                "hover": "#2a2a2a",
                "success": "#34A853",
                "warning": "#FBBC04",
                "error": "#EA4335",
                "waveform": "#00FFFF",
                "grid": "#2a2a2a",
                "opacity": 1.0,
            },
            "light": {
                "name": "Light",
                "bg_primary": "#ffffff",
                "bg_secondary": "#f5f5f5",
                "text_primary": "#111111",
                "text_secondary": "#666666",
                "accent": "#1A73E8",
                "border": "#dddddd",
                "hover": "#e8e8e8",
                "success": "#34A853",
                "warning": "#FBBC04",
                "error": "#EA4335",
                "waveform": "#1A73E8",
                "grid": "#e0e0e0",
                "opacity": 1.0,
            },
            "neon": {
                "name": "Neon",
                "bg_primary": "#0a0a0a",
                "bg_secondary": "#151515",
                "text_primary": "#00FFFF",
                "text_secondary": "#00aaaa",
                "accent": "#FF0040",
                "border": "#00FFFF",
                "hover": "#1a1a1a",
                "success": "#00FF00",
                "warning": "#FFFF00",
                "error": "#FF0040",
                "waveform": "#00FFFF",
                "grid": "#003333",
                "opacity": 1.0,
            },
            "classic": {
                "name": "Classic",
                "bg_primary": "#c0c0c0",
                "bg_secondary": "#d0d0d0",
                "text_primary": "#000000",
                "text_secondary": "#404040",
                "accent": "#000080",
                "border": "#808080",
                "hover": "#b0b0b0",
                "success": "#008000",
                "warning": "#808000",
                "error": "#800000",
                "waveform": "#000080",
                "grid": "#a0a0a0",
                "opacity": 1.0,
            },
            "high_contrast": {
                "name": "High Contrast",
                "bg_primary": "#000000",
                "bg_secondary": "#000000",
                "text_primary": "#ffffff",
                "text_secondary": "#ffff00",
                "accent": "#ffff00",
                "border": "#ffffff",
                "hover": "#333333",
                "success": "#00ff00",
                "warning": "#ffff00",
                "error": "#ff0000",
                "waveform": "#00ff00",
                "grid": "#444444",
                "opacity": 1.0,
            },
        }

    def get_theme(self, name=None):
        if name is None:
            name = self.current_theme
        if name in self.themes:
            return self.themes[name]
        if name in self.custom_themes:
            return self.custom_themes[name]
        return self.themes["dark"]

    def set_theme(self, name):
        if name in self.themes or name in self.custom_themes:
            self.current_theme = name
            return True
        return False

    @staticmethod
    def clean_theme_name(name):
        if not isinstance(name, str):
            return None
        cleaned = _UNSAFE_NAME_CHARS_RE.sub("_", name).strip()
        if cleaned.strip(".") == "":
            return None
        return cleaned[:MAX_THEME_NAME_LENGTH]

    def save_custom_theme(self, name, theme_data):
        if not isinstance(name, str) or self.clean_theme_name(name) != name:
            logging.error(f"Failed to save theme: invalid name {name!r}")
            return False
        self.custom_themes[name] = theme_data
        theme_path = self.data_dir / f"{THEME_FILE_PREFIX}{name}.json"
        try:
            tmp = str(theme_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(theme_data, f, indent=2)
            os.replace(tmp, theme_path)
            return True
        except Exception as e:
            logging.error(f"Failed to save theme: {e}")
            return False

    def load_custom_themes(self):
        try:
            theme_files = sorted(self.data_dir.glob("theme_*.json"))
        except Exception as e:
            logging.error(f"Failed to load custom themes: {e}")
            return
        for theme_file in theme_files:
            name = theme_file.stem.removeprefix(THEME_FILE_PREFIX)
            try:
                with open(theme_file, encoding="utf-8") as f:
                    theme_data = json.load(f)
                if not isinstance(theme_data, dict):
                    raise TypeError("theme must be an object")
                self.custom_themes[name] = theme_data
            except Exception as e:
                logging.error(f"Failed to load custom theme {name}: {e}")

    def export_theme(self, name, path):
        theme = self.get_theme(name)
        try:
            with open(path, "w") as f:
                json.dump(theme, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to export theme: {e}")
            return False

    def import_theme(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                theme_data = json.load(f)
            if not isinstance(theme_data, dict):
                raise TypeError("theme must be an object")
            name = self.clean_theme_name(theme_data.get("name"))
            if name is None:
                name = self.clean_theme_name(Path(path).stem)
            if name is None:
                raise ValueError("theme has no usable name")
            if not self.save_custom_theme(name, theme_data):
                return None
            return name
        except Exception as e:
            logging.error(f"Failed to import theme: {e}")
            return None
