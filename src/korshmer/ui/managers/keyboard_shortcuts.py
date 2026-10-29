import json
import logging
import os
from pathlib import Path


class KeyboardShortcutManager:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.shortcuts = {}
        self.conflicts = set()
        self.preset_name = "default"
        self._init_default_shortcuts()

    def _init_default_shortcuts(self):
        self.shortcuts = {
            "play_pause": "Space",
            "tap_tempo": "T",
            "stop": "Escape",
            "split": "S",
            "mute": "M",
            "solo": "O",
            "record": "R",
            "undo": "Ctrl+Z",
            "redo": "Ctrl+Y",
            "redo_alt": "Ctrl+Shift+Z",
            "select_all": "Ctrl+A",
            "deselect": "Ctrl+Shift+A",
            "duplicate": "Ctrl+D",
            "build_song": "Ctrl+B",
            "octave_up": "Ctrl+Shift+Up",
            "octave_down": "Ctrl+Shift+Down",
            "delete": "Delete",
            "cut": "Ctrl+X",
            "copy": "Ctrl+C",
            "paste": "Ctrl+V",
            "save": "Ctrl+S",
            "open": "Ctrl+O",
            "new": "Ctrl+N",
            "export": "Ctrl+E",
            "zoom_in": "Ctrl+=",
            "zoom_out": "Ctrl+-",
            "zoom_h": "H",
            "zoom_v": "V",
            "home": "Home",
            "end": "End",
            "page_up": "PageUp",
            "page_down": "PageDown",
            "toggle_grid": "G",
            "toggle_snap": "N",
            "quantize": "Q",
            "humanize": "U",
            "transpose_up": "Ctrl+Up",
            "transpose_down": "Ctrl+Down",
            "help": "F1",
            "fullscreen": "F11",
            "performance_mode": "F10",
        }

    def register(self, action, key_sequence):
        existing_key = self.shortcuts.get(action)

        if existing_key == key_sequence:
            return True

        if action in self.shortcuts:
            del self.shortcuts[action]

        if key_sequence in self.shortcuts.values():
            self.conflicts.add(key_sequence)
            if existing_key is not None:
                self.shortcuts[action] = existing_key
            return False

        self.shortcuts[action] = key_sequence
        self.conflicts.discard(key_sequence)
        return True

    def get(self, action):
        return self.shortcuts.get(action, "")

    def get_all(self):
        return dict(self.shortcuts)

    def detect_conflicts(self):
        seen = {}
        conflicts = []
        for action, keys in self.shortcuts.items():
            if keys in seen:
                conflicts.append((action, seen[keys], keys))
            else:
                seen[keys] = action
        return conflicts

    def save_preset(self, name):
        if self.data_dir is None:
            logging.warning("Cannot save preset: data_dir not set")
            return False
        preset_path = self.data_dir / f"shortcuts_{name}.json"
        try:
            tmp = preset_path.parent / f".{preset_path.name}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(self.shortcuts, f, indent=2)
                os.replace(tmp, preset_path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            self.preset_name = name
            return True
        except Exception as e:
            logging.error(f"Failed to save shortcut preset: {e}")
            return False

    def load_preset(self, name):
        if self.data_dir is None:
            logging.warning("Cannot load preset: data_dir not set")
            return False
        preset_path = self.data_dir / f"shortcuts_{name}.json"
        try:
            if preset_path.exists():
                with open(preset_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise TypeError("shortcut preset must be an object")
                self.shortcuts = {
                    action: keys
                    for action, keys in data.items()
                    if isinstance(action, str) and isinstance(keys, str)
                }
                self.preset_name = name
                return True
        except Exception as e:
            logging.error(f"Failed to load shortcut preset: {e}")
        return False
