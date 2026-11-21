import json
import logging
import os
import time
from pathlib import Path

from korshmer.project_management.validators.project_validator import (
    is_valid_new_name,
)

WORKSPACE_FILE_PREFIX = "workspace_"
WINDOW_LAYOUT_FILE = "window_layout.json"


def _whole_numbers(values, count):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        return None
    if any(type(value) is not int for value in values):
        return None
    return tuple(values)


def _shared_area(first, second):
    width = min(first[0] + first[2], second[0] + second[2]) - max(
        first[0], second[0]
    )
    height = min(first[1] + first[3], second[1] + second[3]) - max(
        first[1], second[1]
    )
    return max(0, width) * max(0, height)


def geometry_on_screen(geometry, screens, min_width, min_height):
    wanted = _whole_numbers(geometry, 4)
    areas = [area for area in screens if _whole_numbers(area, 4)]
    if wanted is None or not areas:
        return None
    x, y, width, height = wanted
    width = max(width, min_width)
    height = max(height, min_height)
    home = max(
        areas, key=lambda area: _shared_area((x, y, width, height), area)
    )
    width = min(width, home[2])
    height = min(height, home[3])
    x = min(max(x, home[0]), home[0] + home[2] - width)
    y = min(max(y, home[1]), home[1] + home[3] - height)
    return x, y, width, height


class WorkspaceManager:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.workspaces = {
            "mixing": {"name": "Mixing", "layout": {}},
            "editing": {"name": "Editing", "layout": {}},
            "recording": {"name": "Recording", "layout": {}},
            "mastering": {"name": "Mastering", "layout": {}},
        }
        self.current_workspace = "editing"
        self.auto_save = True

    def save_workspace(self, name, layout_data):
        if not is_valid_new_name(name):
            logging.error(f"Failed to save workspace: invalid name {name!r}")
            return False
        builtin_workspaces = {"mixing", "editing", "recording", "mastering"}
        if name in builtin_workspaces and name in self.workspaces:
            if not self.workspaces[name].get("layout"):
                logging.warning(
                    f"Saving changes to built-in workspace '{name}'"
                )

        self.workspaces[name] = {
            "name": name,
            "layout": layout_data,
            "timestamp": time.time(),
        }
        if self.auto_save:
            return self._save_to_disk(name)
        return True

    def load_workspace(self, name):
        if name in self.workspaces:
            self.current_workspace = name
            return self.workspaces[name]["layout"]
        return None

    def _save_to_disk(self, name):
        if self.data_dir is None:
            return True
        workspace_path = self.data_dir / f"{WORKSPACE_FILE_PREFIX}{name}.json"
        try:
            tmp = workspace_path.parent / f".{workspace_path.name}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(self.workspaces[name], f, indent=2)
                os.replace(tmp, workspace_path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return True
        except Exception as e:
            logging.error(f"Failed to save workspace: {e}")
            return False

    def _load_from_disk(self):
        if self.data_dir is None:
            return
        try:
            workspace_files = sorted(self.data_dir.glob("workspace_*.json"))
        except Exception as e:
            logging.error(f"Failed to load workspaces: {e}")
            return
        for workspace_file in workspace_files:
            name = workspace_file.stem.removeprefix(WORKSPACE_FILE_PREFIX)
            if not is_valid_new_name(name):
                logging.error(f"Skipping workspace file {workspace_file.name}")
                continue
            try:
                with open(workspace_file, encoding="utf-8") as f:
                    workspace = json.load(f)
                if not isinstance(workspace, dict) or not isinstance(
                    workspace.get("layout"), dict
                ):
                    raise TypeError("workspace must be an object with a layout")
                self.workspaces[name] = workspace
            except Exception as e:
                logging.error(f"Failed to load workspace {name}: {e}")

    def load_window_layout(self):
        if self.data_dir is None:
            return {}
        try:
            with open(
                self.data_dir / WINDOW_LAYOUT_FILE, encoding="utf-8"
            ) as f:
                layout = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logging.warning(f"Could not read the saved window layout: {e}")
            return {}
        return layout if isinstance(layout, dict) else {}

    def save_window_layout(self, layout):
        if self.data_dir is None:
            return False
        layout_path = self.data_dir / WINDOW_LAYOUT_FILE
        tmp = layout_path.parent / f".{layout_path.name}.tmp"
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(layout, f, indent=2)
                os.replace(tmp, layout_path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        except (OSError, TypeError, ValueError) as e:
            logging.error(f"Failed to save the window layout: {e}")
            return False
        return True

    def remember_layout_value(self, key, value):
        layout = self.load_window_layout()
        layout[key] = value
        return self.save_window_layout(layout)

    def get_all_workspaces(self):
        return list(self.workspaces.keys())

    def delete_workspace(self, name):
        if name in self.workspaces and name not in [
            "mixing",
            "editing",
            "recording",
            "mastering",
        ]:
            del self.workspaces[name]
            if self.data_dir is not None:
                workspace_path = (
                    self.data_dir / f"{WORKSPACE_FILE_PREFIX}{name}.json"
                )
                if workspace_path.exists():
                    workspace_path.unlink()
            return True
        return False
