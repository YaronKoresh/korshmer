import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.validators.project_validator import (
    is_safe_name,
)

DEFAULT_CROSSFADE_MS = 2000.0


@dataclass
class SetlistItem:
    path: str
    crossfade_ms: float = DEFAULT_CROSSFADE_MS

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "crossfade_ms": self.crossfade_ms}


def _valid_crossfade(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


class Setlist:
    def __init__(
        self,
        project_manager: ProjectManager,
        name: str = "setlist",
        crossfader=None,
    ):
        if not is_safe_name(name):
            raise ValueError("invalid setlist name")
        self.project_manager = project_manager
        self.name = name
        self.crossfader = crossfader
        self.items: list[SetlistItem] = []
        self.index = 0
        self._playing_removed = False
        self.current_data: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._preloaded: dict[int, dict[str, Any]] = {}
        self._preload_thread: threading.Thread | None = None

    @property
    def current_item(self) -> SetlistItem | None:
        with self._lock:
            if self._playing_removed:
                return None
            if 0 <= self.index < len(self.items):
                return self.items[self.index]
            return None

    def add(self, path, crossfade_ms: float = DEFAULT_CROSSFADE_MS) -> bool:
        if not _valid_crossfade(crossfade_ms):
            logging.error(f"Rejected setlist crossfade: {crossfade_ms!r}")
            return False
        if not self._path_allowed(path):
            logging.error(f"Rejected setlist path: {path!r}")
            return False
        with self._lock:
            self.items.append(SetlistItem(str(path), float(crossfade_ms)))
        return True

    def remove(self, position: int) -> bool:
        with self._lock:
            if not 0 <= position < len(self.items):
                return False
            del self.items[position]
            self._preloaded.clear()
            if position < self.index:
                self.index -= 1
            elif (
                position == self.index
                and self.current_data is not None
                and not self._playing_removed
            ):
                self._playing_removed = True
            if not self._playing_removed:
                self.index = max(0, min(self.index, len(self.items) - 1))
            elif not self.items:
                self.index = 0
            return True

    def _path_allowed(self, path) -> bool:
        if not isinstance(path, (str, Path)) or not str(path):
            return False
        if "\x00" in str(path):
            return False
        return self.project_manager.is_path_within_saved_data(path)

    def _load_item(self, position: int) -> dict[str, Any] | None:
        return self._load_entry(self.items[position])

    def _load_entry(self, item: SetlistItem) -> dict[str, Any] | None:
        if not self._path_allowed(item.path):
            logging.error(
                f"Skipping setlist item outside saved_data: {item.path!r}"
            )
            return None
        data = self.project_manager.load_project_with_fallback(Path(item.path))
        if data is None and not Path(item.path).is_file():
            logging.error(f"Skipping missing setlist item: {item.path!r}")
            return None
        if data is None:
            logging.error(f"Skipping unloadable setlist item: {item.path!r}")
        return data

    def _take(self, position: int) -> dict[str, Any] | None:
        with self._lock:
            data = self._preloaded.pop(position, None)
        return data if data is not None else self._load_item(position)

    def _enter(self, position: int, data: dict[str, Any]) -> None:
        self.index = position
        self._playing_removed = False
        self.current_data = data
        self._preloaded.clear()
        if self.crossfader is not None:
            item = self.items[position]
            self.crossfader.switch(item.path, item.crossfade_ms)

    def _wait_for_preload(self) -> None:

        thread = self._preload_thread
        if thread is not None and thread.is_alive():
            thread.join()

    def _move(self, step: int) -> dict[str, Any] | None:
        self._wait_for_preload()
        with self._lock:
            position = self.index + step
            if self._playing_removed and step > 0:
                position = self.index
            while 0 <= position < len(self.items):
                data = self._take(position)
                if data is not None:
                    self._enter(position, data)
                    return data
                position += step
            return None

    def start(self) -> dict[str, Any] | None:
        self._wait_for_preload()
        with self._lock:
            position = self.index
            while 0 <= position < len(self.items):
                data = self._take(position)
                if data is not None:
                    self._enter(position, data)
                    return data
                position += 1
            return None

    def next(self) -> dict[str, Any] | None:
        return self._move(1)

    def previous(self) -> dict[str, Any] | None:
        return self._move(-1)

    def preload_next(self) -> None:
        with self._lock:
            position = self.index + (0 if self._playing_removed else 1)
            if position >= len(self.items) or position in self._preloaded:
                return
            item = self.items[position]

            def work():
                data = self._load_entry(item)
                if data is not None:
                    with self._lock:
                        if (
                            position < len(self.items)
                            and self.items[position] is item
                        ):
                            self._preloaded[position] = data

            self._preload_thread = threading.Thread(target=work, daemon=True)
            self._preload_thread.start()

    def _file_path(self) -> Path | None:
        if not is_safe_name(self.name):
            return None
        return self.project_manager.saved_data_dir / f"{self.name}.setlist.json"

    def save(self) -> bool:
        path = self._file_path()
        if path is None:
            return False
        with self._lock:
            payload = {
                "name": self.name,
                "items": [item.to_dict() for item in self.items],
            }
        try:
            tmp = path.parent / f".{path.name}.tmp"
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp, path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return True
        except OSError as e:
            logging.error(f"Failed to save setlist: {e}")
            return False

    @classmethod
    def load(
        cls, project_manager: ProjectManager, name: str, crossfader=None
    ) -> "Setlist | None":
        try:
            setlist = cls(project_manager, name, crossfader)
            path = setlist._file_path()
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload["items"]:
                setlist.add(entry["path"], entry["crossfade_ms"])
            return setlist
        except (OSError, ValueError, KeyError, TypeError) as e:
            logging.error(f"Failed to load setlist {name!r}: {e}")
            return None
