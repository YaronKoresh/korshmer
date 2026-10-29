from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from korshmer.constants import DATA_DIR
from korshmer.midi.controller_mapping import ControllerMapping

MAPPINGS_FILE: Path = DATA_DIR / "controller_mappings.json"


class ControllerMappingStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else MAPPINGS_FILE
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logging.debug("Could not read controller mappings", exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp, self.path)

    def device_names(self) -> list[str]:
        with self._lock:
            return sorted(self._read())

    def apply(self, device_name: str, mapping: ControllerMapping) -> bool:
        with self._lock:
            entry = self._read().get(device_name)
        if not isinstance(entry, dict):
            return False
        mapping.load_dict(entry)
        return True

    def save(self, device_name: str, mapping: ControllerMapping) -> None:
        if not isinstance(device_name, str) or not device_name:
            raise ValueError("device_name must be a non-empty string")
        with self._lock:
            data = self._read()
            data[device_name] = mapping.to_dict()
            self._write(data)

    def delete(self, device_name: str) -> bool:
        with self._lock:
            data = self._read()
            if device_name not in data:
                return False
            del data[device_name]
            self._write(data)
            return True
