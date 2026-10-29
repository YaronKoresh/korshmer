from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from korshmer.midi.controller_mapping import (
    ControllerMapping,
    detect_controller_type,
    list_input_device_names,
)
from korshmer.midi.mapping_store import ControllerMappingStore

DEFAULT_POLL_INTERVAL: float = 2.0


class MidiHotplugMonitor(QObject):
    device_connected = Signal(str, str)
    device_disconnected = Signal(str)

    def __init__(
        self,
        list_devices: Callable[[], list[str]] = list_input_device_names,
        interval: float = DEFAULT_POLL_INTERVAL,
        parent=None,
        mapping_store: ControllerMappingStore | None = None,
    ):
        super().__init__(parent)
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval!r}")
        self._list_devices = list_devices
        self._store = (
            mapping_store
            if mapping_store is not None
            else ControllerMappingStore()
        )
        self.interval = interval
        self._lock = threading.Lock()
        self._known: set[str] = set()
        self._mappings: dict[str, ControllerMapping] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def connected_devices(self) -> list[str]:
        with self._lock:
            return sorted(self._known)

    def mapping_for(self, device_name: str) -> ControllerMapping | None:
        with self._lock:
            return self._mappings.get(device_name)

    def reload_mappings(self) -> None:
        with self._lock:
            for name, mapping in self._mappings.items():
                mapping.reset_cc_actions()
                mapping.macros = {}
                self._store.apply(name, mapping)

    def poll(self) -> None:
        try:
            current = {str(name) for name in self._list_devices()}
        except Exception:
            logging.debug("MIDI device listing failed", exc_info=True)
            return

        with self._lock:
            added = sorted(current - self._known)
            removed = sorted(self._known - current)
            self._known = current
            for name in removed:
                self._mappings.pop(name, None)
            for name in added:
                mapping = ControllerMapping.for_device(name)
                self._store.apply(name, mapping)
                self._mappings[name] = mapping

        for name in removed:
            self.device_disconnected.emit(name)
        for name in added:
            self.device_connected.emit(name, detect_controller_type(name))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="MidiHotplugMonitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.interval + 1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.interval)
