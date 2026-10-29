from __future__ import annotations

from PySide6.QtCore import QObject, QTimer

DEFAULT_INTERVAL_MS: int = 16


class EventThrottle(QObject):
    def __init__(self, callback, interval_ms=DEFAULT_INTERVAL_MS, parent=None):
        super().__init__(parent)
        self._callback = callback
        self._pending = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(1, int(interval_ms)))
        self._timer.timeout.connect(self._on_timeout)

    def submit(self, *args):
        if self._timer.isActive():
            self._pending = args
            return
        self._timer.start()
        self._callback(*args)

    def flush(self):
        if self._pending is not None:
            args, self._pending = self._pending, None
            self._callback(*args)

    def cancel(self):
        self._pending = None
        self._timer.stop()

    def _on_timeout(self):
        if self._pending is not None:
            args, self._pending = self._pending, None
            self._timer.start()
            self._callback(*args)
