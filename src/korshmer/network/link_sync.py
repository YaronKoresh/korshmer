from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import threading
from collections.abc import Callable
from typing import Any

from korshmer.utils.tempo import DEFAULT_BPM, validate_bpm

_logger = logging.getLogger(__name__)

_EPSILON = 1e-6
_TEMPO_MATCH = 0.005


def _default_link_factory(bpm: float, loop: asyncio.AbstractEventLoop):
    from aalink import Link

    return Link(bpm, loop)


def link_available() -> bool:
    try:
        return importlib.util.find_spec("aalink") is not None
    except Exception:
        return False


class LinkSync:
    def __init__(
        self,
        on_tempo: Callable[[float], None],
        link_factory: Callable[[float, Any], Any] | None = None,
    ):
        self._on_tempo = on_tempo
        self._link_factory = link_factory or _default_link_factory
        self._lock = threading.Lock()
        self._link = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._last_seen: float | None = None

    @property
    def enabled(self) -> bool:
        return self._link is not None

    @property
    def num_peers(self) -> int:
        link = self._link
        if link is None:
            return 0
        try:
            return int(link.num_peers)
        except Exception:
            return 0

    def start(self, bpm: float = DEFAULT_BPM) -> bool:
        with self._lock:
            if self._link is not None:
                return True
            bpm = validate_bpm(bpm)
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run():
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()
                loop.close()

            thread = threading.Thread(
                target=run, name="korshmer-link", daemon=True
            )
            thread.start()
            ready.wait(5)
            try:
                link = self._link_factory(bpm, loop)
                link.set_tempo_callback(self._handle_link_tempo)
                link.enabled = True
            except Exception as e:
                _logger.warning("Ableton Link unavailable: %s", e)
                loop.call_soon_threadsafe(loop.stop)
                thread.join(2)
                return False
            self._loop = loop
            self._thread = thread
            self._link = link
            self._last_seen = bpm
            return True

    def stop(self) -> None:
        with self._lock:
            link, loop, thread = self._link, self._loop, self._thread
            self._link = self._loop = self._thread = None
        if link is None:
            return
        try:
            link.enabled = False
        except Exception as e:
            _logger.debug("Disabling Link failed: %s", e)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(2)

    def publish_tempo(self, bpm: float) -> bool:
        link = self._link
        if link is None:
            return False
        bpm = validate_bpm(bpm)
        try:
            current = float(link.tempo)
            if math.isfinite(current) and abs(current - bpm) < _TEMPO_MATCH:
                return False
            self._last_seen = bpm
            link.tempo = bpm
        except Exception as e:
            _logger.warning("Failed to publish tempo to Link: %s", e)
            return False
        return True

    def _handle_link_tempo(self, bpm) -> None:
        try:
            raw = float(bpm)
        except (TypeError, ValueError):
            _logger.warning("Ignoring invalid Link tempo %r", bpm)
            return
        if not math.isfinite(raw):
            _logger.warning("Ignoring non-finite Link tempo %r", bpm)
            return
        value = validate_bpm(raw)
        last = self._last_seen
        if last is not None and abs(value - last) < _EPSILON:
            return
        self._last_seen = value
        try:
            self._on_tempo(value)
        except Exception as e:
            _logger.error("Link tempo handler failed: %s", e)
