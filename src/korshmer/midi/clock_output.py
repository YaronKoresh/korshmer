import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from korshmer.midi.controller_output import open_midi_output
from korshmer.utils.tempo import DEFAULT_BPM, validate_bpm

PULSES_PER_QUARTER = 24

STATE_STOPPED = 0
STATE_PLAYING = 1
STATE_PAUSED = 2


class MidiClockOutput:
    def __init__(
        self,
        open_output: Callable[[str], Any] = open_midi_output,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._open_output = open_output
        self._clock = clock
        self._lock = threading.RLock()
        self._port = None
        self._bpm = DEFAULT_BPM
        self._state = STATE_STOPPED
        self._started_once = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_open(self) -> bool:
        return self._port is not None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def interval(self) -> float:
        return 60.0 / (self._bpm * PULSES_PER_QUARTER)

    def open(self, device_name: str) -> bool:
        self.close()
        port = self._open_output(device_name)
        if port is None:
            return False
        with self._lock:
            self._port = port
        return True

    def close(self) -> None:
        self._halt_thread()
        with self._lock:
            port, self._port = self._port, None
            self._state = STATE_STOPPED
            self._started_once = False
        if port is not None:
            try:
                port.close()
            except Exception:
                logging.debug("Could not close MIDI clock port", exc_info=True)

    def set_tempo(self, bpm) -> None:
        self._bpm = validate_bpm(bpm, default=self._bpm)

    def on_playback_state_changed(self, state: int) -> None:
        state = int(state)
        if state == self._state:
            return
        if state == STATE_PLAYING:
            self._begin()
        elif state in (STATE_PAUSED, STATE_STOPPED):
            self._end(state)

    def _send(self, message_type: str) -> bool:
        with self._lock:
            port = self._port
            if port is None:
                return False
            try:
                port.send({"type": message_type})
            except Exception:
                logging.debug("MIDI clock send failed", exc_info=True)
                return False
        return True

    def _begin(self) -> None:
        if self._port is None:
            self._state = STATE_PLAYING
            return
        self._halt_thread()
        message = "continue" if self._state == STATE_PAUSED else "start"
        self._state = STATE_PLAYING
        self._send(message)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(self._stop_event,),
            name="midi-clock",
            daemon=True,
        )
        self._thread.start()

    def _end(self, state: int) -> None:
        was_playing = self._state == STATE_PLAYING
        self._halt_thread()
        self._state = state
        if was_playing:
            self._send("stop")

    def _halt_thread(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _run(self, stop_event: threading.Event) -> None:
        next_tick = self._clock()
        while not stop_event.is_set():
            if self._port is None:
                return
            self._send("clock")
            next_tick += self.interval
            delay = next_tick - self._clock()
            if delay > 0:
                if stop_event.wait(delay):
                    return
            else:
                next_tick = self._clock()
