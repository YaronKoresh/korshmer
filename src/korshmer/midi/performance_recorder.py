from __future__ import annotations

import threading
import time
from collections.abc import Callable

from korshmer.midi.controller_mapping import note_message_to_event
from korshmer.midi.event_processor import MidiEvent, MidiEventQueue
from korshmer.midi.message_validator import validate_cc, validate_channel


class PerformanceRecorder:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._start: float | None = None
        self._queue = MidiEventQueue()
        self._ccs: list[tuple[float, int, int, int]] = []
        self._next_id = 0

    @property
    def recording(self) -> bool:
        return self._start is not None

    def start(self) -> None:
        with self._lock:
            self._queue = MidiEventQueue()
            self._ccs = []
            self._next_id = 0
            self._start = self._clock()

    def feed(self, message: dict, now: float | None = None) -> bool:
        with self._lock:
            if self._start is None:
                return False
            t = max(0.0, (self._clock() if now is None else now) - self._start)
            kind = message.get("type")
            try:
                if kind == "control_change":
                    channel = validate_channel(message.get("channel"))
                    controller, value = validate_cc(
                        message.get("control"), message.get("value")
                    )
                    self._ccs.append((t, channel, controller, value))
                    return True
                if kind in ("note_on", "note_off"):
                    self._next_id += 1
                    self._queue.push(
                        note_message_to_event(message, t, self._next_id)
                    )
                    return True
            except (ValueError, TypeError):
                return False
        return False

    def stop(self) -> None:
        with self._lock:
            self._start = None

    @property
    def events(self) -> list[MidiEvent]:
        return self._queue.sorted_events()

    @property
    def control_changes(self) -> list[tuple[float, int, int, int]]:
        with self._lock:
            return sorted(self._ccs)

    def to_pretty_midi(self, tempo: float = 120.0, program: int = 0):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
        inst = pretty_midi.Instrument(program=int(program), name="Performance")
        open_notes: dict[tuple[int, int], list[tuple[float, int]]] = {}
        events = self.events
        ccs = self.control_changes
        end = max([e.time for e in events] + [c[0] for c in ccs] + [0.0])
        for e in events:
            key = (e.channel, e.note)
            if e.etype == "noteon":
                open_notes.setdefault(key, []).append((e.time, e.velocity))
            elif open_notes.get(key):
                start, velocity = open_notes[key].pop(0)
                inst.notes.append(
                    pretty_midi.Note(
                        velocity, e.note, start, max(e.time, start + 1e-3)
                    )
                )
        for (_, pitch), pending in open_notes.items():
            for start, velocity in pending:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity, pitch, start, max(end, start + 1e-3)
                    )
                )
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        for t, _channel, controller, value in ccs:
            inst.control_changes.append(
                pretty_midi.ControlChange(controller, value, t)
            )
        pm.instruments.append(inst)
        return pm

    def write(self, path: str, tempo: float = 120.0) -> str:
        self.to_pretty_midi(tempo).write(path)
        return path


def open_input_port(
    recorder: PerformanceRecorder, device_name: str | None = None
):
    try:
        import mido
    except ImportError:
        raise RuntimeError("MIDI input needs the mido package.") from None
    try:
        names = mido.get_input_names()
        name = device_name or (names[0] if names else None)
        if name is None:
            raise RuntimeError("No MIDI input device found.")

        def on_message(msg):
            data = {"type": msg.type, "channel": getattr(msg, "channel", 0)}
            for key in ("note", "velocity", "control", "value"):
                if hasattr(msg, key):
                    data[key] = getattr(msg, key)
            recorder.feed(data)

        return mido.open_input(name, callback=on_message)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"MIDI input failed ({type(e).__name__}).") from None
