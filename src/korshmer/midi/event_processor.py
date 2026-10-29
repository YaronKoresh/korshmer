import threading
from typing import NamedTuple


class MidiEvent(NamedTuple):
    time: float
    etype: str
    channel: int
    note: int
    velocity: int
    note_id: int


class MidiEventQueue:
    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._events: list[MidiEvent] = []

    def __len__(self) -> int:
        with self._mutex:
            return len(self._events)

    def push(self, event: MidiEvent) -> None:
        with self._mutex:
            self._events.append(event)

    def sorted_events(self) -> list[MidiEvent]:
        with self._mutex:
            return sorted(self._events, key=lambda e: e[0])
