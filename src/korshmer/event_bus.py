import logging
import threading
from enum import Enum, auto

_logger = logging.getLogger(__name__)


class Event(Enum):
    STATUS_UPDATE = auto()
    AUDIO_LOAD_REQUESTED = auto()
    MIDI_LOAD_REQUESTED = auto()
    PLAYBACK_STARTED = auto()
    PLAYBACK_STOPPED = auto()
    PLAYBACK_PAUSED = auto()
    PROGRESS_UPDATE = auto()
    PROCESS_STARTED = auto()
    PROCESS_FINISHED = auto()
    CONVERT_BUTTON_STATE = auto()
    CONVERT_BUTTON_TEXT = auto()
    CONVERT_BUTTON_ROLE = auto()
    PROGRESS_BAR_STATE = auto()
    TEMPO_CHANGED = auto()
    SCALE_VIEW_CHANGED = auto()
    FILE_LOCK_RELEASED = auto()
    OUTPUT_VIEW_REFRESH = auto()
    UI_RESET = auto()
    STEMS_READY = auto()
    EDITOR_LOAD_MIDI = auto()
    EDITOR_ADD_AUDIO_CLIP = auto()
    EDITOR_TRACKS_CHANGED = auto()


class EventBus:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._subscribers = {}
                    inst._sub_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    def subscribe(self, event, callback):
        with self._sub_lock:
            if event not in self._subscribers:
                self._subscribers[event] = []
            if callback not in self._subscribers[event]:
                self._subscribers[event].append(callback)

    def unsubscribe(self, event, callback):
        with self._sub_lock:
            if event in self._subscribers:
                try:
                    self._subscribers[event].remove(callback)
                except ValueError:
                    pass

    def publish(self, event, **data):
        with self._sub_lock:
            callbacks = list(self._subscribers.get(event, []))
        for cb in callbacks:
            try:
                cb(**data)
            except Exception as e:
                _logger.error(f"Event handler error for {event.name}: {e}")

    def clear(self):
        with self._sub_lock:
            self._subscribers.clear()

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None


def subscribe(event, callback):
    EventBus().subscribe(event, callback)


def unsubscribe(event, callback):
    EventBus().unsubscribe(event, callback)


def publish(event, **data):
    EventBus().publish(event, **data)


def clear():
    EventBus().clear()
