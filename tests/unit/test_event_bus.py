import threading

import pytest

from korshmer.event_bus import (
    Event,
    EventBus,
    clear,
    publish,
    subscribe,
    unsubscribe,
)


@pytest.fixture(autouse=True)
def reset_bus():
    EventBus.reset()
    yield
    EventBus.reset()


class TestEventBus:
    def test_subscribe_and_publish(self):
        results = []
        subscribe(Event.STATUS_UPDATE, lambda text="": results.append(text))
        publish(Event.STATUS_UPDATE, text="hello")
        assert results == ["hello"]

    def test_multiple_subscribers(self):
        r1, r2 = [], []
        subscribe(Event.STATUS_UPDATE, lambda text="": r1.append(text))
        subscribe(Event.STATUS_UPDATE, lambda text="": r2.append(text))
        publish(Event.STATUS_UPDATE, text="msg")
        assert r1 == ["msg"]
        assert r2 == ["msg"]

    def test_unsubscribe(self):
        results = []
        cb = lambda text="": results.append(text)
        subscribe(Event.STATUS_UPDATE, cb)
        publish(Event.STATUS_UPDATE, text="first")
        unsubscribe(Event.STATUS_UPDATE, cb)
        publish(Event.STATUS_UPDATE, text="second")
        assert results == ["first"]

    def test_unsubscribe_nonexistent(self):
        unsubscribe(Event.STATUS_UPDATE, lambda: None)

    def test_publish_no_subscribers(self):
        publish(Event.STATUS_UPDATE, text="nobody listening")

    def test_different_events_isolated(self):
        status, progress = [], []
        subscribe(Event.STATUS_UPDATE, lambda text="": status.append(text))
        subscribe(
            Event.PROGRESS_UPDATE,
            lambda value=0, text="": progress.append(value),
        )
        publish(Event.STATUS_UPDATE, text="s")
        publish(Event.PROGRESS_UPDATE, value=50, text="half")
        assert status == ["s"]
        assert progress == [50]

    def test_clear_removes_all(self):
        results = []
        subscribe(Event.STATUS_UPDATE, lambda text="": results.append(text))
        clear()
        publish(Event.STATUS_UPDATE, text="gone")
        assert results == []

    def test_singleton(self):
        bus1 = EventBus()
        bus2 = EventBus()
        assert bus1 is bus2

    def test_handler_exception_does_not_break_others(self):
        results = []

        def bad_handler(**kwargs):
            raise ValueError("oops")

        subscribe(Event.STATUS_UPDATE, bad_handler)
        subscribe(Event.STATUS_UPDATE, lambda text="": results.append(text))
        publish(Event.STATUS_UPDATE, text="still works")
        assert results == ["still works"]

    def test_thread_safety(self):
        results = []
        lock = threading.Lock()

        def safe_append(text=""):
            with lock:
                results.append(text)

        subscribe(Event.STATUS_UPDATE, safe_append)

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=publish,
                args=(Event.STATUS_UPDATE,),
                kwargs={"text": str(i)},
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10

    def test_multiple_kwargs(self):
        results = []

        def handler(path="", analyze=True, load_player=True):
            results.append((path, analyze, load_player))

        subscribe(Event.AUDIO_LOAD_REQUESTED, handler)
        publish(
            Event.AUDIO_LOAD_REQUESTED,
            path="/test.wav",
            analyze=False,
            load_player=True,
        )
        assert results == [("/test.wav", False, True)]

    def test_reset_creates_new_instance(self):
        bus1 = EventBus()
        results = []
        subscribe(Event.STATUS_UPDATE, lambda text="": results.append(text))
        EventBus.reset()
        bus2 = EventBus()
        assert bus1 is not bus2
        publish(Event.STATUS_UPDATE, text="after reset")
        assert results == []

    def test_duplicate_subscribe_ignored(self):
        results = []
        cb = lambda text="": results.append(text)
        subscribe(Event.STATUS_UPDATE, cb)
        subscribe(Event.STATUS_UPDATE, cb)
        publish(Event.STATUS_UPDATE, text="once")
        assert results == ["once"]

    def test_all_event_types_exist(self):
        expected = [
            "STATUS_UPDATE",
            "AUDIO_LOAD_REQUESTED",
            "MIDI_LOAD_REQUESTED",
            "PLAYBACK_STARTED",
            "PLAYBACK_STOPPED",
            "PLAYBACK_PAUSED",
            "PROGRESS_UPDATE",
            "PROCESS_STARTED",
            "PROCESS_FINISHED",
            "CONVERT_BUTTON_STATE",
            "CONVERT_BUTTON_TEXT",
            "CONVERT_BUTTON_ROLE",
            "PROGRESS_BAR_STATE",
            "TEMPO_CHANGED",
            "SCALE_VIEW_CHANGED",
            "FILE_LOCK_RELEASED",
            "OUTPUT_VIEW_REFRESH",
            "UI_RESET",
        ]
        for name in expected:
            assert hasattr(Event, name), f"Event.{name} missing"
