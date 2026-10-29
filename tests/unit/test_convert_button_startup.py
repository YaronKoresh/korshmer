from types import SimpleNamespace

import pytest

from korshmer.event_bus import Event, EventBus, subscribe
from korshmer.mixins.files import FilesMixin


@pytest.fixture(autouse=True)
def reset_bus():
    EventBus.reset()
    yield
    EventBus.reset()


def _states(paths):
    states = []
    subscribe(
        Event.CONVERT_BUTTON_STATE, lambda enabled=True: states.append(enabled)
    )
    FilesMixin.sync_convert_button(SimpleNamespace(file_list_paths=paths))
    return states


def test_button_enabled_when_saved_inputs_were_loaded_at_startup():
    assert _states(["a.wav"]) == [True]


def test_button_disabled_when_input_list_is_empty():
    assert _states([]) == [False]
