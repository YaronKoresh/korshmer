import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

try:
    import korshmer.__main__ as main_module
    from korshmer.__main__ import MainWindow
except Exception as import_error:
    MainWindow = None
    MAIN_IMPORT_ERROR = str(import_error)
else:
    MAIN_IMPORT_ERROR = ""

from korshmer.core import control_bar as control_bar_module
from korshmer.core.audio_engine import AudioEngine
from korshmer.event_bus import Event, subscribe, unsubscribe
from korshmer.services import file_management
from korshmer.utils import audio_host

CONNECTED = [(1, "Speakers (ALSA)"), (2, "Studio (JACK)")]


class RememberingEngine(AudioEngine):
    def set_preferred_buffer_size(self, frames):
        self.asked_buffer = frames

    def set_output_device(self, index):
        self.asked_device = index


@pytest.fixture
def device_queries(tmp_path, monkeypatch):
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    folder = tmp_path / "saved_data"
    folder.mkdir()
    monkeypatch.setattr(main_module, "DATA_DIR", folder)
    monkeypatch.setattr(
        file_management, "INPUT_LIST_FILE", folder / "inputs.json"
    )
    monkeypatch.setattr(
        audio_host, "AUDIO_HOST_FILE", folder / "audio_host.json"
    )
    audio_host.reset_host_preference_cache()
    monkeypatch.setattr(control_bar_module, "AudioEngine", RememberingEngine)
    queries = []

    def list_devices(sd_module=None):
        queries.append(1)
        return list(CONNECTED)

    monkeypatch.setattr(audio_host, "list_output_devices", list_devices)
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox, name, staticmethod(lambda *a, **k: None)
        )
    yield queries
    audio_host.reset_host_preference_cache()


@pytest.fixture
def open_window(device_queries):
    opened = []

    def build():
        win = MainWindow()
        opened.append(win)
        return win

    yield build
    for win in opened:
        win.close()
    QApplication.processEvents()


@pytest.fixture
def window(open_window):
    return open_window()


def offered(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def open_device_list(window):
    window.combo_audio_device.showPopup()
    window.combo_audio_device.hidePopup()


def test_safeguard_opening_the_window_does_not_ask_the_sound_system_for_devices(
    device_queries, window
):
    assert device_queries == []


def test_the_export_page_always_offers_an_output_device(window):
    assert offered(window.combo_audio_device) == ["System default"]


def test_opening_the_device_list_shows_the_connected_devices(window):
    open_device_list(window)
    assert offered(window.combo_audio_device) == [
        "System default",
        "Speakers (ALSA)",
        "Studio (JACK)",
    ]


def test_opening_the_device_list_again_looks_for_new_devices(
    device_queries, window
):
    open_device_list(window)
    open_device_list(window)
    assert len(device_queries) == 2
    assert window.combo_audio_device.count() == 3


def test_the_audio_settings_offer_every_allowed_buffer_size(window):
    combo = window.combo_audio_buffer
    assert [combo.itemData(i) for i in range(combo.count())] == [
        None,
        512,
        1024,
        2048,
        4096,
        8192,
    ]


def test_choosing_a_buffer_size_saves_it(window):
    combo = window.combo_audio_buffer
    combo.setCurrentIndex(combo.findData(1024))
    assert audio_host.get_buffer_size_preference() == 1024


def test_choosing_a_buffer_size_reaches_the_player(window):
    combo = window.combo_audio_buffer
    combo.setCurrentIndex(combo.findData(4096))
    assert window.control_bar.player.asked_buffer == 4096


def test_choosing_an_output_device_saves_it(window):
    open_device_list(window)
    window.combo_audio_device.setCurrentIndex(2)
    assert audio_host.get_output_device_preference() == "Studio (JACK)"


def test_choosing_an_output_device_reaches_the_player(window):
    open_device_list(window)
    window.combo_audio_device.setCurrentIndex(2)
    assert window.control_bar.player.asked_device == 2


def test_choosing_the_system_default_forgets_the_saved_device(window):
    open_device_list(window)
    window.combo_audio_device.setCurrentIndex(2)
    window.combo_audio_device.setCurrentIndex(0)
    assert audio_host.get_output_device_preference() is None


def test_the_saved_audio_choices_are_shown_when_the_window_opens(open_window):
    audio_host.set_buffer_size_preference(8192)
    audio_host.set_output_device_preference("Studio (JACK)")
    window = open_window()
    assert (
        window.combo_audio_buffer.currentData(),
        window.combo_audio_device.currentText(),
    ) == (8192, "Studio (JACK)")


def test_the_saved_device_stays_selected_after_the_list_is_opened(open_window):
    audio_host.set_output_device_preference("Studio (JACK)")
    window = open_window()
    open_device_list(window)
    assert window.combo_audio_device.currentText() == "Studio (JACK)"
    assert audio_host.get_output_device_preference() == "Studio (JACK)"


def test_a_saved_device_that_is_unplugged_is_reported_when_the_app_starts(
    open_window,
):
    told = []

    def hear(text=""):
        told.append(text)

    audio_host.set_output_device_preference("Old USB Interface (ALSA)")
    subscribe(Event.STATUS_UPDATE, hear)
    try:
        window = open_window()
    finally:
        unsubscribe(Event.STATUS_UPDATE, hear)
    assert sum("Old USB Interface (ALSA)" in text for text in told) == 1
    assert window.control_bar.player.asked_device is None


def test_an_unplugged_saved_device_stays_listed_as_not_connected(open_window):
    audio_host.set_output_device_preference("Old USB Interface (ALSA)")
    window = open_window()
    open_device_list(window)
    assert window.combo_audio_device.currentText() == (
        "Old USB Interface (ALSA) (not connected)"
    )


def test_the_measured_output_latency_is_shown_while_a_song_plays(
    window, monkeypatch
):
    player = window.control_bar.player
    monkeypatch.setattr(type(player), "output_latency_ms", 12.5, raising=False)
    monkeypatch.setattr(type(player), "active_buffer_size", 1024, raising=False)
    player.playbackStateChanged.emit(1)
    text = window.lbl_audio_latency.text()
    assert "12.5 ms" in text and "1024" in text


def test_without_a_measurement_the_latency_line_says_when_it_appears(window):
    assert "while a song plays" in window.lbl_audio_latency.text()
