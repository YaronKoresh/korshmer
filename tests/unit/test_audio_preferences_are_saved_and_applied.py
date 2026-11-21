import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.core import control_bar as control_bar_module
from korshmer.core.audio_engine import AudioEngine
from korshmer.utils import audio_host

DEVICES = [
    {"name": "Built-in Mic", "max_output_channels": 0, "hostapi": 0},
    {"name": "Speakers", "max_output_channels": 2, "hostapi": 0},
    {"name": "Studio Interface", "max_output_channels": 8, "hostapi": 1},
]


class FakeSoundDevice:
    def __init__(self, devices=DEVICES, broken=False):
        self.devices = devices
        self.broken = broken
        self.queries = 0

    def query_devices(self):
        self.queries += 1
        if self.broken:
            raise RuntimeError("PortAudio not initialised")
        return self.devices

    def query_hostapis(self):
        return [{"name": "ALSA"}, {"name": "JACK"}]


class FakeEngine:
    def __init__(self):
        self.buffer_sizes = []
        self.devices = []

    def set_preferred_buffer_size(self, frames):
        self.buffer_sizes.append(frames)

    def set_output_device(self, index):
        self.devices.append(index)


class RememberingEngine(AudioEngine):
    def set_preferred_buffer_size(self, frames):
        self.asked_buffer = frames

    def set_output_device(self, index):
        self.asked_device = index


@pytest.fixture(autouse=True)
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "saved_data" / "audio_host.json"
    monkeypatch.setattr(audio_host, "AUDIO_HOST_FILE", path)
    audio_host.reset_host_preference_cache()
    yield path
    audio_host.reset_host_preference_cache()


def write_settings(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def saved_choices():
    return (
        audio_host.get_buffer_size_preference(),
        audio_host.get_output_device_preference(),
    )


def test_a_chosen_buffer_size_is_remembered():
    audio_host.set_buffer_size_preference(4096)
    assert audio_host.get_buffer_size_preference() == 4096


def test_without_a_choice_the_buffer_size_is_automatic():
    assert audio_host.get_buffer_size_preference() is None


def test_a_buffer_size_the_engine_does_not_allow_is_not_saved():
    audio_host.set_buffer_size_preference(1000)
    assert audio_host.get_buffer_size_preference() is None


def test_a_chosen_output_device_is_remembered():
    audio_host.set_output_device_preference("Speakers (ALSA)")
    assert audio_host.get_output_device_preference() == "Speakers (ALSA)"


def test_choosing_the_system_default_device_forgets_the_saved_device():
    audio_host.set_output_device_preference("Speakers (ALSA)")
    audio_host.set_output_device_preference(None)
    assert audio_host.get_output_device_preference() is None


def test_a_settings_file_from_an_older_version_still_loads(
    settings_file, monkeypatch
):
    monkeypatch.setattr(audio_host, "jack_supported", lambda: True)
    write_settings(settings_file, json.dumps({"host": "jack"}))
    assert (audio_host.get_host_preference(), saved_choices()) == (
        "jack",
        (None, None),
    )


def test_saving_a_buffer_size_keeps_the_saved_audio_interface(monkeypatch):
    monkeypatch.setattr(audio_host, "jack_supported", lambda: True)
    audio_host.set_host_preference("jack")
    audio_host.set_buffer_size_preference(1024)
    audio_host.reset_host_preference_cache()
    assert audio_host.get_host_preference() == "jack"


def test_saving_the_audio_interface_keeps_the_saved_buffer_size(monkeypatch):
    monkeypatch.setattr(audio_host, "jack_supported", lambda: True)
    audio_host.set_buffer_size_preference(1024)
    audio_host.set_host_preference("jack")
    assert audio_host.get_buffer_size_preference() == 1024


def test_a_damaged_settings_file_reads_as_no_choices(settings_file):
    write_settings(settings_file, "{oops")
    assert saved_choices() == (None, None)


def test_a_settings_file_with_wrong_value_types_reads_as_no_choices(
    settings_file,
):
    write_settings(
        settings_file, json.dumps({"buffer_size": "big", "output_device": 5})
    )
    assert saved_choices() == (None, None)


def test_the_buffer_choices_are_the_sizes_the_engine_allows():
    assert audio_host.BUFFER_SIZE_CHOICES == (512, 1024, 2048, 4096, 8192)


def test_only_devices_that_can_play_sound_are_offered():
    found = audio_host.list_output_devices(FakeSoundDevice())
    assert found == [(1, "Speakers (ALSA)"), (2, "Studio Interface (JACK)")]


def test_no_devices_are_offered_when_the_sound_system_is_unavailable():
    assert audio_host.list_output_devices(FakeSoundDevice(broken=True)) == []


def test_the_saved_buffer_size_is_given_to_the_player():
    engine = FakeEngine()
    audio_host.set_buffer_size_preference(4096)
    audio_host.apply_audio_preferences(engine, FakeSoundDevice())
    assert engine.buffer_sizes == [4096]


def test_the_saved_output_device_is_given_to_the_player():
    engine = FakeEngine()
    audio_host.set_output_device_preference("Studio Interface (JACK)")
    audio_host.apply_audio_preferences(engine, FakeSoundDevice())
    assert engine.devices == [2]


def test_a_saved_device_that_is_unplugged_falls_back_to_the_default_device():
    engine = FakeEngine()
    audio_host.set_output_device_preference("Old USB Interface (ALSA)")
    audio_host.apply_audio_preferences(engine, FakeSoundDevice())
    assert engine.devices == [None]


def test_the_user_is_told_once_that_the_saved_device_is_missing():
    audio_host.set_output_device_preference("Old USB Interface (ALSA)")
    notices = [
        audio_host.apply_audio_preferences(FakeEngine(), FakeSoundDevice())
        for _ in range(3)
    ]
    assert "Old USB Interface (ALSA)" in notices[0]
    assert notices[1:] == [None, None]


def test_nothing_is_reported_when_the_saved_device_is_present():
    audio_host.set_output_device_preference("Speakers (ALSA)")
    assert (
        audio_host.apply_audio_preferences(FakeEngine(), FakeSoundDevice())
        is None
    )


def test_without_saved_choices_the_player_keeps_its_automatic_settings():
    engine = FakeEngine()
    audio_host.apply_audio_preferences(engine, FakeSoundDevice())
    assert (engine.buffer_sizes, engine.devices) == ([None], [None])


def test_without_a_saved_device_the_sound_system_is_not_asked():
    sound = FakeSoundDevice()
    audio_host.apply_audio_preferences(FakeEngine(), sound)
    assert sound.queries == 0


def test_a_player_without_device_settings_is_left_alone():
    audio_host.set_buffer_size_preference(4096)
    assert (
        audio_host.apply_audio_preferences(object(), FakeSoundDevice()) is None
    )


def test_a_new_player_bar_starts_with_the_saved_buffer_size(monkeypatch):
    monkeypatch.setattr(control_bar_module, "AudioEngine", RememberingEngine)
    audio_host.set_buffer_size_preference(4096)
    bar = control_bar_module.ControlBar()
    bar.sys_timer.stop()
    asked = bar.player.asked_buffer
    bar.close()
    assert asked == 4096
