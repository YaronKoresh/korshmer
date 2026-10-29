import time

import pytest
from PySide6.QtCore import QCoreApplication

from korshmer.midi.hotplug import MidiHotplugMonitor


class FakeDevices:
    def __init__(self, names=()):
        self.names = list(names)

    def __call__(self):
        return list(self.names)


def make_monitor(devices):
    monitor = MidiHotplugMonitor(list_devices=devices, interval=0.01)
    connected, disconnected = [], []
    monitor.device_connected.connect(lambda n, t: connected.append((n, t)))
    monitor.device_disconnected.connect(disconnected.append)
    return monitor, connected, disconnected


def test_connect_emits_once():
    devices = FakeDevices()
    monitor, connected, disconnected = make_monitor(devices)
    monitor.poll()
    assert connected == [] and disconnected == []

    devices.names = ["ROLI Seaboard Block"]
    monitor.poll()
    monitor.poll()
    assert connected == [("ROLI Seaboard Block", "roli_seaboard")]
    assert disconnected == []
    assert monitor.connected_devices == ["ROLI Seaboard Block"]


def test_disconnect_emits_once():
    devices = FakeDevices(["ROLI Seaboard Block"])
    monitor, connected, disconnected = make_monitor(devices)
    monitor.poll()
    devices.names = []
    monitor.poll()
    monitor.poll()
    assert disconnected == ["ROLI Seaboard Block"]
    assert len(connected) == 1
    assert monitor.connected_devices == []


def test_identical_list_emits_nothing():
    devices = FakeDevices(["Unknown Keys"])
    monitor, connected, disconnected = make_monitor(devices)
    monitor.poll()
    connected.clear()
    monitor.poll()
    monitor.poll()
    assert connected == [] and disconnected == []


def test_new_device_gets_mapping_from_controller_mapping():
    devices = FakeDevices(["LinnStrument MIDI 1", "Unknown Keys"])
    monitor, _, _ = make_monitor(devices)
    monitor.poll()
    assert monitor.mapping_for("LinnStrument MIDI 1").controller_type == (
        "linnstrument"
    )
    assert monitor.mapping_for("Unknown Keys").controller_type == "generic"
    devices.names = ["Unknown Keys"]
    monitor.poll()
    assert monitor.mapping_for("LinnStrument MIDI 1") is None


def test_listing_error_keeps_state():
    devices = FakeDevices(["A"])
    monitor, connected, disconnected = make_monitor(devices)
    monitor.poll()

    def boom():
        raise OSError("backend gone")

    monitor._list_devices = boom
    monitor.poll()
    assert disconnected == []
    assert monitor.connected_devices == ["A"]


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        MidiHotplugMonitor(list_devices=FakeDevices(), interval=0)


def test_background_thread_detects_and_stops():
    devices = FakeDevices()
    app = QCoreApplication.instance() or QCoreApplication([])
    monitor, connected, _ = make_monitor(devices)
    monitor.start()
    try:
        devices.names = ["Haken Audio Continuum"]
        deadline = time.time() + 2
        while not connected and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
    finally:
        monitor.stop()
    assert connected == [("Haken Audio Continuum", "haken_continuum")]
    assert monitor._thread is None
