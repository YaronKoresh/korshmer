from korshmer.midi.controller_output import (
    ControllerOutput,
    LedSpec,
    find_output_name,
)


class FakePort:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


def make_output(**kwargs):
    port = FakePort()
    out = ControllerOutput(open_output=lambda name: port, **kwargs)
    return out, port


def test_led_change_sends_exactly_one_message():
    out, port = make_output()
    assert out.open("Pioneer DDJ-400", "pioneer_ddj")
    assert out.set_led("play_2", True) is True
    assert port.sent == [
        {"type": "note_on", "channel": 1, "note": 0x0B, "velocity": 127}
    ]
    out.set_led("play_2", False)
    assert port.sent[-1] == {
        "type": "note_off",
        "channel": 1,
        "note": 0x0B,
        "velocity": 0,
    }
    assert len(port.sent) == 2


def test_cc_led_message():
    table = {"generic": {"mute": LedSpec("cc", 2, 20)}}
    out, port = make_output(led_map=table)
    out.open("X", "generic")
    out.set_led("mute", True)
    assert port.sent == [
        {"type": "control_change", "channel": 2, "control": 20, "value": 127}
    ]


def test_unknown_button_or_controller_sends_nothing():
    out, port = make_output()
    out.open("Generic", "generic")
    assert out.set_led("play_1", True) is False
    assert port.sent == []


def test_without_port_nothing_is_raised():
    out = ControllerOutput(open_output=lambda name: None)
    assert out.open("X", "pioneer_ddj") is False
    assert out.set_led("play_1", True) is False
    assert out.send_volume(0.5) is False


def test_open_failure_is_silent():
    def boom(name):
        raise OSError("no backend")

    out = ControllerOutput(open_output=boom)
    assert out.open("X", "generic") is False


def test_send_failure_is_silent():
    out, port = make_output()
    out.open("X", "generic")

    def boom(message):
        raise OSError("gone")

    port.send = boom
    assert out.send_volume(0.5) is False


def test_connect_sends_initial_led_state():
    out = ControllerOutput(open_output=lambda name: port)
    port = FakePort()
    out.set_led("play_1", True)
    out.set_led("cue_1", False)
    out.on_device_connected("Pioneer DDJ-400", "pioneer_ddj")
    assert {m["note"]: m["type"] for m in port.sent} == {
        0x0B: "note_on",
        0x0C: "note_off",
    }


def test_disconnect_closes_port():
    out, port = make_output()
    out.on_device_connected("A", "generic")
    out.on_device_disconnected("B")
    assert out.is_open
    out.on_device_disconnected("A")
    assert port.closed and not out.is_open


def test_volume_sends_one_cc_message():
    out, port = make_output()
    out.open("X", "generic")
    assert out.send_volume(0.5) is True
    assert port.sent == [
        {"type": "control_change", "channel": 0, "control": 7, "value": 64}
    ]


def test_volume_clamped_and_invalid_ignored():
    out, port = make_output()
    out.open("X", "generic")
    out.send_volume(3.0)
    assert port.sent[-1]["value"] == 127
    assert out.send_volume(float("nan")) is False
    assert out.send_volume("abc") is False
    assert len(port.sent) == 1


def test_volume_from_controller_is_not_sent_back():
    out, port = make_output()
    out.open("X", "generic")
    out.volume_received(0.5)
    assert out.send_volume(0.5) is False
    assert port.sent == []
    assert out.send_volume(0.5) is True


def test_led_from_controller_is_not_sent_back():
    out, port = make_output()
    out.open("Pioneer DDJ-400", "pioneer_ddj")
    out.note_received("play_1", True)
    assert out.set_led("play_1", True) is False
    assert port.sent == []


def test_pitch_bend_fader_protocol():
    out, port = make_output(fader_protocols={"generic": "pitch_bend"})
    out.open("X", "generic")
    out.send_volume(1.0)
    assert port.sent == [{"type": "pitchwheel", "channel": 0, "pitch": 8191}]


def test_find_output_name():
    assert find_output_name("A", ["B", "A"]) == "A"
    assert (
        find_output_name("DDJ-400 MIDI 1", ["DDJ-400 MIDI 1"])
        == "DDJ-400 MIDI 1"
    )
    assert find_output_name("A", ["B"]) is None


def test_engine_volume_signal_drives_fader():
    from korshmer.core.audio_engine import AudioEngine

    out, port = make_output()
    out.open("X", "generic")
    engine = AudioEngine()
    out.connect_engine(engine)
    engine.setVolume(0.5)
    assert len(port.sent) == 1 and port.sent[0]["value"] == 64
