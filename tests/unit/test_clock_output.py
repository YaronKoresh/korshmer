import time

from korshmer.midi.clock_output import (
    STATE_PAUSED,
    STATE_PLAYING,
    STATE_STOPPED,
    MidiClockOutput,
)


class FakePort:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message["type"])

    def close(self):
        self.closed = True


def make_clock(bpm=120):
    port = FakePort()
    out = MidiClockOutput(open_output=lambda name: port)
    out.open("dev")
    out.set_tempo(bpm)
    return out, port


def test_open_returns_false_without_device():
    out = MidiClockOutput(open_output=lambda name: None)
    assert out.open("x") is False
    assert not out.is_open


def test_interval_follows_bpm():
    out, _ = make_clock(120)
    assert abs(out.interval - 60 / (120 * 24)) < 1e-9
    out.set_tempo(60)
    assert abs(out.interval - 60 / (60 * 24)) < 1e-9


def test_invalid_bpm_keeps_previous():
    out, _ = make_clock(90)
    out.set_tempo("bad")
    assert out.bpm == 90


def test_start_then_clock_count_at_120_bpm():
    out, port = make_clock(120)
    out.on_playback_state_changed(STATE_PLAYING)
    time.sleep(1.0)
    out.on_playback_state_changed(STATE_STOPPED)
    assert port.sent[0] == "start"
    assert port.sent[-1] == "stop"
    clocks = port.sent.count("clock")
    assert 47 <= clocks <= 50


def test_pause_sends_stop_and_resume_sends_continue():
    out, port = make_clock(300)
    out.on_playback_state_changed(STATE_PLAYING)
    out.on_playback_state_changed(STATE_PAUSED)
    assert port.sent[-1] == "stop"
    count = len(port.sent)
    time.sleep(0.1)
    assert len(port.sent) == count
    out.on_playback_state_changed(STATE_PLAYING)
    assert port.sent[count] == "continue"
    out.on_playback_state_changed(STATE_STOPPED)
    assert not out.is_running


def test_stop_after_pause_sends_no_extra_stop():
    out, port = make_clock(300)
    out.on_playback_state_changed(STATE_PLAYING)
    out.on_playback_state_changed(STATE_PAUSED)
    n = port.sent.count("stop")
    out.on_playback_state_changed(STATE_STOPPED)
    assert port.sent.count("stop") == n


def test_tempo_change_applies_immediately():
    out, port = make_clock(20)
    out.on_playback_state_changed(STATE_PLAYING)
    out.set_tempo(300)
    time.sleep(0.5)
    out.on_playback_state_changed(STATE_STOPPED)
    assert port.sent.count("clock") >= 8


def test_close_closes_port_and_stops_thread():
    out, port = make_clock(300)
    out.on_playback_state_changed(STATE_PLAYING)
    out.close()
    assert port.closed
    assert not out.is_running
