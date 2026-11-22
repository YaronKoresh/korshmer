import logging
import threading
import tracemalloc

import numpy as np
import psutil
import pytest
import soundfile as sf

from korshmer.audio import loader
from korshmer.audio.loader import StreamedAudio
from korshmer.constants import (
    MAX_PLAYBACK_BUFFER_SIZE,
    MIN_PLAYBACK_BUFFER_SIZE,
)
from korshmer.core import audio_engine
from korshmer.core.audio_engine import AudioEngine, PlaybackState, sd
from korshmer.event_bus import Event, subscribe, unsubscribe

RATE = 8000
FRAMES = 49152
BLOCK = 2048
PREFETCH_THREAD = "korshmer-audio-prefetch"


class DacTime:
    def __init__(self, seconds):
        self.outputBufferDacTime = seconds


class WatchedFile(StreamedAudio):
    def __init__(self, path):
        super().__init__(path)
        self.reads = []
        self.activity = threading.Event()
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.broken = False

    def readers(self):
        return {name for name, _, _ in self.reads}

    def __getitem__(self, key):
        name = threading.current_thread().name
        self.reads.append((name, key.start, key.stop))
        self.entered.set()
        if name == PREFETCH_THREAD:
            self.gate.wait(10)
        self.activity.set()
        if self.broken:
            raise OSError("the drive was unplugged")
        return super().__getitem__(key)


class FakeDevice:
    opened = []
    broken_devices = ()

    def __init__(self, **kwargs):
        if kwargs.get("device") in FakeDevice.broken_devices:
            raise sd.PortAudioError("device unavailable")
        self.kwargs = kwargs
        self.active = False
        self.closed = False
        self.latency = 0.025
        self.first_block = None
        FakeDevice.opened.append(self)

    def start(self):
        self.active = True
        out = np.zeros(
            (self.kwargs["blocksize"], self.kwargs["channels"]), "float32"
        )
        try:
            self.kwargs["callback"](out, len(out), None, None)
        except sd.CallbackStop:
            pass
        self.first_block = out

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


@pytest.fixture
def device(mock_qt_app, monkeypatch):
    FakeDevice.opened = []
    FakeDevice.broken_devices = ()
    monkeypatch.setattr(audio_engine.sd, "OutputStream", FakeDevice)
    monkeypatch.setattr(
        audio_engine.sd,
        "query_devices",
        lambda *args, **kwargs: {"default_samplerate": RATE},
    )
    monkeypatch.setattr(
        "korshmer.utils.audio_host.get_host_preference", lambda: "default"
    )
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 50.0)
    return FakeDevice


@pytest.fixture
def status_messages():
    seen = []

    def remember(text="", **extra):
        seen.append(text)

    subscribe(Event.STATUS_UPDATE, remember)
    yield seen
    unsubscribe(Event.STATUS_UPDATE, remember)


@pytest.fixture
def ramp():
    return np.linspace(0.1, 0.9, FRAMES, dtype="float32").reshape(-1, 1)


@pytest.fixture
def streamed(device, tmp_path, monkeypatch, ramp):
    path = tmp_path / "long_song.wav"
    sf.write(str(path), ramp, RATE, subtype="FLOAT")
    monkeypatch.setattr(loader, "STREAMING_AUDIO_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(loader, "StreamedAudio", WatchedFile)
    eng = AudioEngine()
    assert eng.load(str(path))
    yield eng
    eng.data.gate.set()
    eng.stop()
    eng.data.close()


def _memory_engine(data, fs=1000):
    eng = AudioEngine()
    eng.data = data
    eng.fs = fs
    return eng


def _callback(eng, frames, channels=1, when=None, status=None):
    out = np.full((frames, channels), 7.0, dtype="float32")
    stopped = False
    try:
        eng._cb(out, frames, when, status)
    except sd.CallbackStop:
        stopped = True
    return out, stopped


def _step(eng, block):
    before = eng.idx
    out, stopped = _callback(eng, block)
    if stopped:
        return out[: len(eng.data) - before], True
    span = len(eng.data)
    if eng.loop_start_idx is not None:
        span = eng.loop_end_idx - eng.loop_start_idx
    return out[: (eng.idx - before) % span], False


def _hear(eng, frames_wanted, block=BLOCK):
    heard = [np.zeros((0, 1), "float32")]
    got = 0
    for _ in range(4000):
        eng.data.activity.clear()
        chunk, stopped = _step(eng, block)
        heard.append(chunk)
        got += len(chunk)
        if stopped or got >= frames_wanted:
            break
        if not len(chunk):
            eng.data.activity.wait(0.01)
    return np.concatenate(heard)


def _thread_named(name):
    return [t for t in threading.enumerate() if t.name == name]


def test_playback_that_drifts_past_the_end_of_the_song_ends_in_silence(
    mock_qt_app,
):
    eng = _memory_engine(np.ones((1000, 1), dtype="float32"))
    _callback(eng, 100, when=DacTime(0.0))

    out, stopped = _callback(eng, 100, when=DacTime(1.05))

    assert stopped
    assert not out.any()
    assert eng.position() == eng.duration()


def test_steady_playback_does_not_allocate_a_new_block_for_every_callback(
    mock_qt_app,
):
    block = 4096
    eng = _memory_engine(
        np.full((block * 40, 2), 0.5, dtype="float32"), fs=48000
    )
    eng.setVolume(0.8)
    out = np.zeros((block, 2), dtype="float32")
    eng._cb(out, block, None, None)

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for _ in range(30):
            eng._cb(out, block, None, None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak - before < out.nbytes // 2
    assert np.allclose(out, 0.4)


def test_the_audio_callback_leaves_status_reporting_to_the_timer(
    mock_qt_app, caplog
):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))

    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            _callback(eng, 10, status="output underflow")
        during_callbacks = len(caplog.records)
        eng.timer.timeout.emit()

    reports = [
        r.message for r in caplog.records if "Audio stream status" in r.message
    ]
    assert during_callbacks == 0
    assert len(reports) == 1
    assert "output underflow" in reports[0] and "3" in reports[0]
    assert eng.dropout_count == 3


def test_a_burst_of_dropouts_is_logged_once_not_on_every_timer_tick(
    mock_qt_app, caplog
):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))

    with caplog.at_level(logging.WARNING, logger="korshmer.performance"):
        for _ in range(5):
            _callback(eng, 10, status="output underflow")
            eng.timer.timeout.emit()

    reports = [r for r in caplog.records if "dropouts" in r.message]
    assert len(reports) == 1


def test_a_streamed_song_is_only_ever_read_by_the_prefetch_thread(
    streamed, ramp
):
    streamed.play()

    heard = _hear(streamed, FRAMES)

    assert np.allclose(heard, ramp[BLOCK:])
    assert streamed.data.readers() == {PREFETCH_THREAD}


def test_the_first_block_of_a_streamed_song_is_ready_when_the_device_starts(
    streamed, ramp, device
):
    streamed.play()

    first = device.opened[-1].first_block
    assert np.allclose(first, ramp[: len(first)])
    assert streamed.dropout_count == 0


def test_a_streamed_song_waits_in_silence_when_the_disk_is_behind(streamed):
    streamed.play()
    _hear(streamed, BLOCK)
    streamed.data.gate.clear()
    for _ in range(12):
        _callback(streamed, BLOCK)
    position = streamed.idx

    out, _ = _callback(streamed, BLOCK)

    assert not out.any()
    assert streamed.idx == position
    assert streamed.dropout_count > 0
    assert streamed.data.readers() == {PREFETCH_THREAD}


def test_a_disk_slower_than_the_callback_still_delivers_the_whole_song(
    streamed, ramp
):
    streamed.data.gate.clear()
    streamed.play()
    heard = []
    for _ in range(FRAMES // BLOCK + 1):
        for _ in range(3):
            heard.append(_step(streamed, BLOCK)[0])
        streamed.data.gate.set()
        heard.append(_hear(streamed, BLOCK))
        streamed.data.gate.clear()
        if streamed.idx >= FRAMES:
            break

    starts = [start for _, start, _ in streamed.data.reads]
    assert np.allclose(np.concatenate(heard), ramp)
    assert len(starts) == len(set(starts))


def test_a_streamed_song_survives_a_device_that_asks_for_odd_block_sizes(
    streamed, ramp
):
    streamed.play()

    heard = _hear(streamed, FRAMES, block=700)

    assert np.allclose(heard, ramp[BLOCK:])


def test_starting_and_seeking_a_streamed_song_are_not_counted_as_dropouts(
    streamed,
):
    streamed.data.gate.clear()
    streamed.play()
    for _ in range(4):
        _callback(streamed, BLOCK)
    for ms in (500, 900, 300, 1200):
        streamed.setPosition(ms)
        for _ in range(4):
            _callback(streamed, BLOCK)
    streamed.timer.timeout.emit()

    assert streamed.dropout_count == 0
    assert streamed.active_buffer_size == BLOCK


def test_a_streamed_song_that_can_no_longer_be_read_stops_with_a_message(
    streamed, caplog, status_messages
):
    streamed.play()
    streamed.setPosition(1000)
    streamed.data.broken = True
    streamed.data.entered.clear()
    streamed.setPosition(0)
    assert streamed.data.entered.wait(10)

    with caplog.at_level(logging.ERROR):
        for _ in range(2000):
            streamed.timer.timeout.emit()
            if streamed.playbackState() == PlaybackState.STOPPED:
                break
            streamed.data.activity.wait(0.005)
        for _ in range(5):
            streamed.timer.timeout.emit()

    errors = [r for r in caplog.records if "can no longer be read" in r.message]
    assert streamed.playbackState() == PlaybackState.STOPPED
    assert len(errors) == 1
    assert any("can no longer be read" in text for text in status_messages)


def test_setting_a_loop_on_a_streamed_song_does_not_read_on_the_callers_thread(
    streamed,
):
    streamed.data.gate.clear()

    assert streamed.set_loop(100, 200)

    for loader_thread in _thread_named("korshmer-loop-loader"):
        loader_thread.join(10)
    assert "MainThread" not in streamed.data.readers()


def test_a_loaded_loop_of_a_streamed_song_wraps_without_touching_the_disk(
    streamed, ramp
):
    assert streamed.set_loop(100, 200)
    for loader_thread in _thread_named("korshmer-loop-loader"):
        loader_thread.join(10)
    reads_before = len(streamed.data.reads)

    out, _ = _callback(streamed, 2000)

    region = ramp[800:1600]
    assert len(streamed.data.reads) == reads_before
    assert np.allclose(out[:800], region)
    assert np.allclose(out[800:1600], region)
    assert np.allclose(out[1600:], region[:400])


def test_a_loop_too_big_to_keep_in_memory_is_fetched_round_without_a_restart(
    streamed, ramp, monkeypatch
):
    monkeypatch.setattr(audio_engine, "SEGMENT_CACHE_MAX_BYTES", 0)
    streamed.play()
    assert streamed.set_loop(100, 600)
    streamed.setPosition(100)
    del streamed.data.reads[:]

    heard = _hear(streamed, 12000, block=1000)

    region = ramp[800:4800]
    fetched = [(start, stop) for _, start, stop in streamed.data.reads]
    in_flight = [fetched.pop(0)] if fetched[0][0] != 800 else []
    assert np.allclose(heard[:12000], np.tile(region, (3, 1)))
    assert len(in_flight) <= 1
    assert fetched[:6] == [(800, 2848), (2848, 4800)] * 3
    assert streamed.data.readers() == {PREFETCH_THREAD}


def test_the_chosen_buffer_size_is_used_when_the_stream_opens(device):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    eng.set_preferred_buffer_size(1024)

    eng.play()
    try:
        assert device.opened[-1].kwargs["blocksize"] == 1024
        assert eng.active_buffer_size == 1024
    finally:
        eng.stop()


def test_a_chosen_buffer_size_is_never_changed_behind_the_users_back(
    device, monkeypatch
):
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 99.0)
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    eng.set_preferred_buffer_size(1024)

    for _ in range(6):
        _callback(eng, 10, status="output underflow")
        eng.timer.timeout.emit()

    assert eng.active_buffer_size == 1024
    assert eng.dropout_count == 6


@pytest.mark.parametrize(
    "frames",
    [MIN_PLAYBACK_BUFFER_SIZE - 1, MAX_PLAYBACK_BUFFER_SIZE + 1, 1024.0, "big"],
)
def test_an_impossible_buffer_size_is_refused(mock_qt_app, frames):
    eng = AudioEngine()
    with pytest.raises(ValueError):
        eng.set_preferred_buffer_size(frames)
    assert eng.active_buffer_size == 2048


def test_clearing_the_chosen_buffer_size_returns_to_automatic_sizing(
    device, monkeypatch
):
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 99.0)
    eng = AudioEngine()
    eng.set_preferred_buffer_size(4096)
    eng.set_preferred_buffer_size(None)
    before = eng.active_buffer_size

    eng.timer.timeout.emit()

    assert before == 2048
    assert eng.active_buffer_size == 4096


def test_the_chosen_output_device_is_the_one_opened(device):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    eng.set_output_device(3)

    eng.play()
    try:
        assert device.opened[-1].kwargs["device"] == 3
        assert eng.output_device == 3
    finally:
        eng.stop()


def test_without_a_chosen_device_the_host_default_is_opened(device):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    eng.set_output_device(3)
    eng.set_output_device(None)

    eng.play()
    try:
        assert "device" not in device.opened[-1].kwargs
    finally:
        eng.stop()


def test_a_chosen_device_that_will_not_open_falls_back_to_the_default_once(
    device, status_messages
):
    device.broken_devices = (3,)
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    eng.set_output_device(3)

    eng.play()
    playing = eng.playbackState()
    eng.stop()
    device.broken_devices = ()
    eng.play()
    eng.stop()

    assert playing == PlaybackState.PLAYING
    assert ["device" in d.kwargs for d in device.opened] == [False, False]
    assert eng.output_device is None
    assert sum("output device 3" in text for text in status_messages) == 1


@pytest.mark.parametrize("chosen", [-1, 1.5, True, "speakers"])
def test_a_nonsense_output_device_is_refused(mock_qt_app, chosen):
    eng = AudioEngine()
    with pytest.raises(ValueError):
        eng.set_output_device(chosen)


def test_output_latency_is_reported_in_milliseconds_while_playing(device):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    idle = eng.output_latency_ms

    eng.play()
    try:
        assert idle == 0.0
        assert eng.output_latency_ms == pytest.approx(25.0)
    finally:
        eng.stop()


def test_output_latency_is_reported_for_a_chosen_device_too(device):
    eng = _memory_engine(np.zeros((1000, 1), dtype="float32"))
    eng.set_output_device(3)

    eng.play()
    try:
        assert eng.output_latency_ms == pytest.approx(25.0)
    finally:
        eng.stop()
