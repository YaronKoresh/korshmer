import threading
import time

import numpy as np
import sounddevice as sd

from korshmer.core.audio_engine import AudioEngine, PlaybackState


class DummyTime:
    def __init__(self, dac_time):
        self.outputBufferDacTime = dac_time


def test_set_position_resets_timebase_and_emits_signal(monkeypatch):
    eng = AudioEngine()
    eng.data = np.zeros((1000, 1), dtype="float32")
    eng.fs = 1000

    eng.stream = object()

    recorded = []

    def on_pos(ms):
        recorded.append(ms)

    eng.positionChanged.connect(on_pos)

    eng.setPosition(500)
    assert eng.idx == 500

    assert eng._stream_start_time is None

    assert recorded == [eng.position()]


def test_seek_during_playback_does_not_snap_back(monkeypatch):
    eng = AudioEngine()
    eng.data = np.zeros((2000, 1), dtype="float32")
    eng.fs = 1000

    eng._stream_start_time = 0.0
    eng.idx = 100
    eng.stream = object()

    eng.setPosition(1500)
    assert eng.idx == 1500

    assert eng._stream_start_time is None

    out = np.zeros((100, 1), dtype="float32")
    eng._cb(out, 100, DummyTime(10.0), None)

    assert eng._stream_start_time == 10.0
    assert eng.idx == 1500 + len(out)


def test_playback_state_transitions(monkeypatch):

    class DummyStream:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "korshmer.core.audio_engine.sd.OutputStream",
        DummyStream,
    )

    eng = AudioEngine()
    eng.data = np.zeros((10, 1), dtype="float32")
    eng.fs = 10

    eng.play()
    assert eng.playbackState() == PlaybackState.PLAYING
    eng.pause()
    assert eng.playbackState() == PlaybackState.PAUSED
    eng.stop()
    assert eng.playbackState() == PlaybackState.STOPPED


def test_control_bar_slider_seeks_engine(qapp, monkeypatch):
    from korshmer.core.control_bar import ControlBar

    cb = ControlBar()

    class DummyPlayer:
        def __init__(self):
            self.requested = None

        def setPosition(self, v):
            self.requested = v

        def duration(self):
            return 1000

    dummy = DummyPlayer()
    cb.player = dummy

    cb.slider_seek.setRange(0, 1000)
    cb.slider_seek.setValue(123)

    cb.on_slider_released()
    assert dummy.requested == 123


class _FakeTimer:
    def start(self):
        pass

    def stop(self):
        pass


def test_stop_during_slow_stream_start_leaves_no_active_stream(monkeypatch):

    import sounddevice as sd

    created = []

    class SlowStream:
        def __init__(self, **kwargs):
            time.sleep(0.2)
            self.open = True
            created.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            self.open = False

    monkeypatch.setattr(sd, "OutputStream", SlowStream)

    eng = AudioEngine()
    eng.timer = _FakeTimer()
    eng.data = np.zeros((44100, 1), dtype="float32")
    eng.fs = 44100

    worker = threading.Thread(target=eng.play)
    worker.start()
    time.sleep(0.05)
    eng.stop()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert eng.playbackState() == PlaybackState.STOPPED
    assert eng.stream is None
    assert all(not s.open for s in created)


def _collect_playback(eng, block, after_blocks=None, on_block=None):
    chunks = []
    while True:
        out = np.zeros((block, eng.data.shape[1]), dtype="float32")
        try:
            eng._cb(out, block, None, None)
        except sd.CallbackStop:
            chunks.append(out)
            break
        chunks.append(out)
        if on_block is not None and len(chunks) == after_blocks:
            on_block(len(chunks))
        time.sleep(0.0005)
    return np.concatenate(chunks)


def test_playback_output_matches_source_through_prefetch(monkeypatch):
    created = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "OutputStream", FakeStream)

    rng = np.random.default_rng(0)
    data = rng.standard_normal((44100 * 2, 2)).astype("float32")
    eng = AudioEngine()
    eng.timer = _FakeTimer()
    eng.data = data
    eng.fs = 44100
    eng._current_buffer_size = 512

    eng.play()
    rendered = _collect_playback(eng, 512)

    hits = eng._prefetch.hits
    eng.stop()

    assert hits > 0
    assert np.array_equal(rendered[: len(data)], data)
    assert not rendered[len(data) :].any()


def test_seek_during_prefetched_playback_plays_from_new_position(monkeypatch):
    class FakeStream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "OutputStream", FakeStream)

    rng = np.random.default_rng(1)
    data = rng.standard_normal((44100 * 2, 2)).astype("float32")
    eng = AudioEngine()
    eng.timer = _FakeTimer()
    eng.data = data
    eng.fs = 44100
    eng._current_buffer_size = 512

    eng.play()
    seek_to = {}

    def seek(blocks_rendered):
        eng.setPosition(250)
        seek_to["frame"] = eng.idx
        seek_to["offset"] = blocks_rendered * 512

    rendered = _collect_playback(eng, 512, after_blocks=20, on_block=seek)
    eng.stop()

    start = seek_to["frame"]
    offset = seek_to["offset"]
    expected = data[start:]
    played = rendered[offset : offset + len(expected)]
    assert np.array_equal(played, expected)
    assert not rendered[offset + len(expected) :].any()


def _retry_engine(monkeypatch, fail_times):
    import sounddevice as sd

    calls = []

    class FlakyStream:
        def __init__(self, **kwargs):
            calls.append(1)
            if len(calls) <= fail_times:
                raise sd.PortAudioError("Device unavailable")

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "OutputStream", FlakyStream)
    monkeypatch.setattr(
        "korshmer.core.audio_engine.AUDIO_DEVICE_RETRY_DELAY_SECONDS", 0.01
    )
    eng = AudioEngine()
    eng.timer = _FakeTimer()
    eng.data = np.zeros((44100, 1), dtype="float32")
    eng.fs = 44100
    return eng, calls


def test_stream_open_retries_then_succeeds(monkeypatch):
    eng, calls = _retry_engine(monkeypatch, 2)
    eng.play()
    assert eng.playbackState() == PlaybackState.PLAYING
    assert len(calls) == 3
    eng.stop()


def test_stream_open_gives_up_after_max_attempts(monkeypatch):
    eng, calls = _retry_engine(monkeypatch, 99)
    eng.play()
    assert eng.playbackState() == PlaybackState.STOPPED
    assert len(calls) == 3
    assert eng.stream is None


def test_stop_during_retry_delay_cancels_further_attempts(monkeypatch):
    eng, calls = _retry_engine(monkeypatch, 99)
    monkeypatch.setattr(
        "korshmer.core.audio_engine.AUDIO_DEVICE_RETRY_DELAY_SECONDS", 2.0
    )
    worker = threading.Thread(target=eng.play)
    worker.start()
    time.sleep(0.1)
    eng.stop()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(calls) == 1
    assert eng.playbackState() == PlaybackState.STOPPED
    assert eng.stream is None


def _run_callbacks(eng, blocks, block, dac_start=10.0):
    out = np.zeros((block, 1), dtype="float32")
    for i in range(blocks):
        eng._cb(out, block, DummyTime(dac_start + i * block / eng.fs), None)


def test_resume_after_pause_continues_from_paused_position():
    eng = AudioEngine()
    eng.fs = 1000
    eng.data = np.zeros((20000, 1), dtype="float32")
    eng.idx = 5000
    eng._stream_start_time = None

    _run_callbacks(eng, blocks=5, block=100)

    assert eng.idx == 5500


def test_seek_while_playing_continues_from_new_position():
    eng = AudioEngine()
    eng.fs = 1000
    eng.data = np.zeros((20000, 1), dtype="float32")
    eng.stream = object()
    eng.idx = 0
    _run_callbacks(eng, blocks=3, block=100)

    eng.setPosition(12000)
    _run_callbacks(eng, blocks=5, block=100, dac_start=11.0)

    assert eng.idx == 12500


def test_drift_correction_still_snaps_to_the_device_clock():
    eng = AudioEngine()
    eng.fs = 1000
    eng.data = np.zeros((20000, 1), dtype="float32")
    eng.idx = 3000
    _run_callbacks(eng, blocks=1, block=100)
    out = np.zeros((100, 1), dtype="float32")
    eng.idx += 400
    eng._cb(out, 100, DummyTime(10.0 + 0.1), None)

    assert abs(eng.idx - 3200) <= 1
