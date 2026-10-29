import numpy as np
import pytest
import soundfile as sf

from korshmer.ui.widgets.arranger import timeline
from korshmer.utils import audio_host

RATE = 44100


class FakeInput:
    def __init__(self, recorder_box, audio, latency, **kwargs):
        self.callback = kwargs["callback"]
        self.audio = audio
        self.latency = latency
        self.recorder_box = recorder_box

    def __enter__(self):
        recorder = self.recorder_box[0]
        for start in range(0, len(self.audio), 1024):
            self.callback(self.audio[start : start + 1024], 1024, None, None)
        recorder.is_recording = False
        return self

    def __exit__(self, *exc):
        return False


class FakeSD:
    PortAudioError = Exception

    def __init__(self, recorder_box, audio, latency):
        self.recorder_box = recorder_box
        self.audio = audio
        self.latency = latency

    def InputStream(self, **kwargs):
        return FakeInput(self.recorder_box, self.audio, self.latency, **kwargs)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_host, "AUDIO_HOST_FILE", tmp_path / "host.json")
    audio_host.reset_host_preference_cache()
    monkeypatch.setattr(audio_host, "_output_ref", None)
    yield
    audio_host.reset_host_preference_cache()


def record(monkeypatch, tmp_path, audio, latency):
    box = []
    monkeypatch.setattr(timeline, "sd", FakeSD(box, audio, latency))
    rec = timeline.AudioRecorderThread(str(tmp_path / "rec.wav"))
    box.append(rec)
    rec.run()
    data, rate = sf.read(str(tmp_path / "rec.wav"))
    assert rate == RATE
    return data


def test_input_latency_trims_leading_frames(monkeypatch, tmp_path):
    audio = np.random.default_rng(0).random((RATE, 1)).astype("float32")
    data = record(monkeypatch, tmp_path, audio, latency=0.05)
    assert len(data) == RATE - 2205


def test_zero_latency_keeps_whole_recording(monkeypatch, tmp_path):
    audio = np.ones((RATE, 1), dtype="float32") * 0.5
    data = record(monkeypatch, tmp_path, audio, latency=0.0)
    assert len(data) == RATE


def test_output_latency_of_active_playback_is_added(monkeypatch, tmp_path):
    class Playing:
        latency = 0.05
        active = True
        closed = False

    class OutSD:
        PortAudioError = Exception

        def OutputStream(self, **kwargs):
            return Playing()

    stream = audio_host.open_output_stream(OutSD(), samplerate=RATE)
    audio = np.ones((RATE, 1), dtype="float32")
    data = record(monkeypatch, tmp_path, audio, latency=0.05)
    assert len(data) == RATE - 4410
    stream.active = False
    data = record(monkeypatch, tmp_path, audio, latency=0.05)
    assert len(data) == RATE - 2205


def test_recording_shorter_than_latency_is_kept(monkeypatch, tmp_path):
    audio = np.ones((1000, 1), dtype="float32")
    data = record(monkeypatch, tmp_path, audio, latency=0.05)
    assert len(data) == 1000


def test_click_recorded_during_click_playback_is_aligned(monkeypatch, tmp_path):
    in_lat, out_lat = 0.03, 0.04

    class Playing:
        latency = out_lat
        active = True
        closed = False

    class OutSD:
        PortAudioError = Exception

        def OutputStream(self, **kwargs):
            return Playing()

    keep = audio_host.open_output_stream(OutSD(), samplerate=RATE)

    delay = int(round((in_lat + out_lat) * RATE))
    audio = np.zeros((RATE, 1), dtype="float32")
    audio[delay, 0] = 1.0
    data = record(monkeypatch, tmp_path, audio, latency=in_lat)
    offset_s = int(np.argmax(np.abs(data))) / RATE
    assert offset_s < 0.005
    assert keep is not None


class FakeDuplex(FakeInput):
    def __init__(self, recorder_box, audio, latency, **kwargs):
        super().__init__(recorder_box, audio, latency, **kwargs)
        self.kwargs = kwargs
        self.outputs = []

    def __enter__(self):
        recorder = self.recorder_box[0]
        for start in range(0, len(self.audio), 1024):
            block = self.audio[start : start + 1024]
            out = np.zeros_like(block)
            self.callback(block, out, len(block), None, None)
            self.outputs.append(out)
        recorder.is_recording = False
        self.recorder_box.append(self)
        return self


class FakeDuplexSD(FakeSD):
    def Stream(self, **kwargs):
        requested = kwargs.pop("latency")
        stream = FakeDuplex(
            self.recorder_box, self.audio, self.latency, **kwargs
        )
        stream.kwargs["latency"] = requested
        return stream


def monitored(monkeypatch, tmp_path, audio, effects):
    box = []
    monkeypatch.setattr(timeline, "sd", FakeDuplexSD(box, audio, 0.0))
    rec = timeline.AudioRecorderThread(
        str(tmp_path / "rec.wav"), monitoring=True, monitor_effects=effects
    )
    box.append(rec)
    rec.run()
    stream = box[1]
    data, _ = sf.read(
        str(tmp_path / "rec.wav"), dtype="float32", always_2d=True
    )
    return data, np.concatenate(stream.outputs), stream.kwargs


def test_monitor_empty_chain_passes_input_through(monkeypatch, tmp_path):
    audio = np.random.default_rng(1).random((4096, 1)).astype("float32")
    data, out, kwargs = monitored(monkeypatch, tmp_path, audio, [])
    np.testing.assert_allclose(out, audio)
    np.testing.assert_allclose(data, audio, atol=1e-4)
    assert kwargs["latency"] == "low"


def test_monitor_gain_chain_is_wet_and_file_stays_dry(monkeypatch, tmp_path):
    audio = np.random.default_rng(2).random((4096, 1)).astype("float32") * 0.4
    data, out, _ = monitored(monkeypatch, tmp_path, audio, [lambda a: a * 2])
    np.testing.assert_allclose(out, audio * 2)
    np.testing.assert_allclose(data, audio, atol=1e-4)


def test_monitor_effect_error_plays_silence(monkeypatch, tmp_path):
    def boom(_a):
        raise RuntimeError("x")

    audio = np.ones((2048, 1), dtype="float32")
    data, out, _ = monitored(monkeypatch, tmp_path, audio, [boom])
    assert not out.any()
    assert len(data) == 2048


def test_monitor_rejects_overlong_chain():
    with pytest.raises(ValueError):
        timeline.AudioRecorderThread("x.wav", True, [lambda a: a] * 17)


def test_duplex_stream_does_not_replace_playback_reference(monkeypatch):
    class S:
        PortAudioError = Exception

        def Stream(self, **kw):
            return object()

    monkeypatch.setattr(audio_host, "_output_ref", None)
    audio_host.open_duplex_stream(S())
    assert audio_host.active_output_latency() == 0.0


def test_long_recording_streams_to_disk_and_reports_levels(
    monkeypatch, tmp_path
):
    audio = np.full((600 * 1024, 1), 0.25, dtype="float32")
    box = []
    monkeypatch.setattr(timeline, "sd", FakeSD(box, audio, 0.0))
    rec = timeline.AudioRecorderThread(str(tmp_path / "rec.wav"))
    levels = []
    rec.level_ready.connect(levels.append)
    box.append(rec)
    rec.run()
    data, _ = sf.read(str(tmp_path / "rec.wav"))
    assert len(data) == 614400
    assert rec.frames == []
    assert len(levels) >= 1
    assert levels[0] == pytest.approx(0.25, abs=1e-4)


def test_streamed_recording_trims_compensation(monkeypatch, tmp_path):
    audio = np.random.default_rng(1).random((614400, 1)).astype("float32")
    data = record(monkeypatch, tmp_path, audio, latency=0.05)
    assert len(data) == 614400 - 2205
    np.testing.assert_allclose(data, audio[2205:, 0], atol=1e-4)


def test_recording_shorter_than_latency_is_kept_whole(monkeypatch, tmp_path):
    audio = np.ones((1000, 1), dtype="float32") * 0.5
    data = record(monkeypatch, tmp_path, audio, latency=0.05)
    assert len(data) == 1000
