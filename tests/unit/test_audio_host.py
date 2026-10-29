import pytest

from korshmer.utils import audio_host


class FakePortAudioError(Exception):
    pass


class FakeSD:
    PortAudioError = FakePortAudioError

    def __init__(self, apis, fail_device=False):
        self._apis = apis
        self.fail_device = fail_device
        self.opened = []

    def query_hostapis(self):
        return self._apis

    def OutputStream(self, **kwargs):
        if "device" in kwargs and self.fail_device:
            raise FakePortAudioError("cannot open")
        self.opened.append(kwargs)
        return kwargs

    InputStream = OutputStream


ASIO_APIS = [
    {"name": "MME", "default_output_device": 0, "default_input_device": 0},
    {"name": "ASIO", "default_output_device": 3, "default_input_device": 4},
]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_host, "AUDIO_HOST_FILE", tmp_path / "host.json")
    audio_host.reset_host_preference_cache()
    yield
    audio_host.reset_host_preference_cache()


def test_default_preference_opens_default_stream(monkeypatch):
    monkeypatch.setattr(audio_host, "asio_supported", lambda: True)
    sd = FakeSD(ASIO_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]


def test_asio_selected_opens_asio_device(monkeypatch):
    monkeypatch.setattr(audio_host, "asio_supported", lambda: True)
    audio_host.set_host_preference("asio")
    sd = FakeSD(ASIO_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert sd.opened[0]["device"] == 3
    audio_host.open_input_stream(sd, samplerate=44100)
    assert sd.opened[1]["device"] == 4


def test_asio_missing_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(audio_host, "asio_supported", lambda: True)
    audio_host.set_host_preference("asio")
    sd = FakeSD(ASIO_APIS[:1])
    with caplog.at_level("WARNING"):
        audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]
    assert "no ASIO output device" in caplog.text


def test_asio_open_failure_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(audio_host, "asio_supported", lambda: True)
    audio_host.set_host_preference("asio")
    sd = FakeSD(ASIO_APIS, fail_device=True)
    with caplog.at_level("WARNING"):
        audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]
    assert "Could not open ASIO device" in caplog.text


def test_asio_ignored_off_windows(monkeypatch):
    monkeypatch.setattr(audio_host, "asio_supported", lambda: False)
    assert audio_host.set_host_preference("asio") == "default"
    sd = FakeSD(ASIO_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]


def test_preference_persists(monkeypatch):
    monkeypatch.setattr(audio_host, "asio_supported", lambda: True)
    audio_host.set_host_preference("asio")
    audio_host.reset_host_preference_cache()
    assert audio_host.get_host_preference() == "asio"


COREAUDIO_APIS = [
    {
        "name": "Core Audio",
        "default_output_device": 2,
        "default_input_device": 1,
    },
]
JACK_APIS = [
    {"name": "ALSA", "default_output_device": 0, "default_input_device": 0},
    {
        "name": "JACK Audio Connection Kit",
        "default_output_device": 5,
        "default_input_device": 6,
    },
]


def test_coreaudio_selected_opens_coreaudio_device(monkeypatch):
    monkeypatch.setattr(audio_host.sys, "platform", "darwin")
    assert audio_host.set_host_preference("coreaudio") == "coreaudio"
    sd = FakeSD(COREAUDIO_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert sd.opened[0]["device"] == 2
    audio_host.open_input_stream(sd, samplerate=44100)
    assert sd.opened[1]["device"] == 1


def test_coreaudio_missing_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(audio_host.sys, "platform", "darwin")
    audio_host.set_host_preference("coreaudio")
    sd = FakeSD([{"name": "Other", "default_output_device": 0}])
    with caplog.at_level("WARNING"):
        audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]
    assert "no CoreAudio output device" in caplog.text


def test_coreaudio_normalized_off_macos(monkeypatch):
    monkeypatch.setattr(audio_host.sys, "platform", "linux")
    assert audio_host.set_host_preference("coreaudio") == "default"
    sd = FakeSD(COREAUDIO_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]


def test_jack_selected_opens_jack_device(monkeypatch):
    monkeypatch.setattr(audio_host.sys, "platform", "linux")
    assert audio_host.set_host_preference("jack") == "jack"
    sd = FakeSD(JACK_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert sd.opened[0]["device"] == 5
    audio_host.open_input_stream(sd, samplerate=44100)
    assert sd.opened[1]["device"] == 6


def test_jack_missing_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(audio_host.sys, "platform", "linux")
    audio_host.set_host_preference("jack")
    sd = FakeSD(JACK_APIS[:1])
    with caplog.at_level("WARNING"):
        audio_host.open_output_stream(sd, samplerate=44100)
        audio_host.open_input_stream(sd, samplerate=44100)
    assert all("device" not in o for o in sd.opened)
    assert "no JACK output device" in caplog.text
    assert "no JACK input device" in caplog.text


def test_jack_open_failure_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(audio_host.sys, "platform", "linux")
    audio_host.set_host_preference("jack")
    sd = FakeSD(JACK_APIS, fail_device=True)
    with caplog.at_level("WARNING"):
        audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]
    assert "Could not open JACK device" in caplog.text


def test_jack_normalized_off_linux(monkeypatch):
    monkeypatch.setattr(audio_host.sys, "platform", "win32")
    assert audio_host.set_host_preference("jack") == "default"
    sd = FakeSD(JACK_APIS)
    audio_host.open_output_stream(sd, samplerate=44100)
    assert "device" not in sd.opened[0]


def test_stored_unsupported_host_reads_as_default(monkeypatch):
    monkeypatch.setattr(audio_host.sys, "platform", "linux")
    audio_host.set_host_preference("jack")
    audio_host.reset_host_preference_cache()
    monkeypatch.setattr(audio_host.sys, "platform", "darwin")
    assert audio_host.get_host_preference() == "default"


def test_available_hosts_per_platform(monkeypatch):
    monkeypatch.setattr(audio_host.sys, "platform", "win32")
    assert audio_host.available_hosts() == ("default", "asio")
    monkeypatch.setattr(audio_host.sys, "platform", "darwin")
    assert audio_host.available_hosts() == ("default", "coreaudio")
    monkeypatch.setattr(audio_host.sys, "platform", "linux")
    assert audio_host.available_hosts() == ("default", "jack")


class LatencyStream:
    def __init__(self, latency, active=True):
        self.latency = latency
        self.active = active
        self.closed = False


def test_compensation_frames_sums_latencies():
    assert audio_host.compensation_frames(0.05, 0.0, 44100) == 2205
    assert audio_host.compensation_frames(0.05, 0.05, 44100) == 4410
    assert audio_host.compensation_frames(-1.0, float("nan"), 44100) == 0


def test_stream_input_latency_handles_float_pair_and_garbage():
    assert audio_host.stream_input_latency(LatencyStream(0.05)) == 0.05
    assert audio_host.stream_input_latency(LatencyStream((0.02, 0.09))) == 0.02
    assert audio_host.stream_input_latency(LatencyStream(None)) == 0.0
    assert audio_host.stream_input_latency(object()) == 0.0


def test_active_output_latency_tracks_open_playback_stream():
    class OutSD(FakeSD):
        def OutputStream(self, **kwargs):
            return LatencyStream(0.1)

    assert audio_host.active_output_latency() in (0.0, 0.1)
    sd = OutSD([])
    stream = audio_host.open_output_stream(sd, samplerate=44100)
    assert audio_host.active_output_latency() == 0.1
    stream.active = False
    assert audio_host.active_output_latency() == 0.0
    stream.active = True
    stream.closed = True
    assert audio_host.active_output_latency() == 0.0


class StartFailSD:
    PortAudioError = FakePortAudioError

    def __init__(self, fail_devices, default_fails=False):
        self.fail_devices = fail_devices
        self.default_fails = default_fails
        self.streams = []

    def query_hostapis(self):
        return [
            {"name": "JACK Audio Connection Kit", "default_output_device": 3}
        ]

    def OutputStream(self, **kwargs):
        device = kwargs.get("device")
        fails = (
            device in self.fail_devices
            if device is not None
            else self.default_fails
        )
        stream = FakeStream(device, fails)
        self.streams.append(stream)
        return stream


class FakeStream:
    def __init__(self, device, fails):
        self.device = device
        self.fails = fails
        self.started = False
        self.closed = False

    def start(self):
        if self.fails:
            raise FakePortAudioError("device busy")
        self.started = True

    def close(self):
        self.closed = True


def test_start_failure_falls_back_to_default_device(monkeypatch, caplog):
    monkeypatch.setattr(audio_host, "jack_supported", lambda: True)
    audio_host.set_host_preference("jack")
    sd = StartFailSD({3})
    with caplog.at_level("WARNING"):
        stream = audio_host.open_and_start_output_stream(sd, samplerate=44100)
    assert stream.device is None and stream.started
    assert sd.streams[0].closed
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    assert audio_host.get_host_preference() == "jack"


def test_start_failure_on_default_too_raises(monkeypatch):
    monkeypatch.setattr(audio_host, "jack_supported", lambda: True)
    audio_host.set_host_preference("jack")
    sd = StartFailSD({3}, default_fails=True)
    with pytest.raises(FakePortAudioError):
        audio_host.open_and_start_output_stream(sd, samplerate=44100)
    assert all(s.closed for s in sd.streams)


def test_value_error_on_open_falls_back(monkeypatch):
    monkeypatch.setattr(audio_host, "jack_supported", lambda: True)
    audio_host.set_host_preference("jack")
    sd = StartFailSD(set())
    real = sd.OutputStream

    def factory(**kw):
        if "device" in kw:
            raise ValueError("bad arg")
        return real(**kw)

    sd.OutputStream = factory
    stream = audio_host.open_and_start_output_stream(sd, samplerate=44100)
    assert stream.device is None and stream.started


class SpareSD:
    PortAudioError = FakePortAudioError

    def __init__(self, devices, bad, default_output=0):
        self._devices = devices
        self.bad = set(bad)
        self.tried = []
        self.streams = []

        class _Default:
            device = (default_output, default_output)

        self.default = _Default()

    def query_devices(self):
        return self._devices

    def OutputStream(self, **kwargs):
        device = kwargs.get("device")
        self.tried.append(device)
        stream = FakeStream(device, device is None or device in self.bad)
        self.streams.append(stream)
        return stream


def _devices(*channels):
    return [
        {"name": f"d{i}", "max_output_channels": c}
        for i, c in enumerate(channels)
    ]


def test_other_output_device_used_when_default_cannot_start(caplog):
    sd = SpareSD(_devices(2, 0, 2), bad=set())
    with caplog.at_level("WARNING"):
        stream = audio_host.open_and_start_output_stream(
            sd, samplerate=44100, channels=2
        )
    assert stream.device == 2 and stream.started
    assert sd.streams[0].closed
    assert any("instead" in r.message for r in caplog.records)


def test_spare_devices_skip_failed_and_too_narrow_ones():
    sd = SpareSD(_devices(2, 2, 1, 2, 2), bad={1})
    stream = audio_host.open_and_start_output_stream(
        sd, samplerate=44100, channels=2
    )
    assert stream.device == 3
    assert sd.tried == [None, 1, 3]
    assert sd.streams[1].closed


def test_all_devices_failing_raises_original_error():
    sd = SpareSD(_devices(2, 2, 2), bad={1, 2})
    with pytest.raises(FakePortAudioError):
        audio_host.open_and_start_output_stream(
            sd, samplerate=44100, channels=2
        )
    assert all(s.closed for s in sd.streams)


def test_spare_device_tries_are_capped():
    from korshmer.constants import AUDIO_FALLBACK_DEVICE_TRIES

    sd = SpareSD(_devices(*([2] * 20)), bad=set(range(20)))
    with pytest.raises(FakePortAudioError):
        audio_host.open_and_start_output_stream(
            sd, samplerate=44100, channels=2
        )
    assert len(sd.tried) == 1 + AUDIO_FALLBACK_DEVICE_TRIES


def test_explicit_device_is_not_replaced():
    sd = SpareSD(_devices(2, 2), bad={1})
    with pytest.raises(FakePortAudioError):
        audio_host.open_and_start_output_stream(
            sd, samplerate=44100, channels=2, device=1
        )
    assert sd.tried == [1]
