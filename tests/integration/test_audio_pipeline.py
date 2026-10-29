import sys
import types

import numpy as np
import pytest
import soundfile as sf

from korshmer.constants import DEFAULT_PLAYBACK_BUFFER_SIZE
from korshmer.core import audio_engine as audio_engine_module
from korshmer.core.audio_engine import AudioEngine, PlaybackState

pytestmark = pytest.mark.integration


def _fake_sounddevice_module():
    module = types.ModuleType("sounddevice")

    class PortAudioError(Exception):
        pass

    class CallbackStop(Exception):
        pass

    module.PortAudioError = PortAudioError
    module.CallbackStop = CallbackStop
    module.OutputStream = object
    module.query_devices = lambda *args, **kwargs: {}
    module.default = types.SimpleNamespace(reset=lambda: None)
    return module


try:
    pass
except OSError:
    sys.modules["sounddevice"] = _fake_sounddevice_module()


SAMPLE_RATE = 44100
DURATION_SECONDS = 0.5
CALLBACK_FRAMES = 1024


class RecordingStream:
    instances = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        RecordingStream.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def engine(mock_qt_app, monkeypatch):
    RecordingStream.instances = []
    monkeypatch.setattr(
        "korshmer.core.audio_engine.sd.OutputStream", RecordingStream
    )
    monkeypatch.setattr(
        "korshmer.core.audio_engine.sd.query_devices",
        lambda *args, **kwargs: {"default_samplerate": SAMPLE_RATE},
    )
    eng = AudioEngine()
    yield eng
    eng.timer.stop()


@pytest.fixture
def wav_path(tmp_path):
    t = np.arange(int(SAMPLE_RATE * DURATION_SECONDS)) / SAMPLE_RATE
    mono = (0.5 * np.sin(2 * np.pi * 330.0 * t)).astype("float32")
    path = tmp_path / "tone.wav"
    sf.write(str(path), mono, SAMPLE_RATE, subtype="FLOAT")
    return str(path)


def test_load_play_and_callback_to_end_of_file(engine, wav_path):
    assert engine.load(wav_path) is True
    assert engine.data.shape == (int(SAMPLE_RATE * DURATION_SECONDS), 1)
    assert engine.fs == SAMPLE_RATE
    total_frames = len(engine.data)

    engine.play()

    assert engine.playbackState() == PlaybackState.PLAYING
    assert len(RecordingStream.instances) == 1
    stream_kwargs = RecordingStream.instances[0].kwargs
    assert stream_kwargs["blocksize"] == DEFAULT_PLAYBACK_BUFFER_SIZE
    assert stream_kwargs["samplerate"] == SAMPLE_RATE
    assert stream_kwargs["channels"] == 1

    callback = stream_kwargs["callback"]
    rendered = []
    stop_raised = False
    for _ in range(total_frames // CALLBACK_FRAMES + 2):
        outdata = np.full((CALLBACK_FRAMES, 1), 9.0, dtype="float32")
        try:
            callback(outdata, CALLBACK_FRAMES, None, None)
        except audio_engine_module.sd.CallbackStop:
            stop_raised = True
            rendered.append(outdata.copy())
            break
        rendered.append(outdata.copy())

    assert stop_raised
    assert engine.idx == total_frames

    output = np.concatenate(rendered)
    played = output[:total_frames]
    np.testing.assert_allclose(played, engine.data, atol=1e-6)

    last = rendered[-1]
    remainder = total_frames - (len(rendered) - 1) * CALLBACK_FRAMES
    assert np.all(last[remainder:] == 0.0)
