import numpy as np
import pytest
import soundfile as sf

from korshmer.audio import loader
from korshmer.audio.loader import StreamedAudio, open_streamed_audio
from korshmer.core import audio_engine as audio_engine_module
from korshmer.core.audio_engine import AudioEngine, PlaybackState
from korshmer.exceptions import AudioFileTooLargeError

SAMPLE_RATE = 44100
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


def _write_stereo(path, frames=SAMPLE_RATE * 2):
    t = np.arange(frames) / SAMPLE_RATE
    data = (
        np.stack(
            [np.sin(2 * np.pi * 220 * t), np.cos(2 * np.pi * 330 * t)], axis=1
        ).astype("float32")
        * 0.5
    )
    sf.write(str(path), data, SAMPLE_RATE, subtype="FLOAT")
    return data


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


def _render_to_end(engine):
    stream_kwargs = RecordingStream.instances[-1].kwargs
    callback = stream_kwargs["callback"]
    total = len(engine.data)
    rendered = []
    for _ in range(total // CALLBACK_FRAMES + 2):
        outdata = np.full(
            (CALLBACK_FRAMES, stream_kwargs["channels"]), 9.0, dtype="float32"
        )
        try:
            callback(outdata, CALLBACK_FRAMES, None, None)
        except audio_engine_module.sd.CallbackStop:
            rendered.append(outdata.copy())
            break
        rendered.append(outdata.copy())
    return np.concatenate(rendered)[:total]


def test_windows_match_full_decode(tmp_path):
    path = tmp_path / "stereo.wav"
    expected = _write_stereo(path)
    stream = StreamedAudio(path)
    try:
        assert stream.shape == expected.shape
        assert len(stream) == len(expected)
        assert stream.samplerate == SAMPLE_RATE
        for start, stop in [
            (0, 1000),
            (5000, 9000),
            (len(expected) - 10, len(expected) + 50),
        ]:
            window = stream[start:stop]
            assert window.dtype == np.float32
            np.testing.assert_allclose(window, expected[start:stop], atol=1e-6)
        assert stream[len(expected) :].shape == (0, 2)
    finally:
        stream.close()


def test_closed_stream_returns_silence(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_stereo(path)
    stream = StreamedAudio(path)
    stream.close()
    window = stream[0:512]
    assert window.shape == (512, 2)
    assert np.all(window == 0.0)


def test_small_file_is_not_streamed(tmp_path):
    path = tmp_path / "small.wav"
    _write_stereo(path, frames=1000)
    assert open_streamed_audio(path) is None


def test_file_above_threshold_is_streamed(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    _write_stereo(path)
    monkeypatch.setattr(loader, "STREAMING_AUDIO_THRESHOLD_BYTES", 0)
    stream = open_streamed_audio(path)
    try:
        assert isinstance(stream, StreamedAudio)
        assert stream.channels == 2
    finally:
        stream.close()


def test_streamed_file_with_too_many_channels_is_refused(tmp_path):
    path = tmp_path / "nine.wav"
    sf.write(
        str(path),
        np.zeros((1000, 9), dtype="float32"),
        SAMPLE_RATE,
        subtype="FLOAT",
    )
    with pytest.raises(AudioFileTooLargeError):
        open_streamed_audio(path, min_bytes=0)


def test_engine_plays_streamed_file_like_full_decode(
    engine, tmp_path, monkeypatch
):
    path = tmp_path / "stereo.wav"
    expected = _write_stereo(path)

    assert engine.load(str(path)) is True
    assert not isinstance(engine.data, StreamedAudio)
    engine.play()
    full_output = _render_to_end(engine)
    engine.stop()
    np.testing.assert_allclose(full_output, expected, atol=1e-6)

    monkeypatch.setattr(loader, "STREAMING_AUDIO_THRESHOLD_BYTES", 0)
    assert engine.load(str(path)) is True
    assert isinstance(engine.data, StreamedAudio)
    assert engine.duration() == int(len(expected) / SAMPLE_RATE * 1000)
    engine.play()
    assert engine.playbackState() == PlaybackState.PLAYING
    streamed_output = _render_to_end(engine)
    np.testing.assert_allclose(streamed_output, full_output, atol=1e-6)
    assert engine.idx == len(expected)


def test_loading_another_file_closes_previous_stream(
    engine, tmp_path, monkeypatch
):
    big = tmp_path / "big.wav"
    small = tmp_path / "small.wav"
    _write_stereo(big)
    _write_stereo(small, frames=1000)

    monkeypatch.setattr(loader, "STREAMING_AUDIO_THRESHOLD_BYTES", 0)
    assert engine.load(str(big)) is True
    previous = engine.data
    monkeypatch.setattr(
        loader, "STREAMING_AUDIO_THRESHOLD_BYTES", 100 * 1024 * 1024
    )
    assert engine.load(str(small)) is True
    assert not isinstance(engine.data, StreamedAudio)
    assert previous._handle is None
