import numpy as np
import pytest
import soundfile as sf

from korshmer.audio import effects as audio_effects
from korshmer.audio.loader import check_audio_file_channels
from korshmer.audio_file import AudioFile
from korshmer.audio_processing import basic_effects
from korshmer.audio_processing.smart_mixing import SmartMixingAssistant
from korshmer.audio_processing.stem_session import _read_audio_any
from korshmer.audio_processing.validation import (
    audio_channel_count,
    validate_channel_count,
)
from korshmer.constants import MAX_AUDIO_CHANNELS
from korshmer.exceptions import AudioFileTooLargeError

SR = 22050


def _write_wav(tmp_path, channels):
    path = tmp_path / f"ch{channels}.wav"
    rng = np.random.default_rng(channels)
    data = (rng.standard_normal((SR // 2, channels)) * 0.1).astype("float32")
    if channels == 1:
        data = data[:, 0]
    sf.write(str(path), data, SR)
    return str(path)


@pytest.fixture(params=[1, 2, 8])
def accepted_wav(request, tmp_path):
    return _write_wav(tmp_path, request.param), request.param


@pytest.fixture
def nine_channel_wav(tmp_path):
    return _write_wav(tmp_path, 9)


def _effect_calls(module, path, out):
    return [
        lambda: module.apply_filter(path, out, filter_type="low", cutoff=1000),
        lambda: module.apply_normalize(path, out),
        lambda: module.apply_reverse_audio(path, out),
        lambda: module.apply_noise_gate(path, out),
    ]


def _entry_points(path, out):
    return (
        _effect_calls(basic_effects, path, out)
        + _effect_calls(audio_effects, path, out)
        + [
            lambda: SmartMixingAssistant().analyze_reference_track(path),
            lambda: _read_audio_any(path),
            lambda: AudioFile.load(path),
        ]
    )


class TestValidateChannelCount:
    @pytest.mark.parametrize("value", [1, 2, 6, MAX_AUDIO_CHANNELS])
    def test_valid(self, value):
        assert validate_channel_count(value) == value

    def test_numpy_integer_accepted(self):
        assert validate_channel_count(np.int64(2)) == 2

    @pytest.mark.parametrize("value", [0, -1, 9, 64, True, 2.5, "2", None])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            validate_channel_count(value)

    def test_message_names_count_and_range(self):
        with pytest.raises(ValueError, match=r"9 channels.*1 to 8"):
            validate_channel_count(9)


class TestAudioChannelCount:
    def test_one_dimensional_is_mono(self):
        assert audio_channel_count(np.zeros(10)) == 1

    def test_channels_first(self):
        assert audio_channel_count(np.zeros((2, 10))) == 2

    def test_channels_last(self):
        assert audio_channel_count(np.zeros((10, 2)), channels_first=False) == 2

    def test_three_dimensional_rejected(self):
        with pytest.raises(ValueError):
            audio_channel_count(np.zeros((2, 2, 2)))


class TestCheckAudioFileChannels:
    def test_accepted(self, accepted_wav):
        path, channels = accepted_wav
        assert check_audio_file_channels(path) == channels

    def test_nine_channels_refused(self, nine_channel_wav):
        with pytest.raises(AudioFileTooLargeError, match=r"9 channels.*1 to 8"):
            check_audio_file_channels(nine_channel_wav)

    def test_unreadable_header_skipped(self, tmp_path):
        path = tmp_path / "not_audio.wav"
        path.write_bytes(b"not audio at all")
        assert check_audio_file_channels(str(path)) is None


class TestEntryPoints:
    def test_accepted_files_still_load(self, accepted_wav, tmp_path):
        path, _ = accepted_wav
        out = str(tmp_path / "out.wav")
        for call in _entry_points(path, out):
            call()

    def test_nine_channel_file_refused_everywhere(
        self, nine_channel_wav, tmp_path, monkeypatch
    ):
        def fail_load(*args, **kwargs):
            raise AssertionError("audio decoded before channel check")

        monkeypatch.setattr("librosa.load", fail_load)
        monkeypatch.setattr(
            "korshmer.audio_processing.stem_session.sf.read", fail_load
        )
        out = str(tmp_path / "out.wav")
        for call in _entry_points(nine_channel_wav, out):
            with pytest.raises(
                AudioFileTooLargeError, match=r"9 channels.*1 to 8"
            ):
                call()
