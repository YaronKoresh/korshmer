import numpy as np
import pytest

from korshmer.audio_processing.validation import validate_audio_params


@validate_audio_params
def process(audio, gain=1.0):
    return audio * gain


class Proc:
    @validate_audio_params(max_channels=2)
    def run(self, audio):
        return audio


def test_valid_mono_and_stereo_run_unchanged():
    assert process(np.ones(4)).shape == (4,)
    assert process(np.ones((2, 4))).shape == (2, 4)


def test_nine_channels_rejected():
    with pytest.raises(ValueError, match="9 channels"):
        process(np.ones((9, 10)))


def test_eight_channels_accepted():
    assert process(np.ones((8, 10))).shape == (8, 10)


def test_channels_last_option():
    @validate_audio_params(channels_first=False)
    def f(audio):
        return audio

    with pytest.raises(ValueError, match="9 channels"):
        f(np.ones((10, 9)))
    assert f(np.ones((10, 2))).shape == (10, 2)


@pytest.mark.parametrize("empty", [np.array([]), np.zeros((2, 0))])
def test_empty_raises(empty):
    with pytest.raises(ValueError, match="empty"):
        process(empty)


def test_none_raises():
    with pytest.raises(ValueError, match="empty"):
        process(None)


def test_keyword_audio_argument_checked():
    with pytest.raises(ValueError):
        process(audio=np.ones((9, 3)))


def test_passthrough_returns_empty_copy():
    @validate_audio_params(empty="passthrough")
    def f(audio):
        return "ran"

    src = np.zeros((2, 0))
    out = f(src)
    assert out.shape == (2, 0) and out is not src
    with pytest.raises(ValueError):
        f(None)


def test_method_and_custom_max_channels():
    p = Proc()
    assert p.run(np.ones((2, 3))).shape == (2, 3)
    with pytest.raises(ValueError, match="3 channels"):
        p.run(np.ones((3, 3)))


def test_invalid_empty_option_and_metadata():
    with pytest.raises(ValueError):
        validate_audio_params(empty="bad")
    assert process.__name__ == "process"
