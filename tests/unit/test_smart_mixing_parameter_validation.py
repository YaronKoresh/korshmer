import numpy as np
import pytest

from korshmer.audio_processing.smart_mixing import SmartMixingAssistant

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]


@pytest.fixture
def stereo():
    t = np.arange(22050)
    return np.vstack([np.sin(t / 7.0), np.sin(t / 5.0)]) * 0.3


@pytest.mark.parametrize("value", NOT_NUMBERS + [1, 1e18, -100])
def test_target_loudness_rejected(value, stereo):
    with pytest.raises(ValueError, match="Target loudness"):
        SmartMixingAssistant().optimize_dynamic_range(stereo, 44100, value)


@pytest.mark.parametrize("value", NOT_NUMBERS + [-0.1, 4.1, 1e18])
def test_target_width_rejected(value, stereo):
    with pytest.raises(ValueError, match="Target width"):
        SmartMixingAssistant().optimize_stereo_width(stereo, value)


def test_defaults_and_edges_still_work(stereo):
    assistant = SmartMixingAssistant()
    assert np.all(np.isfinite(assistant.optimize_dynamic_range(stereo, 44100)))
    assert np.all(np.isfinite(assistant.optimize_stereo_width(stereo)))
    mono = assistant.optimize_stereo_width(stereo, 0.0)
    np.testing.assert_allclose(mono[0], mono[1])
    assert assistant.optimize_stereo_width(stereo[0]).shape[0] == 2
