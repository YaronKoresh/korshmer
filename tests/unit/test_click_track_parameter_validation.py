import numpy as np
import pytest

from korshmer.audio_processing.beat_detection import BeatDetector

NOT_NUMBERS = [float("nan"), float("inf"), -float("inf"), "x", None, True]


@pytest.mark.parametrize("value", NOT_NUMBERS + [0, -5, 19.9, 300.1, 1e9])
def test_tempo_rejected(value):
    with pytest.raises(ValueError, match="Tempo"):
        BeatDetector().generate_click_track(value, 3.0, 22050)


@pytest.mark.parametrize("value", NOT_NUMBERS + [-1, 3601, 1e12])
def test_duration_rejected(value):
    with pytest.raises(ValueError, match="Duration"):
        BeatDetector().generate_click_track(120, value, 22050)


@pytest.mark.parametrize("value", [0, -1, float("nan"), "x", None, 1.5, 1e9])
def test_sample_rate_rejected(value):
    with pytest.raises(ValueError, match="Sample rate"):
        BeatDetector().generate_click_track(120, 3.0, value)


def test_edges_and_defaults_still_work():
    detector = BeatDetector()
    assert detector.generate_click_track(120, 0.0, 22050).shape == (0,)
    for tempo in (20, 300):
        track = detector.generate_click_track(tempo, 2.0, 22050)
        assert track.shape == (44100,)
        assert np.all(np.isfinite(track))
        assert np.max(np.abs(track)) > 0.5
    assert detector.generate_click_track(120, 1.0).shape == (44100,)
