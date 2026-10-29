import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.genre_mood_detection import GenreMoodDetector

SAMPLE_RATE = 22050


@pytest.fixture
def noise_wav(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "noise.wav"
    sf.write(
        path,
        (rng.standard_normal(SAMPLE_RATE * 3) * 0.1).astype("float32"),
        SAMPLE_RATE,
    )
    return str(path)


@pytest.mark.parametrize(
    "method",
    ["detect_genre", "detect_mood", "calculate_danceability", "detect_era"],
)
def test_detector_returns_result_for_normal_audio(noise_wav, method):
    result = getattr(GenreMoodDetector(), method)(noise_wav)
    assert isinstance(result, dict)


def test_reported_tempo_is_plain_float(noise_wav):
    detector = GenreMoodDetector()
    assert isinstance(detector.detect_mood(noise_wav)["tempo"], float)
    assert isinstance(
        detector.calculate_danceability(noise_wav)["tempo"], float
    )
    assert isinstance(
        detector.detect_genre(noise_wav)["features"]["tempo"], float
    )
