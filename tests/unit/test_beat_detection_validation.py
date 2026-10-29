import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.beat_detection import (
    BeatDetector,
    MetricModulationDetector,
    RubatoDetector,
)

BAD_NUMBERS = [0, -120, float("nan"), float("inf"), "x", True]


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "noise.wav"
    noise = np.random.default_rng(0).standard_normal(22050 * 3)
    sf.write(path, (noise * 0.1).astype(np.float32), 22050)
    return str(path)


@pytest.mark.parametrize("tempo", BAD_NUMBERS)
def test_rubato_rejects_invalid_expected_tempo(tempo):
    with pytest.raises(ValueError):
        RubatoDetector().detect_rubato(np.arange(0, 5, 0.5), tempo)


@pytest.mark.parametrize("window", BAD_NUMBERS)
def test_tempo_variations_rejects_invalid_window(audio_path, window):
    with pytest.raises(ValueError):
        BeatDetector().detect_tempo_variations(audio_path, window)


@pytest.mark.parametrize("tempo", BAD_NUMBERS)
def test_downbeats_rejects_invalid_tempo(audio_path, tempo):
    with pytest.raises(ValueError):
        BeatDetector().detect_downbeats(audio_path, tempo)


@pytest.mark.parametrize("bad", [0, -120, float("nan"), float("inf")])
def test_metric_modulation_skips_pairs_without_a_usable_tempo(bad):
    timeline = [
        {"time": 0.0, "tempo": bad},
        {"time": 1.0, "tempo": 120.0},
        {"time": 2.0, "tempo": 180.0},
    ]

    result = MetricModulationDetector().detect_metric_modulation("x", timeline)

    assert [m["time"] for m in result] == [2.0]
