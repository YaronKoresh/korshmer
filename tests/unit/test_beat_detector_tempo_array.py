import json

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.beat_detection import (
    BeatDetector,
    RubatoDetector,
)
from korshmer.audio_processing.stem_classification import VocalProcessingAI
from korshmer.utils.tempo import scalar_tempo
from tests.fixtures.drum_loops import SAMPLE_RATE as LOOP_RATE, make_drum_loop

SAMPLE_RATE = 22050


@pytest.fixture
def noise_wav(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "noise.wav"
    audio = (rng.standard_normal(SAMPLE_RATE * 12) * 0.1).astype("float32")
    sf.write(path, audio, SAMPLE_RATE)
    return str(path)


@pytest.mark.parametrize("multiple", [True, False])
def test_detect_tempo_returns_float_tempo(noise_wav, multiple):
    tempo, beat_times = BeatDetector().detect_tempo(noise_wav, multiple)
    assert isinstance(tempo, float)
    assert len(beat_times) >= 0


def test_detect_tempo_variations_report_float_tempos(noise_wav):
    timeline = BeatDetector().detect_tempo_variations(
        noise_wav, window_size=3.0
    )
    assert all(isinstance(entry["tempo"], float) for entry in timeline)


def test_detect_emotion_returns_float_tempo():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SAMPLE_RATE * 5) * 0.1).astype("float32")
    result = VocalProcessingAI().detect_emotion(audio, SAMPLE_RATE)
    assert isinstance(result["tempo"], float)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (120.5, 120.5),
        (np.float64(90.0), 90.0),
        (np.array([128.0]), 128.0),
        (np.array([]), 0.0),
    ],
)
def test_scalar_tempo_accepts_scalars_and_arrays(value, expected):
    result = scalar_tempo(value)
    assert isinstance(result, float)
    assert result == expected


@pytest.mark.parametrize("bpm", [90, 123.4, 140])
def test_onset_strength_tempo_is_finite_and_in_range(bpm):
    audio, _ = make_drum_loop(bpm, bars=12)
    detector = BeatDetector()
    tempo = detector._onset_strength_tempo(audio, LOOP_RATE)
    assert np.isfinite(tempo)
    assert detector.tempo_range[0] <= tempo <= detector.tempo_range[1]


def test_onset_strength_tempo_of_noise_is_in_range():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SAMPLE_RATE * 10) * 0.1).astype("float32")
    detector = BeatDetector()
    tempo = detector._onset_strength_tempo(audio, SAMPLE_RATE)
    assert detector.tempo_range[0] <= tempo <= detector.tempo_range[1]


def test_rubato_result_is_json_serializable():
    beat_times = np.arange(0.5, 14.0, 0.5)
    result = RubatoDetector().detect_rubato(beat_times, 120.0)
    assert isinstance(result["has_rubato"], bool)
    json.dumps(result)


def test_swing_result_is_json_serializable(tmp_path):
    audio, _ = make_drum_loop(120, bars=8)
    path = tmp_path / "loop.wav"
    sf.write(path, audio, LOOP_RATE)
    detector = BeatDetector()
    _, beat_times = detector.detect_tempo(str(path))
    result = detector.detect_swing_shuffle(str(path), beat_times)
    assert isinstance(result["has_swing"], bool)
    json.dumps(result)
