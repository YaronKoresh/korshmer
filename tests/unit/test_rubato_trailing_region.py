from __future__ import annotations

import numpy as np

from korshmer.audio_processing.beat_detection import RubatoDetector


def test_trailing_rubato_region_is_captured():
    detector = RubatoDetector()
    detector.tolerance = 0.1

    expected_tempo = 120.0

    beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.8, 3.7])

    result = detector.detect_rubato(beat_times, expected_tempo)

    assert result["has_rubato"]
    assert len(result["rubato_regions"]) > 0
    last_region = result["rubato_regions"][-1]
    assert last_region["end_time"] == beat_times[-1]


def test_no_rubato_when_all_beats_on_grid():
    detector = RubatoDetector()
    detector.tolerance = 0.1

    beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    result = detector.detect_rubato(beat_times, 120.0)
    assert not result["has_rubato"]
    assert len(result["rubato_regions"]) == 0


def test_rubato_region_in_middle_still_works():
    detector = RubatoDetector()
    detector.tolerance = 0.1

    beat_times = np.array([0.0, 0.5, 1.0, 1.8, 2.7, 3.2, 3.7])

    result = detector.detect_rubato(beat_times, 120.0)

    assert result["has_rubato"]
    assert len(result["rubato_regions"]) >= 1
