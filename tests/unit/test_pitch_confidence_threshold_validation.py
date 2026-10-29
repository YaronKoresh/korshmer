import numpy as np
import pytest

from korshmer.audio_processing.validation import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    validate_confidence_threshold,
)
from korshmer.audio_processing.vocal_processing import (
    VocalSyllableDetector,
    detect_vocal_syllables,
)

SR = 44100


def _sine(freq=440.0, seconds=0.5, amp=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class TestValidateConfidenceThreshold:
    @pytest.mark.parametrize("value", [0.0, 0.3, 1.0, 1, 0])
    def test_accepts_range_inclusive(self, value):
        assert validate_confidence_threshold(value) == value

    @pytest.mark.parametrize(
        "value", [-0.1, 1.5, float("nan"), float("inf"), float("-inf")]
    )
    def test_rejects_out_of_range_and_non_finite(self, value):
        with pytest.raises(ValueError):
            validate_confidence_threshold(value)

    @pytest.mark.parametrize("value", [True, None, "0.5", [0.5]])
    def test_rejects_non_numbers(self, value):
        with pytest.raises(ValueError):
            validate_confidence_threshold(value)

    def test_default_is_previous_cutoff(self):
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.3


class TestExtractVocalNotesThreshold:
    @pytest.mark.parametrize("value", [1.5, -0.1])
    def test_invalid_threshold_raises(self, value):
        detector = VocalSyllableDetector(sample_rate=SR)
        with pytest.raises(ValueError):
            detector.extract_vocal_notes(
                _sine(), [(0.0, 0.5)], confidence_threshold=value
            )

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_boundary_thresholds_accepted(self, value):
        detector = VocalSyllableDetector(sample_rate=SR)
        notes = detector.extract_vocal_notes(
            _sine(), [(0.0, 0.5)], confidence_threshold=value
        )
        assert len(notes) == 1

    def test_threshold_gates_autocorrelation_fallback(self):
        detector = VocalSyllableDetector(sample_rate=SR)
        segment = _sine()
        pitch, _ = detector._autocorrelation_pitch(segment)
        assert pitch > 0
        assert detector._autocorrelation_pitch(segment, 1.0) == (0, 0.0)


class TestDetectVocalSyllablesThreshold:
    @pytest.mark.parametrize("value", [1.5, -0.1])
    def test_invalid_threshold_raises_before_loading(self, value, tmp_path):
        missing = tmp_path / "does_not_exist.wav"
        with pytest.raises(ValueError):
            detect_vocal_syllables(str(missing), confidence_threshold=value)
