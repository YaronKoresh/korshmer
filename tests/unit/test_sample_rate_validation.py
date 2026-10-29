import math

import pytest

from korshmer.audio_processing.distortion_noise import NoiseGate, NoiseReduction
from korshmer.audio_processing.dynamics import (
    Compressor,
    Limiter,
    MultiBandCompressor,
)
from korshmer.audio_processing.effects import (
    Chorus,
    Delay,
    Flanger,
    Phaser,
    Reverb,
    StereoWidener,
    Tremolo,
)
from korshmer.audio_processing.equalizers import GraphicEQ, ParametricEQ
from korshmer.audio_processing.metering import Analyzer, AudioMetering
from korshmer.audio_processing.range_eq import RangeEQ
from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
from korshmer.audio_processing.validation import validate_sample_rate
from korshmer.audio_processing.vocal_processing import VocalSyllableDetector

SAMPLE_RATE_CLASSES = [
    Reverb,
    Delay,
    Chorus,
    Flanger,
    Phaser,
    Tremolo,
    StereoWidener,
    Compressor,
    MultiBandCompressor,
    Limiter,
    ParametricEQ,
    GraphicEQ,
    AudioMetering,
    Analyzer,
    RangeEQ,
    NoiseGate,
    NoiseReduction,
    SpectralAnalyzer,
    VocalSyllableDetector,
]


class TestValidateSampleRate:
    def test_returns_positive_values_unchanged(self):
        assert validate_sample_rate(44100) == 44100
        assert validate_sample_rate(48000.0) == 48000.0

    @pytest.mark.parametrize("bad", [0, -1, -44100, 0.0, -0.5])
    def test_rejects_zero_and_negative(self, bad):
        with pytest.raises(ValueError):
            validate_sample_rate(bad)

    def test_rejects_non_numbers_bool_and_nan(self):
        for bad in ["44100", None, True, False, math.nan]:
            with pytest.raises(ValueError):
                validate_sample_rate(bad)

    def test_error_message_includes_the_value(self):
        with pytest.raises(ValueError, match="0"):
            validate_sample_rate(0)

    @pytest.mark.parametrize("value", [8000, 192000, 44100, 48000.0])
    def test_in_range_values_pass_unchanged(self, value):

        assert validate_sample_rate(value) == value

    @pytest.mark.parametrize("bad", [7999, 100, 192001, 1_000_000_000, 7999.9])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            validate_sample_rate(bad)

    def test_out_of_range_message_includes_value_and_range(self):
        with pytest.raises(ValueError, match="7999") as exc_low:
            validate_sample_rate(7999)
        assert "8000" in str(exc_low.value)
        assert "192000" in str(exc_low.value)

        with pytest.raises(ValueError, match="192001") as exc_high:
            validate_sample_rate(192001)
        assert "8000" in str(exc_high.value)
        assert "192000" in str(exc_high.value)


class TestSampleRateClasses:
    @pytest.mark.parametrize("cls", SAMPLE_RATE_CLASSES)
    def test_zero_raises(self, cls):
        with pytest.raises(ValueError):
            cls(sample_rate=0)

    @pytest.mark.parametrize("cls", SAMPLE_RATE_CLASSES)
    def test_negative_raises(self, cls):
        with pytest.raises(ValueError):
            cls(sample_rate=-44100)

    @pytest.mark.parametrize("cls", SAMPLE_RATE_CLASSES)
    def test_default_construction_still_works(self, cls):
        instance = cls()
        assert instance.sr == 44100

    def test_delay_does_not_silently_accept_zero(self):

        with pytest.raises(ValueError):
            Delay(0)

    @pytest.mark.parametrize("cls", SAMPLE_RATE_CLASSES)
    def test_out_of_range_raises(self, cls):
        with pytest.raises(ValueError):
            cls(sample_rate=4000)
        with pytest.raises(ValueError):
            cls(sample_rate=200000)

    def test_multiband_compressor_fails_at_construction_not_in_scipy(self):

        with pytest.raises(ValueError, match="8000"):
            MultiBandCompressor(sample_rate=4000)

    def test_vocal_syllable_detector_has_no_separate_range_check(self):

        with pytest.raises(ValueError):
            VocalSyllableDetector(sample_rate=4000)


class TestSmartMastering:
    @staticmethod
    def _signal():
        import numpy as np

        t = np.linspace(0, 1, 44100, endpoint=False)
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)

    def test_process_accepts_none_sr(self):
        from korshmer.audio_processing.smart_mastering import SmartMastering

        master = SmartMastering()

        out = master.process(self._signal(), sr=None)
        assert out is not None

    def test_process_rejects_zero_sr(self):
        from korshmer.audio_processing.smart_mastering import SmartMastering

        master = SmartMastering()
        with pytest.raises(ValueError):
            master.process(self._signal(), sr=0)


class TestApplyFilter:
    def test_array_branch_rejects_zero_sample_rate(self):
        import numpy as np

        from korshmer.audio_processing.basic_effects import apply_filter

        audio = np.zeros(1024, dtype=np.float64)
        with pytest.raises(ValueError):
            apply_filter(audio, 0, filter_type="lowpass", cutoff=1000)
