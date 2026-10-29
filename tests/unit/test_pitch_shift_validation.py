from unittest import mock

import numpy as np
import pytest

from korshmer.audio_processing.spectral_analysis import SpectralAnalyzer
from korshmer.audio_processing.stem_classification import VocalProcessingAI
from korshmer.audio_processing.validation import (
    MAX_PITCH_SHIFT_SEMITONES,
    MIN_PITCH_SHIFT_SEMITONES,
    validate_pitch_shift_semitones,
)


class TestValidatePitchShiftSemitones:
    @pytest.mark.parametrize("value", [-24, -24.0, -12, 0, 0.5, 12, 24, 24.0])
    def test_valid_values_unchanged(self, value):
        assert validate_pitch_shift_semitones(value) == value

    def test_numpy_float_accepted(self):
        assert validate_pitch_shift_semitones(np.float64(6.0)) == 6.0

    def test_numpy_integer_accepted(self):
        assert validate_pitch_shift_semitones(np.int64(-6)) == -6

    def test_bounds_constants(self):
        assert (MIN_PITCH_SHIFT_SEMITONES, MAX_PITCH_SHIFT_SEMITONES) == (
            -24.0,
            24.0,
        )

    @pytest.mark.parametrize("value", [24.01, 25, -24.01, 500, -500])
    def test_out_of_range(self, value):
        with pytest.raises(ValueError, match="Pitch shift must be between"):
            validate_pitch_shift_semitones(value)

    def test_out_of_range_message_example(self):
        with pytest.raises(
            ValueError, match="must be between -24 and 24 semitones, got 500"
        ):
            validate_pitch_shift_semitones(500)

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), "12", None, True, False],
    )
    def test_wrong_types_and_non_finite(self, value):
        with pytest.raises(
            ValueError, match="Pitch shift must be a finite number"
        ):
            validate_pitch_shift_semitones(value)


@pytest.fixture
def audio():
    t = np.linspace(0, 0.1, 4410, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestPitchShiftFile:
    def test_semitones_out_of_range_raises_and_writes_nothing(self, tmp_path):
        from korshmer.audio_processing import core_analysis

        out = tmp_path / "out.wav"
        with (
            mock.patch.object(core_analysis, "librosa") as m_librosa,
            mock.patch.object(core_analysis, "sf") as m_sf,
        ):
            with pytest.raises(ValueError, match="Pitch shift"):
                core_analysis.pitch_shift_file("in.wav", str(out), semitones=30)
        m_librosa.load.assert_not_called()
        m_sf.write.assert_not_called()
        assert not out.exists()

    def test_n_steps_out_of_range_raises(self, tmp_path):
        from korshmer.audio_processing import core_analysis

        out = tmp_path / "out.wav"
        with (
            mock.patch.object(core_analysis, "librosa") as m_librosa,
            mock.patch.object(core_analysis, "sf"),
        ):
            with pytest.raises(ValueError, match="Pitch shift"):
                core_analysis.pitch_shift_file(
                    "in.wav", str(out), semitones=0, n_steps=30
                )
        m_librosa.load.assert_not_called()


class TestProcessAudioPitch:
    def test_out_of_range_raises_before_makedirs(self):
        from korshmer import audio_processors

        with (
            mock.patch.object(audio_processors.os, "makedirs") as m_mk,
            mock.patch.object(audio_processors, "pitch_shift_file") as m_ps,
        ):
            with pytest.raises(ValueError, match="Pitch shift"):
                audio_processors.process_audio_pitch("in.wav", 30)
        m_mk.assert_not_called()
        m_ps.assert_not_called()


class TestSpectralAnalyzerPitchShift:
    def test_pitch_shift_rejects_large_shift_without_processing(self, audio):
        sa = SpectralAnalyzer()
        sa.sr = 44100
        with mock.patch(
            "korshmer.audio_processing.spectral_analysis.librosa"
        ) as m_librosa:
            with pytest.raises(ValueError, match="Pitch shift"):
                sa.pitch_shift(audio, n_steps=120)
        m_librosa.effects.pitch_shift.assert_not_called()

    def test_formant_preserving_rejects_large_shift(self, audio):
        sa = SpectralAnalyzer()
        sa.sr = 44100
        with mock.patch(
            "korshmer.audio_processing.spectral_analysis.librosa"
        ) as m_librosa:
            with pytest.raises(ValueError, match="Pitch shift"):
                sa.formant_preserving_pitch_shift(audio, n_steps=120)
        m_librosa.stft.assert_not_called()

    def test_empty_audio_passthrough_skips_validation(self):

        sa = SpectralAnalyzer()
        sa.sr = 44100
        empty = np.zeros(0, dtype=np.float32)
        result = sa.pitch_shift(empty, n_steps=120)
        assert result.size == 0


class TestShiftFormants:
    def test_rejects_large_shift(self, audio):
        vp = VocalProcessingAI()
        with mock.patch(
            "korshmer.audio_processing.stem_classification.librosa"
        ) as m_librosa:
            with pytest.raises(ValueError, match="Pitch shift"):
                vp.shift_formants(audio, 44100, semitones=120)
            with pytest.raises(ValueError, match="Pitch shift"):
                vp.shift_formants(audio, 44100, semitones=-120)
        m_librosa.effects.pitch_shift.assert_not_called()

    def test_zero_shift_returns_audio_unchanged(self, audio):
        vp = VocalProcessingAI()
        result = vp.shift_formants(audio, 44100, semitones=0)
        assert np.array_equal(result, audio)
