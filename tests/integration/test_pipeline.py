import numpy as np
import pytest


@pytest.mark.integration
class TestAudioProcessingPipeline:
    def test_import_audio_processing_modules(self):
        from korshmer.audio_processing import range_eq, vocal_processing

        assert range_eq is not None
        assert vocal_processing is not None

    def test_range_eq_basic_pipeline(self):
        from korshmer.audio_processing.range_eq import RangeEQ

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        eq = RangeEQ(sample_rate=sample_rate)
        result = eq.process(
            audio, freq_start=300, freq_end=600, total_gain_db=6.0
        )

        assert result.shape == audio.shape
        assert np.isfinite(result).all()
        assert not np.allclose(result, audio)

    def test_vocal_processing_pipeline(self):
        from korshmer.audio_processing.vocal_processing import (
            VocalSyllableDetector,
        )

        detector = VocalSyllableDetector(
            sample_rate=44100, voice_type="soprano"
        )

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        syllables = detector.detect_syllables(audio, min_syllable_duration=0.05)

        assert isinstance(syllables, list)
        for start, end in syllables:
            assert start < end
            assert start >= 0
            assert end <= duration


@pytest.mark.integration
class TestMIDIProcessingPipeline:
    def test_import_midi_processing_modules(self):
        from korshmer.midi_processing import (
            drum_map,
            instrument_filter,
            scale_analysis,
        )

        assert instrument_filter is not None
        assert drum_map is not None
        assert scale_analysis is not None

    def test_instrument_filter_pipeline(self):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        stems = {
            "piano": np.random.randn(44100),
            "vocals": np.random.randn(44100),
            "guitar": np.random.randn(44100),
            "speech": np.random.randn(44100),
        }

        filtered = InstrumentFilter.filter_stems(stems)

        assert "piano" in filtered
        assert "guitar" in filtered
        assert "vocals" not in filtered
        assert "speech" not in filtered


@pytest.mark.integration
class TestUIComponents:
    def test_import_ui_modules(self):
        from korshmer.ui.widgets import material
        from korshmer.ui.widgets.midi import piano_roll

        assert piano_roll is not None
        assert material is not None

    def test_piano_roll_initialization(self):
        import sys

        from PySide6.QtWidgets import QApplication

        from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

        if not QApplication.instance():
            QApplication(sys.argv)

        piano_roll = InteractivePianoRoll()

        assert piano_roll.tempo == 120.0
        assert piano_roll.divisions_per_octave == 12
        assert piano_roll.grid_size == 20


@pytest.mark.integration
class TestProjectManagement:
    def test_import_project_modules(self):
        from korshmer.project_management import (
            analytics,
            asset_manager,
            project_manager,
        )

        assert project_manager is not None
        assert asset_manager is not None
        assert analytics is not None


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    def test_audio_to_midi_workflow_imports(self):

        from korshmer.audio_processing import vocal_processing
        from korshmer.midi_processing import instrument_filter

        assert hasattr(vocal_processing, "VocalSyllableDetector")
        assert hasattr(instrument_filter, "InstrumentFilter")

    def test_constants_integration(self):
        import sys

        from PySide6.QtWidgets import QApplication

        from korshmer.constants import MICROTONAL_DIVISIONS, NOTE_NAMES, SCALES
        from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

        if not QApplication.instance():
            QApplication(sys.argv)

        assert len(NOTE_NAMES) == 12
        assert "Major" in SCALES
        assert "Semitones (12-TET)" in MICROTONAL_DIVISIONS

        piano_roll = InteractivePianoRoll()
        piano_roll.set_scale_view("C", "Major")
        assert piano_roll.scale_intervals == SCALES["Major"]
