import numpy as np
import pytest

from korshmer.audio_processing.vocal_processing import (
    VocalNote,
    VocalSyllableDetector,
)


class TestVocalNote:
    def test_vocal_note_initialization(self):
        note = VocalNote(start_time=0.5, end_time=1.5, pitch=60, velocity=100)

        assert note.start == 0.5
        assert note.end == 1.5
        assert note.pitch == 60
        assert note.velocity == 100
        assert abs(note.duration - 1.0) < 1e-6

    def test_vocal_note_with_audio_segment(self):
        audio_seg = np.array([0.1, 0.2, 0.3])
        note = VocalNote(
            start_time=0.0, end_time=1.0, pitch=60, audio_segment=audio_seg
        )

        assert note.audio_segment is not None
        np.testing.assert_array_equal(note.audio_segment, audio_seg)

    def test_vocal_note_invalid_time_range(self):
        with pytest.raises(
            ValueError, match="start_time.*must be less than end_time"
        ):
            VocalNote(start_time=1.5, end_time=1.5, pitch=60)

        with pytest.raises(
            ValueError, match="start_time.*must be less than end_time"
        ):
            VocalNote(start_time=2.0, end_time=1.0, pitch=60)

    def test_vocal_note_invalid_pitch(self):
        with pytest.raises(
            ValueError, match="MIDI pitch.*must be in range 0-127"
        ):
            VocalNote(start_time=0.0, end_time=1.0, pitch=-1)

        with pytest.raises(
            ValueError, match="MIDI pitch.*must be in range 0-127"
        ):
            VocalNote(start_time=0.0, end_time=1.0, pitch=128)

    def test_vocal_note_invalid_velocity(self):
        with pytest.raises(
            ValueError, match="Velocity.*must be in range 0-127"
        ):
            VocalNote(start_time=0.0, end_time=1.0, pitch=60, velocity=-1)

        with pytest.raises(
            ValueError, match="Velocity.*must be in range 0-127"
        ):
            VocalNote(start_time=0.0, end_time=1.0, pitch=60, velocity=128)

    def test_vocal_note_equality(self):
        note1 = VocalNote(start_time=0.5, end_time=1.5, pitch=60, velocity=100)
        note2 = VocalNote(start_time=0.5, end_time=1.5, pitch=60, velocity=100)
        note3 = VocalNote(start_time=0.5, end_time=1.5, pitch=62, velocity=100)

        assert note1 == note2
        assert note1 != note3

    def test_vocal_note_hash(self):
        note1 = VocalNote(start_time=0.5, end_time=1.5, pitch=60, velocity=100)
        note2 = VocalNote(start_time=0.5, end_time=1.5, pitch=60, velocity=100)

        assert hash(note1) == hash(note2)

        note_set = {note1, note2}
        assert len(note_set) == 1

    def test_vocal_note_repr(self):
        note = VocalNote(start_time=0.5, end_time=1.5, pitch=60, velocity=100)
        repr_str = repr(note)

        assert "VocalNote" in repr_str
        assert "start=0.500" in repr_str
        assert "end=1.500" in repr_str
        assert "pitch=60" in repr_str


class TestVocalSyllableDetector:
    def test_detector_initialization_default(self):
        detector = VocalSyllableDetector()

        assert detector.sr == 44100
        assert detector.voice_type == "soprano"
        assert detector.fmin == 260
        assert detector.fmax == 2000

    def test_detector_initialization_bass(self):
        detector = VocalSyllableDetector(voice_type="bass")

        assert detector.voice_type == "bass"
        assert detector.fmin == 80
        assert detector.fmax == 350

    def test_detector_initialization_tenor(self):
        detector = VocalSyllableDetector(voice_type="tenor")

        assert detector.voice_type == "tenor"
        assert detector.fmin == 130
        assert detector.fmax == 520

    def test_detector_initialization_alto(self):
        detector = VocalSyllableDetector(voice_type="alto")

        assert detector.voice_type == "alto"
        assert detector.fmin == 175
        assert detector.fmax == 700

    def test_detector_custom_sample_rate(self):
        detector = VocalSyllableDetector(sample_rate=48000)

        assert detector.sr == 48000

    def test_remove_dc_offset(self):
        detector = VocalSyllableDetector()
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = detector._remove_dc_offset(audio)

        assert abs(np.mean(result)) < 1e-10

    def test_noise_gate_invalid_threshold(self):
        detector = VocalSyllableDetector()
        audio = np.random.randn(1000)

        with pytest.raises(ValueError, match="threshold_db must be negative"):
            detector._noise_gate(audio, threshold_db=0)

        with pytest.raises(ValueError, match="threshold_db must be negative"):
            detector._noise_gate(audio, threshold_db=10)

    def test_noise_gate_silent_audio(self):
        detector = VocalSyllableDetector()
        audio = np.zeros(1000)

        result = detector._noise_gate(audio, threshold_db=-40)

        np.testing.assert_array_equal(result, audio)

    def test_noise_gate_empty_audio(self):
        detector = VocalSyllableDetector()
        audio = np.array([])

        result = detector._noise_gate(audio, threshold_db=-40)

        assert len(result) == 0

    def test_detect_syllables_empty_audio(self):
        detector = VocalSyllableDetector()
        audio = np.array([])

        syllables = detector.detect_syllables(audio)

        assert syllables == []

    def test_detect_syllables_silent_audio(self):
        detector = VocalSyllableDetector()
        audio = np.zeros(44100)

        syllables = detector.detect_syllables(audio)

        assert len(syllables) == 0

    def test_detect_syllables_and_notes_on_seeded_noise(self):
        detector = VocalSyllableDetector()
        rng = np.random.default_rng(0)
        audio = 0.3 * rng.standard_normal(2 * 44100)

        syllables = detector.detect_syllables(audio)
        notes = detector.extract_vocal_notes(audio, syllables)

        assert len(notes) <= len(syllables)
        for start, end in syllables:
            assert 0.0 <= start < end <= 2.0 + 1e-6
        for note in notes:
            assert 0 <= note.pitch <= 127

    def test_detect_syllables_polyphonic_chord_respects_duration_limits(self):
        detector = VocalSyllableDetector()
        t = np.arange(2 * 44100) / 44100
        audio = sum(
            0.2 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63)
        )
        min_dur, max_dur = 0.05, 5.0

        syllables = detector.detect_syllables(
            audio, min_syllable_duration=min_dur, max_syllable_duration=max_dur
        )
        notes = detector.extract_vocal_notes(audio, syllables)

        for start, end in syllables:
            assert min_dur - 1e-6 <= end - start <= max_dur + 1e-6
        for note in notes:
            assert 0 <= note.pitch <= 127

    def test_detect_syllables_with_progress_callback(self):
        detector = VocalSyllableDetector()

        t = np.linspace(0, 1, 44100)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        progress_values = []

        def callback(progress):
            progress_values.append(progress)

        detector.detect_syllables(audio, progress_callback=callback)

        assert len(progress_values) > 0

        assert progress_values[-1] == 1.0

    def test_detect_syllables_min_duration_filter(self):
        detector = VocalSyllableDetector()

        t = np.linspace(0, 1, 44100)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        syllables = detector.detect_syllables(audio, min_syllable_duration=0.1)

        for start, end in syllables:
            assert end - start >= 0.1

    @pytest.mark.slow
    def test_autocorrelation_pitch_empty_audio(self):
        detector = VocalSyllableDetector()
        audio_segment = np.array([])

        pitch, confidence = detector._autocorrelation_pitch(audio_segment)

        assert pitch == 0
        assert confidence == 0.0

    @pytest.mark.slow
    def test_autocorrelation_pitch_silent_audio(self):
        detector = VocalSyllableDetector()
        audio_segment = np.zeros(1000)

        pitch, confidence = detector._autocorrelation_pitch(audio_segment)

        assert pitch == 0
        assert confidence == 0.0

    def test_voice_ranges_defined(self):
        expected_ranges = ["bass", "tenor", "alto", "soprano"]

        for voice_type in expected_ranges:
            assert voice_type in VocalSyllableDetector.VOICE_RANGES
            fmin, fmax = VocalSyllableDetector.VOICE_RANGES[voice_type]
            assert fmin < fmax
            assert fmin > 0
            assert fmax > 0


def test_syllable_detector_extraction():
    from korshmer.audio_processing.syllable_detection import SyllableDetector
    from korshmer.audio_processing.vocal_processing import (
        VocalSyllableDetector,
    )

    assert issubclass(VocalSyllableDetector, SyllableDetector)
    detector = VocalSyllableDetector(sample_rate=44100, voice_type="alto")

    for name in (
        "_remove_dc_offset",
        "_noise_gate",
        "_find_onset_times",
        "detect_syllables",
    ):
        assert hasattr(detector, name)
    assert detector.fmin == 175
    assert detector.fmax == 700
