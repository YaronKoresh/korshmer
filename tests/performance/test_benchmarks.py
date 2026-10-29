import time

import numpy as np
import pytest


@pytest.mark.performance
class TestAudioProcessingPerformance:
    @pytest.mark.slow
    def test_range_eq_performance(self):
        from korshmer.audio_processing.range_eq import RangeEQ

        sample_rate = 44100
        duration = 10.0
        audio = np.random.randn(int(sample_rate * duration))

        eq = RangeEQ(sample_rate=sample_rate)

        start = time.time()
        result = eq.process(
            audio, freq_start=200, freq_end=4000, total_gain_db=6.0
        )
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Processing took {elapsed:.3f}s, expected < 1.0s"
        assert result.shape == audio.shape

    @pytest.mark.slow
    def test_vocal_syllable_detection_performance(self):
        from korshmer.audio_processing.vocal_processing import (
            VocalSyllableDetector,
        )

        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        detector = VocalSyllableDetector(sample_rate=sample_rate)

        start = time.time()
        syllables = detector.detect_syllables(audio)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Detection took {elapsed:.3f}s, expected < 5.0s"
        assert isinstance(syllables, list)

    def test_instrument_filter_performance(self):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        stems = {f"instrument_{i}": np.random.randn(44100) for i in range(100)}

        start = time.time()
        InstrumentFilter.filter_stems(stems)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Filtering took {elapsed:.3f}s, expected < 0.1s"

    def test_normalization_cache_performance(self):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        InstrumentFilter._normalized_cache.clear()

        instrument_name = "Electric_Grand_Piano"

        result1 = InstrumentFilter._normalize_name(instrument_name)

        result2 = InstrumentFilter._normalize_name(instrument_name)

        assert result1 == result2

        assert instrument_name in InstrumentFilter._normalized_cache
        assert InstrumentFilter._normalized_cache[instrument_name] == result1


@pytest.mark.performance
class TestMemoryUsage:
    def test_audio_buffer_allocation(self):

        sample_rate = 44100
        duration = 60.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        size_mb = audio.nbytes / (1024 * 1024)
        assert size_mb < 15, (
            f"Audio buffer uses {size_mb:.2f} MB, expected < 15 MB"
        )

    def test_stereo_audio_allocation(self):

        sample_rate = 44100
        duration = 60.0
        audio = np.random.randn(2, int(sample_rate * duration)).astype(
            np.float32
        )

        size_mb = audio.nbytes / (1024 * 1024)
        assert size_mb < 30, (
            f"Stereo audio uses {size_mb:.2f} MB, expected < 30 MB"
        )


@pytest.mark.performance
class TestScalability:
    @pytest.mark.slow
    def test_large_midi_note_count(self):
        import sys

        from PySide6.QtWidgets import QApplication

        from korshmer.ui.widgets.midi.piano_roll import (
            InteractivePianoRoll,
            PianoKeyItem,
        )

        if not QApplication.instance():
            QApplication(sys.argv)

        piano_roll = InteractivePianoRoll()

        start = time.time()
        for i in range(1000):
            item = PianoKeyItem(
                pitch=60 + (i % 12),
                x_pos=i * 50,
                width=40,
                velocity=80,
                grid_x=50,
                grid_y=20,
            )
            piano_roll.scene.addItem(item)
        elapsed = time.time() - start

        assert elapsed < 2.0, (
            f"Adding 1000 notes took {elapsed:.3f}s, expected < 2.0s"
        )
        assert piano_roll.scene.items().__len__() >= 1000

    def test_many_instrument_checks(self):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        instruments = [f"instrument_{i}" for i in range(1000)]

        start = time.time()
        results = [
            InstrumentFilter.should_convert(inst) for inst in instruments
        ]
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Checking 1000 instruments took {elapsed:.3f}s"
        assert len(results) == 1000


@pytest.mark.performance
class TestBenchmarks:
    def test_numpy_operations_baseline(self):

        data = np.random.randn(1000000)

        start = time.time()
        np.fft.fft(data[:44100])
        elapsed = time.time() - start

        assert elapsed < 0.1, f"FFT took {elapsed:.3f}s, expected < 0.1s"

    def test_dict_lookups_baseline(self):
        test_dict = {f"key_{i}": i for i in range(10000)}

        start = time.time()
        for i in range(10000):
            _ = test_dict.get(f"key_{i}")
        elapsed = time.time() - start

        assert elapsed < 0.1, f"10000 dict lookups took {elapsed:.3f}s"
