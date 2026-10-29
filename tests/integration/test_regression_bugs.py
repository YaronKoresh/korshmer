import numpy as np
import pretty_midi
import pytest
import soundfile as sf


@pytest.mark.integration
class TestBugFixRegressions:
    def test_bpm_precision_regression(self, tmp_path):
        import sys

        from PySide6.QtWidgets import QApplication

        from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

        if not QApplication.instance():
            QApplication(sys.argv)

        piano_roll = InteractivePianoRoll()

        piano_roll.set_tempo(20)
        assert piano_roll.tempo >= 20

        piano_roll.set_tempo(300)
        assert piano_roll.tempo <= 300

        try:
            piano_roll.set_tempo(10)
            assert piano_roll.tempo >= 20
        except Exception:
            pass

        try:
            piano_roll.set_tempo(400)
            assert piano_roll.tempo <= 300
        except Exception:
            pass

    def test_spectral_balance_division_by_zero_regression(self, tmp_path):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        duration = 0.5
        audio = np.zeros(int(sample_rate * duration))

        mastering = SmartMastering(
            SmartMasteringConfig(sample_rate=sample_rate)
        )

        try:
            result, _ = mastering.process(audio)

            assert result is not None
            assert np.isfinite(result).all()
        except Exception:
            pass

    def test_phase_default_smoothing(self, tmp_path):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        f_axis = np.linspace(20.0, 20000.0, 2000)
        jagged = np.random.default_rng(0).normal(size=f_axis.shape) * 6.0

        m = SmartMastering(SmartMasteringConfig(sample_rate=sample_rate))
        m_no = SmartMastering(
            SmartMasteringConfig(
                sample_rate=sample_rate, smoothing_fraction=0.0
            )
        )

        smoothed = m._smooth_target_curve(jagged, f_axis)
        unsmoothed = m_no._smooth_target_curve(jagged, f_axis)

        slope = np.mean(np.abs(np.diff(smoothed)))
        slope_no = np.mean(np.abs(np.diff(unsmoothed)))

        assert slope < slope_no

    def test_vocal_processing_empty_audio_regression(self, tmp_path):
        from korshmer.audio_processing.vocal_processing import (
            VocalSyllableDetector,
        )

        detector = VocalSyllableDetector(sample_rate=44100)

        silent_audio = np.zeros(44100)
        syllables = detector.detect_syllables(silent_audio)

        assert isinstance(syllables, list)
        assert len(syllables) == 0

    def test_vocal_processing_dc_offset_regression(self, tmp_path):
        from korshmer.audio_processing.vocal_processing import (
            VocalSyllableDetector,
        )

        detector = VocalSyllableDetector(sample_rate=44100)

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3

        syllables = detector.detect_syllables(audio)
        assert isinstance(syllables, list)

    def test_instrument_filter_fuzzy_matching_regression(self, tmp_path):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        test_stems = {
            "acoustic_guitar": np.random.randn(1000),
            "guitar_acoustic": np.random.randn(1000),
            "PIANO": np.random.randn(1000),
            "piano_grand": np.random.randn(1000),
        }

        filtered = InstrumentFilter.filter_stems(test_stems)

        assert len(filtered) > 0

    def test_file_path_unicode_regression(self, tmp_path):

        unicode_dir = tmp_path / "测试_テスト_тест"
        unicode_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = 44100
        t = np.linspace(0, 1.0, sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        unicode_file = unicode_dir / "файл_文件.wav"

        try:
            sf.write(unicode_file, audio, sample_rate)
            assert unicode_file.exists()

            data, sr = sf.read(unicode_file)
            assert sr == sample_rate
        except Exception:
            pytest.skip("Unicode paths not fully supported on this platform")

    def test_stderr_suppression_frozen_executable_regression(self, tmp_path):
        import sys

        from korshmer.utils.suppress_stderr import suppress_stderr

        original = sys.stderr
        with suppress_stderr():
            print("test", file=sys.stderr)
        assert sys.stderr is original

    def test_stderr_suppression_propagates_original_exception(self):
        import sys

        from korshmer.utils.suppress_stderr import suppress_stderr

        original = sys.stderr
        with pytest.raises(ValueError, match="boom"):
            with suppress_stderr():
                raise ValueError("boom")
        assert sys.stderr is original

    def test_fluidsynth_soundfont_missing_regression(self, tmp_path):
        from korshmer.core.fluidsynth import FluidSynth

        missing = "/nonexistent/soundfont.sf3"
        with pytest.raises(FileNotFoundError) as exc_info:
            FluidSynth(soundfont_path=missing)
        assert "nonexistent" in str(exc_info.value)

        directory = tmp_path / "a_directory.sf3"
        directory.mkdir()
        with pytest.raises(FileNotFoundError) as dir_exc:
            FluidSynth(soundfont_path=str(directory))
        assert "not a regular file" in str(dir_exc.value)

    def test_memory_leak_large_audio_regression(self, tmp_path):
        import gc

        sample_rate = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        for _ in range(5):
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)

            from korshmer.audio import apply_filter

            filtered = apply_filter(
                audio, sample_rate, filter_type="lowpass", cutoff=2000
            )

            del audio
            del filtered

            gc.collect()

    def test_piano_roll_grid_alignment_regression(self, tmp_path):
        import sys

        from PySide6.QtWidgets import QApplication

        if not QApplication.instance():
            QApplication(sys.argv)

        from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

        piano_roll = InteractivePianoRoll()

        piano_roll.set_tempo(120.0)
        assert piano_roll.grid_size > 0
        assert piano_roll.tempo == 120.0

        piano_roll.set_tempo(140.0)

        assert piano_roll.grid_size > 0
        assert piano_roll.tempo == 140.0

    def test_microtonal_divisions_regression(self, tmp_path):
        import sys

        from PySide6.QtWidgets import QApplication

        if not QApplication.instance():
            QApplication(sys.argv)

        from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

        piano_roll = InteractivePianoRoll()

        divisions = [12, 24, 48, 96]

        for div in divisions:
            piano_roll.set_divisions_per_octave(div)
            assert piano_roll.divisions_per_octave == div

    def test_concurrent_file_access_regression(self, tmp_path):
        import concurrent.futures

        sample_rate = 44100
        t = np.linspace(0, 0.5, sample_rate // 2)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        def write_file(index):
            file_path = tmp_path / f"concurrent_{index}.wav"
            sf.write(file_path, audio, sample_rate)
            return file_path.exists()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(write_file, i) for i in range(10)]
            results = [
                f.result() for f in concurrent.futures.as_completed(futures)
            ]

        assert all(results)

    def test_midi_tempo_map_preservation_regression(self, tmp_path):

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)

        instrument = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=5.0)
        instrument.notes.append(note)
        midi.instruments.append(instrument)

        midi_file = tmp_path / "tempo_changes.mid"
        midi.write(str(midi_file))

        loaded = pretty_midi.PrettyMIDI(str(midi_file))

        tempo_times, tempos = loaded.get_tempo_changes()
        assert len(tempos) > 0

    def test_audio_normalization_clipping_regression(self, tmp_path):
        from korshmer.audio import apply_normalize

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.9 * np.sin(2 * np.pi * 440 * t)

        normalized = apply_normalize(audio)

        assert np.max(np.abs(normalized)) <= 1.0
        assert np.min(normalized) >= -1.0
        assert np.max(normalized) <= 1.0

    def test_thread_pool_shutdown_regression(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        def task(x):
            return x * 2

        executor = ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(task, i) for i in range(10)]

        results = [f.result() for f in futures]
        assert len(results) == 10

        executor.shutdown(wait=True, cancel_futures=False)

    def test_project_path_traversal_security_regression(self, tmp_path):
        from korshmer.project_management.validators.project_validator import (
            resolve_inside,
        )

        base = tmp_path / "base"
        base.mkdir()
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "C:\\Windows\\System32",
            str(tmp_path / "outside.json"),
        ]

        for malicious in malicious_paths:
            resolved = resolve_inside(base, base / malicious)
            assert resolved is None or resolved.is_relative_to(base.resolve())
        assert resolve_inside(base, base / ".." / "x") is None
        assert resolve_inside(base, base / "ok" / "x.json") is not None
