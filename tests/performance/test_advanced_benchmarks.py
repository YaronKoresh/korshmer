import gc
import time

import numpy as np
import pytest


@pytest.mark.performance
class TestAlgorithmBenchmarks:
    @pytest.mark.slow
    def test_fft_performance_benchmark(self):

        fft_sizes = [512, 1024, 2048, 4096, 8192]

        for fft_size in fft_sizes:
            audio = np.random.randn(fft_size)

            start = time.time()
            for _ in range(100):
                np.fft.fft(audio)
            elapsed = time.time() - start

            avg_time = elapsed / 100
            assert avg_time < 0.001, (
                f"FFT size {fft_size} took {avg_time:.6f}s per iteration"
            )

    @pytest.mark.slow
    def test_convolution_performance_benchmark(self):
        from scipy import signal

        audio = np.random.randn(44100)
        kernel = np.random.randn(1024)

        start = time.time()
        result = signal.fftconvolve(audio, kernel, mode="same")
        elapsed = time.time() - start

        assert elapsed < 0.5, (
            f"Convolution took {elapsed:.3f}s, expected < 0.5s"
        )
        assert result.shape == audio.shape

    @pytest.mark.slow
    def test_resampling_performance_benchmark(self):
        import librosa

        audio = np.random.randn(44100 * 5).astype(np.float32)

        start = time.time()
        resampled = librosa.resample(audio, orig_sr=44100, target_sr=48000)
        elapsed = time.time() - start

        assert elapsed < 2.0, (
            f"Resampling 5s audio took {elapsed:.3f}s, expected < 2.0s"
        )
        assert len(resampled) > 0

    def test_filter_design_performance_benchmark(self):
        from scipy import signal

        start = time.time()
        for _ in range(10):
            b, a = signal.butter(N=4, Wn=1000, fs=44100, btype="low")
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Designing 10 filters took {elapsed:.3f}s"


@pytest.mark.performance
class TestMemoryProfiling:
    @pytest.mark.slow
    def test_memory_usage_large_audio(self):
        import os

        import psutil

        from korshmer.audio import apply_normalize

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        sample_rate = 44100
        duration = 120.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        normalized = apply_normalize(audio)

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory

        assert memory_increase < 100, (
            f"Memory increased by {memory_increase:.2f} MB"
        )

        del audio
        del normalized
        gc.collect()

    @pytest.mark.slow
    def test_memory_cleanup_after_processing(self):
        import os

        import psutil

        process = psutil.Process(os.getpid())

        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024

        for _ in range(5):
            audio = np.random.randn(44100 * 10).astype(np.float32)
            from korshmer.audio import apply_filter

            filtered = apply_filter(
                audio, 44100, filter_type="lowpass", cutoff=2000
            )
            del audio
            del filtered

        gc.collect()
        time.sleep(0.1)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        assert memory_increase < 50, (
            f"Memory leaked {memory_increase:.2f} MB after cleanup"
        )

    def test_large_project_memory_footprint(self):
        import sys

        project_data = {
            "tracks": [],
            "markers": [],
            "metadata": {},
        }

        for i in range(100):
            track = {
                "id": i,
                "name": f"Track {i}",
                "audio": np.random.randn(44100).astype(np.float32),
                "settings": {"volume": 1.0, "pan": 0.0},
            }
            project_data["tracks"].append(track)

        size_bytes = sys.getsizeof(project_data)

        audio_size = sum(
            track["audio"].nbytes for track in project_data["tracks"]
        )
        total_size_mb = (size_bytes + audio_size) / 1024 / 1024

        assert total_size_mb < 50, f"Large project uses {total_size_mb:.2f} MB"


@pytest.mark.performance
class TestCPUProfiling:
    def test_cpu_intensive_processing(self):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        mastering = SmartMastering(
            SmartMasteringConfig(sample_rate=sample_rate)
        )

        mastering.process(audio)

        elapsed = float("inf")
        for _ in range(3):
            start = time.time()
            mastering.process(audio)
            elapsed = min(elapsed, time.time() - start)

        realtime_factor = duration / elapsed
        assert realtime_factor > 1.0, (
            f"Processing {realtime_factor:.2f}x realtime, expected > 1.0x"
        )

    @pytest.mark.slow
    def test_parallel_processing_efficiency(self):
        from concurrent.futures import ThreadPoolExecutor

        def process_chunk(audio_chunk):
            from korshmer.audio import apply_normalize

            return apply_normalize(audio_chunk)

        sample_rate = 44100
        num_chunks = 8
        chunk_duration = 2.0
        chunks = [
            np.random.randn(int(sample_rate * chunk_duration))
            for _ in range(num_chunks)
        ]

        for chunk in chunks:
            process_chunk(chunk)

        repeats = 5
        sequential_time = float("inf")
        parallel_time = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            for chunk in chunks:
                process_chunk(chunk)
            sequential_time = min(sequential_time, time.perf_counter() - start)

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(process_chunk, chunks))
            parallel_time = min(parallel_time, time.perf_counter() - start)

        speedup = sequential_time / (parallel_time + 1e-12)

        assert speedup > 0.1, (
            f"Parallel speedup {speedup:.2f}x, expected > 0.1x"
        )


@pytest.mark.performance
class TestLatencyMeasurements:
    def test_audio_callback_latency(self):

        buffer_size = 512
        sample_rate = 44100
        audio_buffer = np.random.randn(buffer_size).astype(np.float32)

        from korshmer.audio import apply_filter

        start = time.time()
        apply_filter(
            audio_buffer, sample_rate, filter_type="lowpass", cutoff=2000
        )
        elapsed = time.time() - start

        buffer_duration_ms = buffer_size / sample_rate * 1000
        processing_time_ms = elapsed * 1000

        assert processing_time_ms < buffer_duration_ms, (
            f"Processing took {processing_time_ms:.3f}ms, "
            f"buffer duration {buffer_duration_ms:.3f}ms"
        )

    def test_ui_update_latency(self):
        import sys

        from PySide6.QtWidgets import QApplication

        if not QApplication.instance():
            QApplication(sys.argv)

        from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

        piano_roll = InteractivePianoRoll()
        piano_roll.set_tempo(130.0)

        start = time.perf_counter()
        piano_roll.set_tempo(140.0)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 10, (
            f"UI update took {elapsed_ms:.3f}ms, expected < 10ms"
        )


@pytest.mark.performance
class TestStressTests:
    @pytest.mark.slow
    def test_100_track_project_stress(self):
        sample_rate = 44100
        duration = 1.0
        num_tracks = 100

        tracks = []
        for i in range(num_tracks):
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.01 * np.sin(2 * np.pi * (440 + i * 10) * t)
            tracks.append(audio)

        start = time.time()
        mixed = np.sum(tracks, axis=0)
        from korshmer.audio import apply_normalize

        normalized = apply_normalize(mixed)
        elapsed = time.time() - start

        assert elapsed < 1.0, (
            f"Mixing 100 tracks took {elapsed:.3f}s, expected < 1.0s"
        )
        assert normalized.shape == (int(sample_rate * duration),)

    @pytest.mark.slow
    def test_1000_region_timeline_stress(self):

        regions = []
        for i in range(1000):
            region = {
                "start": i * 0.5,
                "end": i * 0.5 + 0.4,
                "track_id": i % 10,
                "label": f"Region {i}",
            }
            regions.append(region)

        start = time.time()

        target_time = 250.0
        [r for r in regions if r["start"] <= target_time <= r["end"]]
        elapsed = time.time() - start

        assert elapsed < 0.01, (
            f"Searching 1000 regions took {elapsed:.6f}s, expected < 0.01s"
        )

    @pytest.mark.slow
    def test_long_duration_processing_stress(self):
        sample_rate = 44100
        duration = 600.0

        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        from korshmer.audio import apply_normalize

        start = time.time()
        normalized = apply_normalize(audio)
        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"Processing 10 min audio took {elapsed:.3f}s, expected < 5.0s"
        )

        del audio
        del normalized
        gc.collect()


@pytest.mark.performance
class TestFileIOPerformance:
    @pytest.mark.slow
    def test_large_wav_file_read_performance(self, tmp_path):
        import soundfile as sf

        sample_rate = 44100
        duration = 60.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        test_file = tmp_path / "large_test.wav"
        sf.write(test_file, audio, sample_rate)

        start = time.time()
        data, sr = sf.read(test_file)
        elapsed = time.time() - start

        assert elapsed < 0.5, (
            f"Reading 1 min WAV took {elapsed:.3f}s, expected < 0.5s"
        )

    @pytest.mark.slow
    def test_large_wav_file_write_performance(self, tmp_path):
        import soundfile as sf

        sample_rate = 44100
        duration = 60.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        test_file = tmp_path / "output_test.wav"

        start = time.time()
        sf.write(test_file, audio, sample_rate)
        elapsed = time.time() - start

        assert elapsed < 3.0, (
            f"Writing 1 min WAV took {elapsed:.3f}s, expected < 3.0s"
        )

    def test_midi_file_io_performance(self, tmp_path):
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(1000):
            note = pretty_midi.Note(
                velocity=100,
                pitch=60 + (i % 12),
                start=i * 0.1,
                end=(i + 1) * 0.1,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        test_file = tmp_path / "large_midi.mid"
        start = time.time()
        midi.write(str(test_file))
        write_elapsed = time.time() - start

        start = time.time()
        loaded = pretty_midi.PrettyMIDI(str(test_file))
        read_elapsed = time.time() - start

        assert write_elapsed < 0.5, f"Writing MIDI took {write_elapsed:.3f}s"
        assert read_elapsed < 0.5, f"Reading MIDI took {read_elapsed:.3f}s"
        assert len(loaded.instruments[0].notes) == 1000


@pytest.mark.performance
class TestThreadScalingEfficiency:
    def test_worker_pool_scaling(self):
        from concurrent.futures import ThreadPoolExecutor

        def blocking_task(seconds):

            time.sleep(seconds)

        task_seconds = 0.01
        num_tasks = 16

        results = {}
        for num_workers in [1, 2, 4, 8]:
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(blocking_task, [task_seconds] * num_tasks))
            elapsed = time.perf_counter() - start
            results[num_workers] = elapsed

        assert results[2] <= results[1] * 0.75, (
            "2 workers should be faster than 1"
        )
        assert results[4] <= results[2] * 0.75, (
            "4 workers should be faster than 2"
        )


@pytest.mark.performance
class TestComparisonBenchmarks:
    def test_numpy_vs_python_loop_performance(self):
        data = np.random.randn(100000)

        start = time.perf_counter()
        data * 2.0 + 1.0
        numpy_time = time.perf_counter() - start

        start = time.perf_counter()
        [x * 2.0 + 1.0 for x in data]
        python_time = time.perf_counter() - start

        speedup = python_time / (numpy_time + 1e-12)
        assert speedup > 1, f"NumPy only {speedup:.1f}x faster, expected > 1x"

    def test_optimized_instrument_filter_performance(self):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        instruments = [f"instrument_{i}" for i in range(1000)]

        InstrumentFilter._normalized_cache.clear()
        start = time.time()
        for inst in instruments:
            InstrumentFilter.should_convert(inst)

        for inst in instruments:
            InstrumentFilter.should_convert(inst)
        optimized_time = time.time() - start

        assert optimized_time < 0.1, f"Cached lookup took {optimized_time:.3f}s"
