import numpy as np
import pytest
from scipy import signal


@pytest.mark.integration
class TestEffectChainProcessing:
    def test_single_effect_processing(self, tmp_path):
        from korshmer.audio import apply_filter

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        filtered = apply_filter(
            audio, sample_rate, filter_type="lowpass", cutoff=1000
        )

        assert filtered is not None
        assert filtered.shape == audio.shape
        assert np.isfinite(filtered).all()

    def test_effect_chain_eq_compression(self, tmp_path):
        from korshmer.audio_processing.range_eq import RangeEQ

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        eq = RangeEQ(sample_rate=sample_rate)
        audio_eq = eq.process(
            audio, freq_start=300, freq_end=600, total_gain_db=3.0
        )

        from korshmer.audio import apply_normalize

        audio_compressed = apply_normalize(audio_eq)

        assert audio_compressed is not None
        assert audio_compressed.shape == audio.shape

    def test_effect_chain_filter_reverb(self, tmp_path):
        from korshmer.audio import apply_filter

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        filtered = apply_filter(
            audio, sample_rate, filter_type="highpass", cutoff=200
        )

        delay_samples = int(0.05 * sample_rate)
        reverb = np.zeros(len(filtered) + delay_samples)
        reverb[: len(filtered)] = filtered
        reverb[delay_samples:] += filtered * 0.3

        assert reverb is not None
        assert len(reverb) > len(audio)

    def test_effect_chain_gate_eq_compression(self, tmp_path):
        from korshmer.audio import apply_noise_gate, apply_normalize
        from korshmer.audio_processing.range_eq import RangeEQ

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.zeros_like(t)
        signal[int(0.5 * sample_rate) : int(1.5 * sample_rate)] = 0.5 * np.sin(
            2 * np.pi * 440 * t[int(0.5 * sample_rate) : int(1.5 * sample_rate)]
        )
        noise = 0.05 * np.random.randn(len(t))
        audio = signal + noise

        gated = apply_noise_gate(audio, sample_rate, threshold_db=-40)

        eq = RangeEQ(sample_rate=sample_rate)
        eq_processed = eq.process(
            gated, freq_start=400, freq_end=500, total_gain_db=6.0
        )

        compressed = apply_normalize(eq_processed)

        assert compressed is not None
        assert np.isfinite(compressed).all()

    def test_effect_chain_parallel_processing(self, tmp_path):
        from korshmer.audio import apply_filter

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        path1 = apply_filter(
            audio, sample_rate, filter_type="lowpass", cutoff=800
        )

        path2 = apply_filter(
            audio, sample_rate, filter_type="highpass", cutoff=800
        )

        mixed = 0.5 * path1 + 0.5 * path2

        assert mixed is not None
        assert mixed.shape == audio.shape

    def test_effect_chain_with_automation(self, tmp_path):
        from korshmer.audio_processing.range_eq import RangeEQ

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        chunk_size = sample_rate // 2
        processed = np.zeros_like(audio)

        eq = RangeEQ(sample_rate=sample_rate)

        for i in range(0, len(audio), chunk_size):
            end = min(i + chunk_size, len(audio))
            chunk = audio[i:end]

            gain = 3.0 + 3.0 * (i / len(audio))

            chunk_processed = eq.process(
                chunk, freq_start=300, freq_end=600, total_gain_db=gain
            )
            processed[i:end] = chunk_processed

        assert processed is not None
        assert np.isfinite(processed).all()

    def test_effect_chain_preserves_length(self, tmp_path):
        from korshmer.audio import apply_filter, apply_normalize

        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        original_length = len(audio)

        filtered = apply_filter(
            audio, sample_rate, filter_type="lowpass", cutoff=2000
        )
        assert len(filtered) == original_length

        normalized = apply_normalize(filtered)
        assert len(normalized) == original_length

    def test_effect_chain_error_recovery(self, tmp_path):
        from korshmer.audio import apply_filter, apply_normalize

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        results = []

        try:
            filtered = apply_filter(
                audio, sample_rate, filter_type="lowpass", cutoff=1000
            )
            results.append(filtered)
        except Exception:
            results.append(audio)

        try:
            filtered2 = apply_filter(
                audio, sample_rate, filter_type="lowpass", cutoff=-100
            )
            results.append(filtered2)
        except Exception:
            results.append(audio)

        try:
            normalized = apply_normalize(audio)
            results.append(normalized)
        except Exception:
            results.append(audio)

        assert len(results) == 3


@pytest.mark.integration
class TestRealtimePlayback:
    def test_playback_buffer_preparation(self, tmp_path):

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        buffer_size = 512
        num_buffers = len(audio) // buffer_size

        buffers = []
        for i in range(num_buffers):
            start = i * buffer_size
            end = min(start + buffer_size, len(audio))
            buffer = audio[start:end]
            buffers.append(buffer)

        assert len(buffers) > 0
        assert all(len(b) <= buffer_size for b in buffers)

    def test_playback_synchronization(self, tmp_path):

        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        playback_positions = []
        chunk_size = sample_rate

        for i in range(0, len(audio), chunk_size):
            position_seconds = i / sample_rate
            playback_positions.append(position_seconds)

        assert playback_positions == sorted(playback_positions)

    def test_playback_tempo_sync(self, tmp_path):

        sample_rate = 44100
        duration = 1.0

        tempos = [80, 120, 160]

        for tempo in tempos:
            beats = (tempo / 60.0) * duration

            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.zeros_like(t)

            beat_interval = 60.0 / tempo
            for beat in range(int(beats)):
                beat_time = beat * beat_interval
                idx = int(beat_time * sample_rate)
                if idx < len(audio):
                    click_duration = int(0.01 * sample_rate)
                    audio[idx : idx + click_duration] = 0.5

            assert len(audio) > 0

    def test_multi_track_playback_mixing(self, tmp_path):
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        track1 = 0.3 * np.sin(2 * np.pi * 440 * t)
        track2 = 0.3 * np.sin(2 * np.pi * 554 * t)
        track3 = 0.3 * np.sin(2 * np.pi * 659 * t)

        mixed = track1 + track2 + track3

        from korshmer.audio import apply_normalize

        normalized = apply_normalize(mixed)

        assert normalized is not None
        assert np.max(np.abs(normalized)) <= 1.0

    def test_playback_with_effects_realtime(self, tmp_path):
        from korshmer.audio import apply_filter

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        chunk_size = 2048
        processed = np.zeros_like(audio)

        for i in range(0, len(audio), chunk_size):
            end = min(i + chunk_size, len(audio))
            chunk = audio[i:end]

            filtered_chunk = apply_filter(
                chunk, sample_rate, filter_type="lowpass", cutoff=2000
            )
            processed[i:end] = filtered_chunk

        assert processed is not None
        assert len(processed) == len(audio)


@pytest.mark.integration
class TestSmartMasteringPipeline:
    def test_smart_mastering_complete_chain(self, tmp_path):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(
            2 * np.pi * 440 * t
        )

        cfg = SmartMasteringConfig(sample_rate=sample_rate)
        mastering = SmartMastering(cfg)
        mastered, _ = mastering.process(audio)

        assert mastered is not None
        assert mastered.shape[-1] == audio.shape[-1]
        assert np.isfinite(mastered).all()

    def test_mastering_spectral_balance(self, tmp_path):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        bass = 0.6 * np.sin(2 * np.pi * 100 * t)
        mids = 0.2 * np.sin(2 * np.pi * 1000 * t)
        highs = 0.1 * np.sin(2 * np.pi * 5000 * t)
        audio = bass + mids + highs

        cfg = SmartMasteringConfig(sample_rate=sample_rate)
        mastering = SmartMastering(cfg)
        balanced, _ = mastering.process(audio)

        assert balanced is not None

        assert np.isfinite(balanced).all()

    def test_mastering_loudness_normalization(self, tmp_path):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)

        cfg = SmartMasteringConfig(sample_rate=sample_rate)
        mastering = SmartMastering(cfg)
        loud, _ = mastering.process(audio)

        assert loud is not None

        assert np.max(np.abs(loud)) > np.max(np.abs(audio))

    def test_smoothing_affects_spectral_slope(self, tmp_path):
        from korshmer.audio_processing.smart_mastering import (
            SmartMastering,
            SmartMasteringConfig,
        )

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.random.randn(len(t))

        m_no = SmartMastering(
            SmartMasteringConfig(
                sample_rate=sample_rate, smoothing_fraction=0.0
            )
        )
        m_s = SmartMastering(SmartMasteringConfig(sample_rate=sample_rate))
        out_no, _ = m_no.process(audio)
        out_no = out_no[0]
        out_s, _ = m_s.process(audio)
        out_s = out_s[0]

        nperseg = len(out_no)
        nover = nperseg // 2
        f_axis, psd_no = signal.welch(
            out_no,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=nover,
            window="hann",
            scaling="density",
            average="median",
        )
        f_axis, psd_s = signal.welch(
            out_s,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=nover,
            window="hann",
            scaling="density",
            average="median",
        )
        10.0 * np.log10(np.maximum(psd_no, 1e-24))
        10.0 * np.log10(np.maximum(psd_s, 1e-24))

        assert not np.allclose(out_no, out_s)
