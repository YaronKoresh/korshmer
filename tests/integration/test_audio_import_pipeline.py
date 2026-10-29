import numpy as np
import pytest
import soundfile as sf


@pytest.mark.integration
class TestAudioImportPipeline:
    def test_audio_import_and_analysis(self, tmp_path):
        from korshmer.audio import analyse_tempo, estimate_meter

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio_file = tmp_path / "test_audio.wav"
        sf.write(audio_file, audio, sample_rate)

        tempo = analyse_tempo(str(audio_file))
        assert tempo is not None
        assert isinstance(tempo, (int, float))
        assert 20 <= tempo <= 300

        meter = estimate_meter(str(audio_file))
        assert meter is not None
        assert isinstance(meter, (int, float))

    def test_audio_import_processing_chain(self, tmp_path):
        from korshmer.audio import apply_filter, apply_normalize

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 880 * t)
        audio_file = tmp_path / "test_chain.wav"
        sf.write(audio_file, audio, sample_rate)

        filtered = apply_filter(
            audio, sample_rate, filter_type="lowpass", cutoff=1000
        )
        assert filtered is not None
        assert filtered.shape == audio.shape

        normalized = apply_normalize(filtered)
        assert normalized is not None
        assert np.max(np.abs(normalized)) <= 1.0

    def test_audio_import_with_different_formats(self, tmp_path):
        sample_rates = [22050, 44100, 48000]

        for sr in sample_rates:
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            audio_file = tmp_path / f"test_{sr}.wav"
            sf.write(audio_file, audio, sr)

            data, read_sr = sf.read(audio_file)
            assert read_sr == sr
            assert data.shape[0] == audio.shape[0]

    def test_audio_import_stereo_to_mono(self, tmp_path):
        from korshmer.audio import apply_mono_mix

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 660 * t)
        stereo = np.stack([left, right], axis=-1)
        audio_file = tmp_path / "test_stereo.wav"
        sf.write(audio_file, stereo, sample_rate)

        data, sr = sf.read(audio_file)
        mono = apply_mono_mix(data)
        assert mono.ndim == 1
        assert len(mono) == len(left)

    def test_audio_processing_silence_trim(self, tmp_path):
        from korshmer.audio import trim_audio_silence

        sample_rate = 44100
        silence_duration = 0.5
        signal_duration = 1.0

        silence = np.zeros(int(sample_rate * silence_duration))
        t = np.linspace(0, signal_duration, int(sample_rate * signal_duration))
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.concatenate([silence, signal, silence])

        trimmed = trim_audio_silence(audio, sample_rate)
        assert len(trimmed) < len(audio)
        assert len(trimmed) > len(signal) * 0.8

    def test_audio_onset_detection_pipeline(self, tmp_path):
        from korshmer.audio import slice_audio_onsets

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = np.zeros_like(t)
        onset_times = [0.5, 1.0, 1.5]
        for onset_time in onset_times:
            int(onset_time * sample_rate)
            decay = np.exp(-10 * (t - onset_time))
            decay[t < onset_time] = 0
            audio += 0.5 * decay * np.sin(2 * np.pi * 440 * (t - onset_time))

        slices = slice_audio_onsets(audio, sample_rate)
        assert isinstance(slices, list)

        assert len(slices) >= 1

    def test_audio_import_error_handling(self, tmp_path):
        from korshmer.audio import analyse_tempo

        result = analyse_tempo("nonexistent_file.wav")
        assert result is None or isinstance(result, (int, float))

        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a valid audio file")

        result = analyse_tempo(str(invalid_file))

        assert result is None or isinstance(result, (int, float))

    def test_audio_reverse_processing(self, tmp_path):
        from korshmer.audio import apply_reverse_audio

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) * t

        reversed_audio = apply_reverse_audio(audio)
        assert reversed_audio.shape == audio.shape

        assert np.allclose(reversed_audio, audio[::-1])

    def test_audio_normalization_pipeline(self, tmp_path):
        from korshmer.audio import save_normalized_audio

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        output_file = tmp_path / "normalized.wav"
        save_normalized_audio(audio, sample_rate, str(output_file))

        assert output_file.exists()
        data, sr = sf.read(output_file)
        assert sr == sample_rate

        assert np.max(np.abs(data)) > 0.8

    def test_audio_transient_shaping(self, tmp_path):
        from korshmer.audio import apply_transient_shaper

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = 0.5 * np.exp(-5 * t) * np.sin(2 * np.pi * 440 * t)

        shaped = apply_transient_shaper(
            audio, sample_rate, attack=2.0, sustain=0.5
        )
        assert shaped is not None
        assert shaped.shape == audio.shape
        assert np.isfinite(shaped).all()


@pytest.mark.integration
class TestAudioEffectsPipeline:
    def test_noise_gate_pipeline(self, tmp_path):
        from korshmer.audio import apply_noise_gate

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        signal = np.zeros_like(t)
        signal_mask = (t > 0.8) & (t < 1.2)
        signal[signal_mask] = 0.5 * np.sin(2 * np.pi * 440 * t[signal_mask])

        noise = 0.05 * np.random.randn(len(t))
        audio = signal + noise

        gated = apply_noise_gate(audio, sample_rate, threshold_db=-40)
        assert gated is not None
        assert gated.shape == audio.shape

        quiet_section = gated[: int(0.5 * sample_rate)]
        assert np.mean(np.abs(quiet_section)) < np.mean(
            np.abs(audio[: int(0.5 * sample_rate)])
        )

    def test_pitch_shift_pipeline(self, tmp_path):
        from korshmer.audio import pitch_shift_file

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        input_file = tmp_path / "input.wav"
        output_file = tmp_path / "output.wav"
        sf.write(input_file, audio, sample_rate)

        pitch_shift_file(str(input_file), str(output_file), semitones=2)

        assert output_file.exists()
        shifted, sr = sf.read(output_file)
        assert sr == sample_rate
        assert len(shifted) > 0
        assert np.isfinite(shifted).all()
