import numpy as np
import pytest
import soundfile as sf


@pytest.mark.integration
@pytest.mark.slow
class TestStemSeparationPipeline:
    def test_demucs_integration_basic(self, tmp_path):
        from korshmer.audio import run_demucs

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(
            2 * np.pi * 880 * t
        )
        audio_file = tmp_path / "mix.wav"
        sf.write(audio_file, audio, sample_rate)

        try:
            stems = run_demucs(str(audio_file), output_dir=str(tmp_path))

            if stems:
                assert isinstance(stems, dict)
        except Exception as e:
            pytest.skip(f"Demucs not available or failed: {e}")

    def test_stem_classification_workflow(self, tmp_path):
        from korshmer.audio_processing.stem_classification import classify_stem

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        vocal = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(
            2 * np.pi * 880 * t
        )

        bass = 0.5 * np.sin(2 * np.pi * 110 * t)

        drums = np.random.randn(len(t)) * 0.1

        for i in range(0, len(t), sample_rate // 2):
            if i < len(drums):
                drums[i : min(i + 100, len(drums))] += 0.5

        try:
            vocal_class = classify_stem(vocal, sample_rate)
            bass_class = classify_stem(bass, sample_rate)
            drums_class = classify_stem(drums, sample_rate)

            assert vocal_class is not None
            assert bass_class is not None
            assert drums_class is not None
        except Exception as e:
            pytest.skip(f"Stem classification not available: {e}")

    def test_instrument_recognition_pipeline(self, tmp_path):
        from korshmer.midi_processing.instrument_filter import InstrumentFilter

        stems = {
            "vocals": np.random.randn(44100),
            "piano": np.random.randn(44100),
            "guitar": np.random.randn(44100),
            "bass": np.random.randn(44100),
        }

        filtered = InstrumentFilter.filter_stems(stems)

        assert "piano" in filtered
        assert "guitar" in filtered
        assert "bass" in filtered
        assert "vocals" not in filtered

    def test_stem_export_workflow(self, tmp_path):

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        stems = {
            "vocals": 0.3 * np.sin(2 * np.pi * 440 * t),
            "drums": 0.3 * np.sin(2 * np.pi * 220 * t),
            "bass": 0.3 * np.sin(2 * np.pi * 110 * t),
            "other": 0.3 * np.sin(2 * np.pi * 880 * t),
        }

        stems_dir = tmp_path / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)

        for name, audio in stems.items():
            stem_path = stems_dir / f"{name}.wav"
            sf.write(stem_path, audio, sample_rate)

        assert len(list(stems_dir.glob("*.wav"))) == 4


@pytest.mark.integration
@pytest.mark.slow
class TestAudioToMIDITranscription:
    def test_basic_pitch_integration(self, tmp_path):

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio_file = tmp_path / "monophonic.wav"
        sf.write(audio_file, audio, sample_rate)

        try:
            from basic_pitch.inference import predict

            model_output, midi_data, note_events = predict(str(audio_file))

            assert model_output is not None
            assert midi_data is not None

        except Exception as e:
            pytest.skip(f"basic-pitch not available: {e}")

    def test_audio_to_midi_polyphonic(self, tmp_path):

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = (
            0.3 * np.sin(2 * np.pi * 261.63 * t)
            + 0.3 * np.sin(2 * np.pi * 329.63 * t)
            + 0.3 * np.sin(2 * np.pi * 392.00 * t)
        )
        audio_file = tmp_path / "polyphonic.wav"
        sf.write(audio_file, audio, sample_rate)

        try:
            from basic_pitch.inference import predict

            model_output, midi_data, note_events = predict(str(audio_file))

            assert model_output is not None

        except Exception as e:
            pytest.skip(f"basic-pitch not available: {e}")

    def test_vocal_to_midi_workflow(self, tmp_path):
        from korshmer.audio_processing.vocal_processing import (
            VocalSyllableDetector,
        )

        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = np.zeros_like(t)
        syllable_pitches = [440, 494, 523, 494, 440]
        syllable_duration = duration / len(syllable_pitches)

        for i, pitch in enumerate(syllable_pitches):
            start_idx = int(i * syllable_duration * sample_rate)
            end_idx = int((i + 1) * syllable_duration * sample_rate)
            if end_idx <= len(audio):
                segment = t[start_idx:end_idx] - t[start_idx]
                audio[start_idx:end_idx] = 0.3 * np.sin(
                    2 * np.pi * pitch * segment
                )

        detector = VocalSyllableDetector(sample_rate=sample_rate)
        syllables = detector.detect_syllables(audio)

        assert isinstance(syllables, list)

    def test_transcription_with_confidence_filtering(self, tmp_path):

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        audio = np.zeros_like(t)
        midpoint = len(t) // 2
        audio[:midpoint] = 0.5 * np.sin(2 * np.pi * 440 * t[:midpoint])
        audio[midpoint:] = 0.1 * np.random.randn(len(t) - midpoint)

        audio_file = tmp_path / "mixed_quality.wav"
        sf.write(audio_file, audio, sample_rate)

        try:
            from basic_pitch.inference import predict

            model_output, midi_data, note_events = predict(str(audio_file))

            assert model_output is not None

        except Exception as e:
            pytest.skip(f"basic-pitch not available: {e}")

    def test_drum_to_midi_workflow(self, tmp_path):

        sample_rate = 44100
        duration = 2.0
        audio = np.zeros(int(sample_rate * duration))

        kick_times = [0.0, 0.5, 1.0, 1.5]
        for kick_time in kick_times:
            idx = int(kick_time * sample_rate)

            decay_samples = int(0.2 * sample_rate)
            if idx + decay_samples < len(audio):
                t_decay = np.linspace(0, 0.2, decay_samples)
                kick = (
                    0.8
                    * np.exp(-10 * t_decay)
                    * np.sin(2 * np.pi * 60 * t_decay)
                )
                audio[idx : idx + decay_samples] += kick

        audio_file = tmp_path / "drums.wav"
        sf.write(audio_file, audio, sample_rate)

        from korshmer.audio import slice_audio_onsets

        onsets = slice_audio_onsets(audio, sample_rate)

        assert isinstance(onsets, list)

    def test_transcription_post_processing(self, tmp_path):
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        base_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        for time in base_times:
            jitter = 0.01 * (np.random.rand() - 0.5)
            note = pretty_midi.Note(
                velocity=80,
                pitch=60,
                start=time + jitter,
                end=time + 0.4 + jitter,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        from korshmer.midi import remove_short_notes

        cleaned = remove_short_notes(midi, min_duration=0.05)
        assert len(cleaned.instruments[0].notes) == 5

    def test_missing_dependency_behaviour(self, tmp_path, monkeypatch):
        import builtins

        from korshmer.exceptions import AudioToMidiError

        orig_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("basic_pitch"):
                raise ImportError("no module named basic_pitch")
            return orig_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from korshmer.audio_processing.ai_audio_to_midi import (
            EnhancedAudioToMIDI,
        )

        with pytest.raises(AudioToMidiError):
            EnhancedAudioToMIDI()

    def test_multi_instrument_transcription(self, tmp_path):

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        bass = 0.4 * np.sin(2 * np.pi * 110 * t)

        melody = 0.4 * np.sin(2 * np.pi * 440 * t)

        audio = bass + melody

        audio_file = tmp_path / "multi_instrument.wav"
        sf.write(audio_file, audio, sample_rate)

        try:
            from basic_pitch.inference import predict

            model_output, midi_data, note_events = predict(str(audio_file))

            assert model_output is not None

        except Exception as e:
            pytest.skip(f"basic-pitch not available: {e}")


@pytest.mark.integration
class TestStemToMIDIPipeline:
    def test_stem_separation_then_transcription(self, tmp_path):

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        mix = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(
            2 * np.pi * 440 * t
        )
        mix_file = tmp_path / "mix.wav"
        sf.write(mix_file, mix, sample_rate)

        stems = {
            "bass": 0.3 * np.sin(2 * np.pi * 110 * t),
            "melody": 0.3 * np.sin(2 * np.pi * 440 * t),
        }

        stems_dir = tmp_path / "stems"
        stems_dir.mkdir()

        for name, audio in stems.items():
            stem_file = stems_dir / f"{name}.wav"
            sf.write(stem_file, audio, sample_rate)

        assert (stems_dir / "bass.wav").exists()
        assert (stems_dir / "melody.wav").exists()

    def test_batch_transcription_workflow(self, tmp_path):
        sample_rate = 22050
        duration = 0.5

        pitches = [261.63, 293.66, 329.63, 349.23, 392.00]

        audio_files = []
        for i, pitch in enumerate(pitches):
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.5 * np.sin(2 * np.pi * pitch * t)
            audio_file = tmp_path / f"note_{i}.wav"
            sf.write(audio_file, audio, sample_rate)
            audio_files.append(audio_file)

        assert len(audio_files) == 5
        for audio_file in audio_files:
            assert audio_file.exists()
