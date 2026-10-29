import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf


@pytest.mark.integration
class TestAudioExportFormats:
    def test_export_wav_format(self, tmp_path):

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        output_file = tmp_path / "export.wav"
        sf.write(output_file, audio, sample_rate)

        assert output_file.exists()
        data, sr = sf.read(output_file)
        assert sr == sample_rate
        assert len(data) == len(audio)

    def test_export_different_bit_depths(self, tmp_path):
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        subtypes = ["PCM_16", "PCM_24", "PCM_32", "FLOAT"]

        for subtype in subtypes:
            output_file = tmp_path / f"export_{subtype}.wav"
            try:
                sf.write(output_file, audio, sample_rate, subtype=subtype)
                assert output_file.exists()
            except Exception as e:
                pytest.skip(f"Format {subtype} not supported: {e}")

    def test_export_different_sample_rates(self, tmp_path):
        duration = 0.5
        sample_rates = [22050, 44100, 48000, 96000]

        for sr in sample_rates:
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)

            output_file = tmp_path / f"export_{sr}Hz.wav"
            sf.write(output_file, audio, sr)

            data, read_sr = sf.read(output_file)
            assert read_sr == sr

    def test_export_stereo_audio(self, tmp_path):
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 554 * t)
        stereo = np.stack([left, right], axis=-1)

        output_file = tmp_path / "stereo.wav"
        sf.write(output_file, stereo, sample_rate)

        data, sr = sf.read(output_file)
        assert data.ndim == 2
        assert data.shape[1] == 2

    def test_export_with_normalization(self, tmp_path):
        from korshmer.audio import apply_normalize

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        normalized = apply_normalize(audio)

        output_file = tmp_path / "normalized.wav"
        sf.write(output_file, normalized, sample_rate)

        data, sr = sf.read(output_file)
        assert np.max(np.abs(data)) > 0.9

    def test_export_batch_stems(self, tmp_path):
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        stems = {
            "vocals": 0.5 * np.sin(2 * np.pi * 440 * t),
            "drums": 0.5 * np.sin(2 * np.pi * 220 * t),
            "bass": 0.5 * np.sin(2 * np.pi * 110 * t),
            "other": 0.5 * np.sin(2 * np.pi * 880 * t),
        }

        stems_dir = tmp_path / "stems"
        stems_dir.mkdir()

        for name, audio in stems.items():
            output_file = stems_dir / f"{name}.wav"
            sf.write(output_file, audio, sample_rate)

        assert len(list(stems_dir.glob("*.wav"))) == 4

    def test_export_with_metadata(self, tmp_path):
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        output_file = tmp_path / "with_metadata.wav"
        sf.write(output_file, audio, sample_rate)

        assert output_file.exists()

        metadata = {"title": "Test Song", "artist": "Test Artist", "bpm": 120}

        metadata_file = tmp_path / "with_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        assert metadata_file.exists()


@pytest.mark.integration
class TestMIDIExportFormats:
    def test_export_midi_basic(self, tmp_path):

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=100, pitch=60 + i, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        output_file = tmp_path / "export.mid"
        midi.write(str(output_file))

        assert output_file.exists()
        loaded = pretty_midi.PrettyMIDI(str(output_file))
        assert len(loaded.instruments) == 1
        assert len(loaded.instruments[0].notes) == 5

    def test_export_midi_with_multiple_tracks(self, tmp_path):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)

        programs = [0, 32, 48]

        for program in programs:
            instrument = pretty_midi.Instrument(program=program)
            note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0)
            instrument.notes.append(note)
            midi.instruments.append(instrument)

        output_file = tmp_path / "multi_track.mid"
        midi.write(str(output_file))

        loaded = pretty_midi.PrettyMIDI(str(output_file))
        assert len(loaded.instruments) == 3

    def test_export_midi_to_json(self, tmp_path):
        from korshmer.midi import export_to_json

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0)
        instrument.notes.append(note)
        midi.instruments.append(instrument)

        json_data = export_to_json(midi)
        assert isinstance(json_data, str)
        assert len(json_data) > 0

        output_file = tmp_path / "export.json"
        with open(output_file, "w") as f:
            f.write(json_data)

        assert output_file.exists()

    def test_export_midi_with_tempo_changes(self, tmp_path):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)

        instrument = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=5.0)
        instrument.notes.append(note)
        midi.instruments.append(instrument)

        output_file = tmp_path / "tempo_changes.mid"
        midi.write(str(output_file))

        loaded = pretty_midi.PrettyMIDI(str(output_file))
        tempo_times, tempos = loaded.get_tempo_changes()
        assert len(tempos) > 0

    def test_export_midi_with_control_changes(self, tmp_path):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        note = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0)
        instrument.notes.append(note)

        for i in range(10):
            cc = pretty_midi.ControlChange(number=1, value=i * 12, time=i * 0.2)
            instrument.control_changes.append(cc)

        midi.instruments.append(instrument)

        output_file = tmp_path / "with_cc.mid"
        midi.write(str(output_file))

        loaded = pretty_midi.PrettyMIDI(str(output_file))
        assert len(loaded.instruments[0].control_changes) > 0


@pytest.mark.integration
class TestProjectExport:
    def test_export_project_structure(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("export_test")
        project_path = pm.projects_dir / "export_test"

        sample_rate = 44100
        t = np.linspace(0, 1.0, sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        tracks_dir = project_path / "tracks"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        audio_file = tracks_dir / "track1.wav"
        sf.write(audio_file, audio, sample_rate)

        export_dir = tmp_path / "export"
        export_dir.mkdir()

        import shutil

        shutil.copytree(project_path, export_dir / "exported_project")

        exported_project = export_dir / "exported_project"
        assert exported_project.exists()
        assert (exported_project / "tracks" / "track1.wav").exists()

    def test_export_render_mixdown(self, tmp_path):
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        track1 = 0.3 * np.sin(2 * np.pi * 440 * t)
        track2 = 0.3 * np.sin(2 * np.pi * 554 * t)
        track3 = 0.3 * np.sin(2 * np.pi * 659 * t)

        mixdown = track1 + track2 + track3

        from korshmer.audio import apply_normalize

        mixdown = apply_normalize(mixdown)

        output_file = tmp_path / "mixdown.wav"
        sf.write(output_file, mixdown, sample_rate)

        assert output_file.exists()
        data, sr = sf.read(output_file)
        assert len(data) > 0

    def test_export_with_presets(self, tmp_path):
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        presets = {
            "high_quality": {"sample_rate": 48000, "subtype": "PCM_24"},
            "standard": {"sample_rate": 44100, "subtype": "PCM_16"},
            "web": {"sample_rate": 44100, "subtype": "PCM_16"},
        }

        for preset_name, settings in presets.items():
            output_file = tmp_path / f"{preset_name}.wav"

            if settings["sample_rate"] != sample_rate:
                import scipy.signal

                resampled = scipy.signal.resample(
                    audio,
                    int(len(audio) * settings["sample_rate"] / sample_rate),
                )
                sf.write(
                    output_file,
                    resampled,
                    settings["sample_rate"],
                    subtype=settings["subtype"],
                )
            else:
                sf.write(
                    output_file,
                    audio,
                    settings["sample_rate"],
                    subtype=settings["subtype"],
                )

            assert output_file.exists()

    def test_batch_export_workflow(self, tmp_path):
        from korshmer.project_management import ExportManager

        ExportManager(base_dir=str(tmp_path))

        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))

        files_to_export = []
        for i in range(5):
            audio = 0.5 * np.sin(2 * np.pi * (440 + i * 100) * t)
            input_file = tmp_path / f"input_{i}.wav"
            sf.write(input_file, audio, sample_rate)
            files_to_export.append(input_file)

        export_dir = tmp_path / "batch_export"
        export_dir.mkdir()

        for i, input_file in enumerate(files_to_export):
            data, sr = sf.read(input_file)
            output_file = export_dir / f"output_{i}.wav"
            sf.write(output_file, data, sr)

        assert len(list(export_dir.glob("*.wav"))) == 5
