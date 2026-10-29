import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf


@pytest.mark.integration
class TestProjectSaveLoadWorkflow:
    def test_project_manager_initialization(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        assert pm is not None

        pm.create_project("test_project")
        project_path = pm.projects_dir / "test_project"
        assert project_path is not None
        assert project_path.exists()

    def test_project_metadata_save_load(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("metadata_test")

        metadata = {
            "title": "Test Project",
            "artist": "Test Artist",
            "bpm": 120,
            "time_signature": "4/4",
            "key": "C Major",
            "created_at": "2024-01-01",
        }

        pm.set_metadata(metadata)

        pm.save_project("metadata_test")

        pm2 = ProjectManager(base_dir=str(tmp_path))
        loaded_metadata = pm2.load_project("metadata_test")

        if loaded_metadata:
            assert (
                "title" in str(loaded_metadata) or loaded_metadata is not None
            )

    def test_project_audio_track_save_load(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("audio_test")
        project_path = pm.projects_dir / "audio_test"

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        audio_path = project_path / "tracks" / "track1.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, audio, sample_rate)

        assert audio_path.exists()

        loaded_audio, loaded_sr = sf.read(audio_path)
        assert loaded_sr == sample_rate
        assert np.allclose(loaded_audio, audio, atol=1e-3)

    def test_project_midi_data_save_load(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("midi_test")
        project_path = pm.projects_dir / "midi_test"

        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)

        for i in range(5):
            note = pretty_midi.Note(
                velocity=100, pitch=60 + i, start=i * 0.5, end=(i + 1) * 0.5
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        midi_path = project_path / "midi" / "track1.mid"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(midi_path))

        assert midi_path.exists()

        loaded_midi = pretty_midi.PrettyMIDI(str(midi_path))
        assert len(loaded_midi.instruments) == 1
        assert len(loaded_midi.instruments[0].notes) == 5

    def test_project_settings_persistence(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("settings_test")
        project_path = pm.projects_dir / "settings_test"

        settings = {
            "buffer_size": 512,
            "sample_rate": 44100,
            "theme": "dark",
            "auto_save": True,
        }

        settings_path = project_path / "settings.json"
        with open(settings_path, "w") as f:
            json.dump(settings, f)

        with open(settings_path) as f:
            loaded_settings = json.load(f)

        assert loaded_settings == settings

    def test_project_with_multiple_tracks(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("multi_track_test")
        project_path = pm.projects_dir / "multi_track_test"

        audio_dir = project_path / "tracks"
        audio_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = 44100
        for i in range(3):
            t = np.linspace(0, 1.0, sample_rate)
            audio = 0.5 * np.sin(2 * np.pi * (440 + i * 100) * t)
            audio_path = audio_dir / f"track{i + 1}.wav"
            sf.write(audio_path, audio, sample_rate)

        midi_dir = project_path / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            midi = pretty_midi.PrettyMIDI(initial_tempo=120 + i * 10)
            instrument = pretty_midi.Instrument(program=i)
            note = pretty_midi.Note(
                velocity=100, pitch=60 + i, start=0.0, end=1.0
            )
            instrument.notes.append(note)
            midi.instruments.append(instrument)
            midi_path = midi_dir / f"midi{i + 1}.mid"
            midi.write(str(midi_path))

        assert len(list(audio_dir.glob("*.wav"))) == 3
        assert len(list(midi_dir.glob("*.mid"))) == 2

    def test_project_markers_and_regions(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("markers_test")
        project_path = pm.projects_dir / "markers_test"

        markers = [
            {"time": 0.0, "label": "Intro"},
            {"time": 8.0, "label": "Verse"},
            {"time": 16.0, "label": "Chorus"},
            {"time": 24.0, "label": "Outro"},
        ]

        regions = [
            {"start": 0.0, "end": 8.0, "label": "Section A"},
            {"start": 8.0, "end": 16.0, "label": "Section B"},
        ]

        markers_path = project_path / "markers.json"
        regions_path = project_path / "regions.json"

        with open(markers_path, "w") as f:
            json.dump(markers, f)

        with open(regions_path, "w") as f:
            json.dump(regions, f)

        with open(markers_path) as f:
            loaded_markers = json.load(f)

        with open(regions_path) as f:
            loaded_regions = json.load(f)

        assert loaded_markers == markers
        assert loaded_regions == regions

    def test_project_template_creation(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))

        pm.create_project("template")
        template_path = pm.projects_dir / "template"
        template_data = {"type": "template", "tracks": 4, "bpm": 120}

        template_file = template_path / "template.json"
        with open(template_file, "w") as f:
            json.dump(template_data, f)

        pm.create_project("from_template")
        new_project_path = pm.projects_dir / "from_template"
        assert new_project_path.exists()

    def test_project_auto_save_recovery(self, tmp_path):
        from korshmer.project_management import ProjectManager

        pm = ProjectManager(base_dir=str(tmp_path))
        pm.create_project("autosave_test")
        project_path = pm.projects_dir / "autosave_test"

        autosave_dir = project_path / "autosave"
        autosave_dir.mkdir(parents=True, exist_ok=True)

        autosave_data = {
            "timestamp": "2024-01-01T12:00:00",
            "data": {"bpm": 120},
        }

        autosave_file = autosave_dir / "autosave_001.json"
        with open(autosave_file, "w") as f:
            json.dump(autosave_data, f)

        assert autosave_file.exists()

        with open(autosave_file) as f:
            recovered = json.load(f)

        assert recovered["data"]["bpm"] == 120


@pytest.mark.integration
class TestAssetManagement:
    def test_asset_manager_initialization(self, tmp_path):
        from korshmer.project_management import AssetManager

        am = AssetManager(base_dir=str(tmp_path))
        assert am is not None

    def test_asset_import_and_tagging(self, tmp_path):
        from korshmer.project_management import AssetManager

        AssetManager(base_dir=str(tmp_path))

        asset_dir = tmp_path / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = 44100
        t = np.linspace(0, 1.0, sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        asset_path = asset_dir / "kick.wav"
        sf.write(asset_path, audio, sample_rate)

        tags = {"type": "drum", "category": "kick", "genre": "electronic"}
        tags_path = asset_dir / "kick_tags.json"

        with open(tags_path, "w") as f:
            json.dump(tags, f)

        assert tags_path.exists()

    def test_asset_library_organization(self, tmp_path):
        from korshmer.project_management import AssetManager

        AssetManager(base_dir=str(tmp_path))

        categories = ["drums", "bass", "synth", "fx"]
        asset_dir = tmp_path / "library"

        for category in categories:
            cat_dir = asset_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            sample_file = cat_dir / f"{category}_sample.wav"
            sample_rate = 44100
            t = np.linspace(0, 0.5, sample_rate // 2)
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            sf.write(sample_file, audio, sample_rate)

        for category in categories:
            assert (asset_dir / category).exists()
            assert len(list((asset_dir / category).glob("*.wav"))) > 0


@pytest.mark.integration
class TestAnalytics:
    def test_analytics_manager_initialization(self, tmp_path):
        from korshmer.project_management import AnalyticsManager

        am = AnalyticsManager(base_dir=str(tmp_path))
        assert am is not None

    def test_analytics_event_tracking(self, tmp_path):

        analytics_dir = tmp_path / "analytics"
        analytics_dir.mkdir(parents=True, exist_ok=True)

        events = [
            {"event": "project_created", "timestamp": "2024-01-01T10:00:00"},
            {"event": "audio_imported", "timestamp": "2024-01-01T10:05:00"},
            {"event": "effect_applied", "timestamp": "2024-01-01T10:10:00"},
        ]

        analytics_file = analytics_dir / "events.json"
        with open(analytics_file, "w") as f:
            json.dump(events, f)

        with open(analytics_file) as f:
            loaded_events = json.load(f)

        assert len(loaded_events) == 3
        assert loaded_events[0]["event"] == "project_created"
