import os

from korshmer.services.file_management import FileManagementService
from korshmer.services.preset_service import DEFAULT_PRESET, PresetService


class TestFileManagementService:
    def test_format_size_bytes(self):
        assert FileManagementService.format_size(500) == "500.00 B"

    def test_format_size_kb(self):
        assert FileManagementService.format_size(1024) == "1.00 KB"

    def test_format_size_mb(self):
        assert FileManagementService.format_size(1024 * 1024) == "1.00 MB"

    def test_format_size_gb(self):
        assert FileManagementService.format_size(1024**3) == "1.00 GB"

    def test_get_path_size_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        size = FileManagementService.get_path_size(str(f))
        assert size == len("hello world")

    def test_get_path_size_directory(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbbbb")
        size = FileManagementService.get_path_size(str(tmp_path))
        assert size == 8

    def test_get_path_size_nonexistent(self):
        assert FileManagementService.get_path_size("/nonexistent/path") == 0

    def test_resolve_path(self, tmp_path):
        result = FileManagementService.resolve_path(str(tmp_path / "test.txt"))
        assert os.path.isabs(result)

    def test_is_valid_audio(self):
        assert FileManagementService.is_valid_audio("song.mp3")
        assert FileManagementService.is_valid_audio("song.WAV")
        assert FileManagementService.is_valid_audio("song.flac")
        assert not FileManagementService.is_valid_audio("song.mid")
        assert not FileManagementService.is_valid_audio("song.txt")

    def test_is_valid_midi(self):
        assert FileManagementService.is_valid_midi("song.mid")
        assert FileManagementService.is_valid_midi("song.MID")
        assert not FileManagementService.is_valid_midi("song.wav")

    def test_is_valid_file(self):
        assert FileManagementService.is_valid_file("song.mp3")
        assert FileManagementService.is_valid_file("song.mid")
        assert not FileManagementService.is_valid_file("song.txt")

    def test_list_output_files(self, tmp_path):
        (tmp_path / "output.mid").write_text("")
        (tmp_path / "output.wav").write_text("")
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "_private").write_text("")
        (tmp_path / "temp_preview.wav").write_text("")
        (tmp_path / "logs").mkdir()

        results = FileManagementService.list_output_files(str(tmp_path))
        names = [name for name, _ in results]
        assert "output.mid" in names
        assert "output.wav" in names
        assert ".hidden" not in names
        assert "_private" not in names
        assert "temp_preview.wav" not in names

    def test_list_output_files_empty(self, tmp_path):
        results = FileManagementService.list_output_files(str(tmp_path))
        assert results == []

    def test_get_file_stats(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        stats = FileManagementService.get_file_stats(str(f))
        assert stats is not None
        assert stats["basename"] == "test.txt"
        assert stats["is_dir"] is False
        assert stats["size"] == 5
        assert "created" in stats
        assert "modified" in stats

    def test_get_file_stats_nonexistent(self):
        stats = FileManagementService.get_file_stats("/nonexistent")
        assert stats is None

    def test_find_source_for_comparison(self):
        sources = ["/audio/song1.wav", "/audio/song2.wav"]
        result = FileManagementService.find_source_for_comparison(
            "/output/song1_drums.mid", sources
        )
        assert result == "/audio/song1.wav"

    def test_find_source_prefers_the_cut_copy_of_its_own_session(
        self, tmp_path
    ):
        import soundfile as sf

        from korshmer.audio_processing.stem_session import StemSession

        out = tmp_path / "out"
        (out / "MIDI").mkdir(parents=True)
        sf.write(out / "music_only.wav", [0.0] * 10, 22050)
        midi = out / "MIDI" / "song_drums.mid"
        other = out / "MIDI" / "song_saved_by_hand.mid"
        StemSession(
            source_path="/audio/song.wav",
            output_dir=str(out),
            midi_files=[str(midi)],
            music_cut={
                "timeline": [[0.0, 1.0, 0.0]],
                "source": "music_only.wav",
            },
        ).save()
        sources = ["/audio/song.wav"]
        found = FileManagementService.find_source_for_comparison(
            str(midi), sources
        )
        assert found == str(out / "music_only.wav")
        found = FileManagementService.find_source_for_comparison(
            str(other), sources
        )
        assert found == "/audio/song.wav"

    def test_find_source_no_match(self):
        sources = ["/audio/song1.wav"]
        result = FileManagementService.find_source_for_comparison(
            "/output/unrelated.mid", sources
        )
        assert result is None

    def test_save_and_load_file_list(self, tmp_path, monkeypatch):
        list_file = tmp_path / "input_list.json"
        monkeypatch.setattr(
            "korshmer.services.file_management.INPUT_LIST_FILE", str(list_file)
        )
        test_file = tmp_path / "real.wav"
        test_file.write_text("")
        paths = [str(test_file), "/nonexistent/fake.wav"]
        FileManagementService.save_file_list(paths)
        loaded = FileManagementService.load_file_list()
        assert str(test_file) in loaded
        assert "/nonexistent/fake.wav" not in loaded


class TestPresetService:
    def test_load_all_defaults(self, monkeypatch):
        monkeypatch.setattr(
            "korshmer.services.preset_service.PRESETS_FILE", "/nonexistent"
        )
        presets = PresetService.load_all()
        assert "Default" in presets
        assert presets["Default"]["grid"] == "1/8"

    def test_save_and_load(self, tmp_path, monkeypatch):
        preset_file = tmp_path / "presets.json"
        monkeypatch.setattr(
            "korshmer.services.preset_service.PRESETS_FILE", str(preset_file)
        )
        presets = {"Custom": {"grid": "1/16", "quantize": 50}}
        PresetService.save_all(presets)
        loaded = PresetService.load_all()
        assert "Custom" in loaded
        assert loaded["Custom"]["grid"] == "1/16"

    def test_get_default(self):
        d = PresetService.get_default()
        assert d == DEFAULT_PRESET
        d["grid"] = "changed"
        assert DEFAULT_PRESET["grid"] == "1/8"

    def test_normalize_velocity_mode(self):
        assert PresetService.normalize_velocity_mode("Original") == "Original"
        assert PresetService.normalize_velocity_mode("Fixed") == "Fixed"
        assert (
            PresetService.normalize_velocity_mode("Compressed") == "Compressed"
        )
        assert PresetService.normalize_velocity_mode("unknown") == "Fixed"

    def test_build_preset_data(self):
        data = PresetService.build_preset_data(
            octave="Original",
            root="C",
            scale="Major",
            grid="1/4",
            quantize=80,
            swing=10,
            vel_mode="Fixed",
            vel_val=100,
            export_mode="Separate",
            cc=True,
            gm=False,
            wav32=True,
            stems=["drums", "bass"],
        )
        assert data["octave"] == "Original"
        assert data["root"] == "C"
        assert data["stems"] == ["drums", "bass"]
        assert data["gm"] is False
