import json
import sys
from pathlib import Path

import pytest

from korshmer.exceptions import InvalidFileError
from korshmer.services.file_management import FileManagementService
from korshmer.utils.audit_log import AUDIT_LOGGER_NAME
from korshmer.utils.filesystem import sanitize_path


class TestSanitizePath:
    def test_rejects_null_byte(self):
        with pytest.raises(InvalidFileError):
            sanitize_path("song\x00.wav")

    def test_rejects_control_characters(self):
        with pytest.raises(InvalidFileError):
            sanitize_path("/tmp/song\n.wav")
        with pytest.raises(InvalidFileError):
            sanitize_path("/tmp/song\x1b.wav")

    def test_accepts_non_ascii_path_unchanged(self):
        assert sanitize_path("/tmp/שיר.wav") == "/tmp/שיר.wav"

    def test_accepts_windows_path_with_colon_unchanged(self):
        assert sanitize_path("C:\\Music\\song.wav") == "C:\\Music\\song.wav"

    def test_preserves_path_type(self):
        p = Path("/tmp/song.wav")
        assert sanitize_path(p) is p
        assert isinstance(sanitize_path("/tmp/song.wav"), str)


class TestResolvePath:
    def test_resolve_path_raises_invalid_file_error_on_null(self):
        with pytest.raises(InvalidFileError):
            FileManagementService.resolve_path("song\x00.wav")

    def test_resolve_path_normal(self, tmp_path):
        f = tmp_path / "a.wav"
        assert FileManagementService.resolve_path(str(f)) == str(f.resolve())


class TestLoadFileList:
    def test_skips_unsafe_entries(self, tmp_path, monkeypatch):
        good = tmp_path / "good.wav"
        good.write_bytes(b"x")
        bad = str(tmp_path / "bad\x00.wav")
        list_file = tmp_path / "inputs.json"
        list_file.write_text(json.dumps([bad, str(good)]))
        monkeypatch.setattr(
            "korshmer.services.file_management.INPUT_LIST_FILE",
            str(list_file),
        )
        result = FileManagementService.load_file_list()
        assert result == [str(good)]


def _write_wav(path: Path):
    import struct
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<800h", *([0] * 800)))


def _create_when_the_platform_allows(create, path):
    try:
        create(path)
    except OSError:
        assert sys.platform == "win32"


def _denied_for_control_characters(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == AUDIT_LOGGER_NAME
        and 'reason="control_characters"' in r.getMessage()
    ]


class TestManagersRejectUnsafePaths:
    def test_is_safe_path(self):
        from korshmer.utils.filesystem import is_safe_path

        assert is_safe_path("/tmp/song.wav")
        assert not is_safe_path("/tmp/song\n.wav")
        assert not is_safe_path(Path("/tmp/song\x1b.wav"))

    def test_import_skips_control_character_name(self, tmp_path, caplog):
        from korshmer.project_management.import_manager import ImportManager

        bad = tmp_path / "evil\nname.wav"
        good = tmp_path / "good.wav"
        _create_when_the_platform_allows(_write_wav, bad)
        _write_wav(good)
        manager = ImportManager(tmp_path / "base")
        imported = manager.import_audio_files([bad, good])
        assert [p.name for p in imported] == ["good.wav"]
        assert _denied_for_control_characters(caplog)

    def test_import_directory_with_control_character_is_rejected(
        self, tmp_path, caplog
    ):
        from korshmer.project_management.import_manager import ImportManager

        def create(directory):
            directory.mkdir()
            _write_wav(directory / "a.wav")

        bad_dir = tmp_path / "dir\tname"
        _create_when_the_platform_allows(create, bad_dir)
        manager = ImportManager(tmp_path / "base")
        assert manager.batch_import_from_directory(bad_dir) == []
        assert _denied_for_control_characters(caplog)

    def test_export_queue_rejects_control_character_paths(
        self, tmp_path, caplog
    ):
        from korshmer.project_management.export_manager import ExportManager

        src = tmp_path / "src\nx.wav"
        ok_src = tmp_path / "src.wav"
        _create_when_the_platform_allows(_write_wav, src)
        _write_wav(ok_src)
        manager = ExportManager(tmp_path / "base")
        assert manager.add_to_queue(src) is False
        assert manager.add_to_queue(ok_src, tmp_path / "out\n.wav") is False
        assert manager.batch_export([ok_src], output_dir=tmp_path / "o\nd") == 0
        assert len(_denied_for_control_characters(caplog)) == 3

    def test_project_save_and_load_reject_control_character_path(
        self, tmp_path
    ):
        from korshmer.project_management.project_manager import ProjectManager

        manager = ProjectManager(tmp_path)
        bad = manager.projects_dir / "p\nq.kproj"
        assert manager.save_project(bad) is False
        assert not any(manager.projects_dir.glob("p*"))
        assert manager.load_project(bad) is None
