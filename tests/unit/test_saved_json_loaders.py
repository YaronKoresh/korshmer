import pytest

from korshmer.services import file_management, preset_service
from korshmer.services.file_management import FileManagementService
from korshmer.services.preset_service import PresetService


@pytest.mark.parametrize(
    "content", [b"5", b'{"a": 1}', b"\xff\xfe", b"not json"]
)
def test_load_file_list_ignores_unusable_file(tmp_path, monkeypatch, content):
    path = tmp_path / "files.json"
    path.write_bytes(content)
    monkeypatch.setattr(file_management, "INPUT_LIST_FILE", str(path))

    assert FileManagementService.load_file_list() == []


@pytest.mark.parametrize("content", [b"5", b"[1]", b"\xff\xfe", b"not json"])
def test_load_presets_ignores_unusable_file(tmp_path, monkeypatch, content):
    path = tmp_path / "presets.json"
    path.write_bytes(content)
    monkeypatch.setattr(preset_service, "PRESETS_FILE", str(path))

    assert list(PresetService.load_all()) == [
        preset_service.RESERVED_PRESET_NAME
    ]


def test_load_presets_skips_entries_that_are_not_objects(tmp_path, monkeypatch):
    path = tmp_path / "presets.json"
    path.write_text('{"bad": 5, "good": {"grid": "1/4"}}', encoding="utf-8")
    monkeypatch.setattr(preset_service, "PRESETS_FILE", str(path))

    presets = PresetService.load_all()

    assert "bad" not in presets
    assert presets["good"] == {"grid": "1/4"}
