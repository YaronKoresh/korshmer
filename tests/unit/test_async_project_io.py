import asyncio
import zipfile

import pytest

from korshmer.async_io.project_io import (
    read_presets,
    read_project,
    write_presets,
    write_project,
)
from korshmer.exceptions import ProjectLoadError


def _zip_with_metadata(path, text):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", text)
    return path


def test_read_project_round_trip(tmp_path):
    path = tmp_path / "song.kmr"
    asyncio.run(write_project(path, {"name": "song", "tempo": 120}))
    assert asyncio.run(read_project(path)) == {"name": "song", "tempo": 120}


@pytest.mark.parametrize("text", ["[1, 2]", "5", '"text"', "null"])
def test_read_project_rejects_non_object_metadata(tmp_path, text):
    path = _zip_with_metadata(tmp_path / "bad.kmr", text)
    with pytest.raises(ProjectLoadError):
        asyncio.run(read_project(path))


def test_read_presets_round_trip(tmp_path):
    path = tmp_path / "presets.json"
    asyncio.run(write_presets(path, {"a": {"x": 1}}))
    assert asyncio.run(read_presets(path)) == {"a": {"x": 1}}


@pytest.mark.parametrize("text", ["[1, 2]", "5", '"text"', "null", "{broken"])
def test_read_presets_returns_empty_for_non_object_file(tmp_path, text):
    path = tmp_path / "presets.json"
    path.write_text(text, encoding="utf-8")
    assert asyncio.run(read_presets(path)) == {}


def test_read_presets_returns_empty_for_missing_file(tmp_path):
    assert asyncio.run(read_presets(tmp_path / "none.json")) == {}
