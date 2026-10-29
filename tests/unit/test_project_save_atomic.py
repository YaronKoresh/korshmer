from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from korshmer.async_io.project_io import write_presets, write_project


@pytest.mark.asyncio
async def test_write_project_atomic_no_window():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "project.kor"
        data = {"version": "1.0.0", "tracks": []}
        await write_project(path, data)
        assert path.exists()

        original_content = path.read_bytes()

        await write_project(path, {"version": "2.0.0", "tracks": [1]})
        assert path.exists()
        new_content = path.read_bytes()
        assert new_content != original_content


@pytest.mark.asyncio
async def test_write_presets_atomic_no_window():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "presets.json"
        data = {"eq": {"low": 0}}
        await write_presets(path, data)
        assert path.exists()

        await write_presets(path, {"eq": {"low": 5}})
        assert path.exists()


@pytest.mark.asyncio
async def test_write_project_uses_replace_not_unlink_rename():
    import inspect

    import korshmer.async_io.project_io as mod

    source = inspect.getsource(mod.write_project)
    assert "replace" in source or ".replace(" in source, (
        "write_project should use replace() for atomic overwrite"
    )
    assert "unlink" not in source.split("except")[0], (
        "write_project should not unlink the target before rename"
    )
