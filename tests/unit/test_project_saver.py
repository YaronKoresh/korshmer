import json
import zipfile

import pytest

from korshmer.project_management.project_saver import (
    save_compressed,
    save_uncompressed,
)


def test_failed_json_save_keeps_the_old_file(tmp_path):
    target = tmp_path / "song.json"
    save_uncompressed({"bpm": 120}, target)

    with pytest.raises(TypeError):
        save_uncompressed({"bpm": 100, "bad": object()}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"bpm": 120}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["song.json"]


def test_json_save_replaces_the_old_file(tmp_path):
    target = tmp_path / "song.json"
    save_uncompressed({"bpm": 120}, target)
    save_uncompressed({"bpm": 90}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"bpm": 90}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["song.json"]


def test_failed_archive_save_keeps_the_old_archive(tmp_path, monkeypatch):
    target = tmp_path / "song.kproj"
    asset = tmp_path / "loop.wav"
    asset.write_bytes(b"x" * 100)
    save_compressed({"bpm": 120}, target, tmp_path / "temp")
    before = target.read_bytes()

    real_write = zipfile.ZipFile.write

    def failing_write(self, filename, arcname=None, *args, **kwargs):
        if str(arcname).startswith("assets/"):
            raise OSError(28, "No space left on device")
        return real_write(self, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", failing_write)
    with pytest.raises(OSError):
        save_compressed(
            {"bpm": 100, "assets": [{"path": str(asset)}]},
            target,
            tmp_path / "temp",
        )

    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.glob("*.tmp")) == []


def test_archive_save_replaces_the_old_archive(tmp_path):
    target = tmp_path / "song.kproj"
    save_compressed({"bpm": 120}, target, tmp_path / "temp")
    save_compressed({"bpm": 90}, target, tmp_path / "temp")

    with zipfile.ZipFile(target) as zf:
        assert json.loads(zf.read("manifest.json"))["bpm"] == 90
    assert sorted(p.name for p in tmp_path.glob("*.tmp")) == []
