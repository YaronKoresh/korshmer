import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korshmer.exceptions import InvalidFileError
from korshmer.project_management import project_loader
from korshmer.project_management.project_loader import (
    file_hash,
    is_safe_archive_member,
    load_compressed,
    load_uncompressed,
)
from korshmer.project_management.project_saver import (
    save_compressed,
    save_uncompressed,
)


def _wav(path):
    sf.write(str(path), np.zeros(100, dtype="float32"), 44100)


def test_is_safe_archive_member(tmp_path):
    base = tmp_path.resolve()
    assert is_safe_archive_member("assets/a.wav", base) is True
    assert is_safe_archive_member("manifest.json", base) is True
    assert is_safe_archive_member("/etc/passwd", base) is False
    assert is_safe_archive_member("../evil.txt", base) is False


def test_uncompressed_roundtrip(tmp_path):
    data = {"name": "p", "tracks": [], "assets": []}
    path = save_uncompressed(data, tmp_path / "proj")
    assert path.suffix == ".json"
    assert load_uncompressed(path) == data


def test_compressed_roundtrip_with_asset(tmp_path):
    asset = tmp_path / "sound.wav"
    _wav(asset)
    data = {"name": "p", "tracks": [], "assets": [{"path": str(asset)}]}

    kproj = save_compressed(data, tmp_path / "proj", tmp_path / "temp")
    assert kproj.suffix == ".kproj"
    with zipfile.ZipFile(kproj) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "assets/sound.wav" in names

    assets_target = tmp_path / "assets_out"
    loaded = load_compressed(kproj, tmp_path / "temp", assets_target)
    assert loaded["name"] == "p"
    assert (assets_target / "sound.wav").exists()

    load_compressed(kproj, tmp_path / "temp", assets_target)
    copies = list(assets_target.glob("sound*.wav"))
    assert len(copies) == 1

    (assets_target / "sound.wav").write_bytes(b"different content entirely")
    load_compressed(kproj, tmp_path / "temp", assets_target)
    copies = sorted(p.name for p in assets_target.glob("sound*.wav"))
    assert "sound_1.wav" in copies


def test_path_traversal_member_not_extracted(tmp_path):
    kproj = tmp_path / "evil.kproj"
    with zipfile.ZipFile(kproj, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"name": "x", "assets": []}))
        zf.writestr("../evil.txt", "pwned")

    outside = tmp_path / "evil.txt"
    load_compressed(kproj, tmp_path / "temp", tmp_path / "assets_out")
    assert not outside.exists()


def test_file_hash_matches_for_identical_files(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert file_hash(a) == file_hash(b)


def _archive(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"name": "p"}))
        for name, size in members.items():
            zf.writestr(name, b"\0" * size)
    return path


def test_load_compressed_rejects_member_over_limit_before_extracting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        project_loader, "MAX_PROJECT_ARCHIVE_MEMBER_BYTES", 1000
    )
    kproj = _archive(tmp_path / "p.kproj", {"assets/big.bin": 1001})

    with pytest.raises(InvalidFileError):
        load_compressed(kproj, tmp_path / "temp", tmp_path / "assets_out")

    assert not (tmp_path / "assets_out").exists()


def test_load_compressed_rejects_total_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(project_loader, "MAX_PROJECT_ARCHIVE_TOTAL_BYTES", 1500)
    kproj = _archive(
        tmp_path / "p.kproj", {"assets/a.bin": 800, "assets/b.bin": 800}
    )

    with pytest.raises(InvalidFileError):
        load_compressed(kproj, tmp_path / "temp", tmp_path / "assets_out")


def test_load_compressed_rejects_oversized_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(project_loader, "MAX_PROJECT_MANIFEST_BYTES", 10)
    kproj = _archive(tmp_path / "p.kproj", {})

    with pytest.raises(InvalidFileError):
        load_compressed(kproj, tmp_path / "temp", tmp_path / "assets_out")


def test_load_compressed_accepts_archive_at_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        project_loader, "MAX_PROJECT_ARCHIVE_MEMBER_BYTES", 1000
    )
    kproj = _archive(tmp_path / "p.kproj", {"assets/ok.bin": 1000})

    data = load_compressed(kproj, tmp_path / "temp", tmp_path / "assets_out")

    assert data["name"] == "p"
    assert (tmp_path / "assets_out" / "ok.bin").stat().st_size == 1000


def _same_name_assets(tmp_path):
    for folder, payload in (("a", b"AAAA"), ("b", b"BBBB")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "kick.wav").write_bytes(payload)
    (tmp_path / "temp").mkdir()
    (tmp_path / "out").mkdir()
    return {
        "assets": [
            {"path": str(tmp_path / "a" / "kick.wav")},
            {"path": str(tmp_path / "b" / "kick.wav")},
        ]
    }


def test_save_compressed_keeps_assets_with_same_file_name(tmp_path):
    data = _same_name_assets(tmp_path)
    saved = save_compressed(data, tmp_path / "proj", tmp_path / "temp")
    with zipfile.ZipFile(saved) as zf:
        names = zf.namelist()
    assert len(names) == len(set(names))
    assert len([n for n in names if n.startswith("assets/")]) == 2
    assert "archive_name" not in data["assets"][0]


def test_load_compressed_restores_each_same_name_asset(tmp_path):
    data = _same_name_assets(tmp_path)
    saved = save_compressed(data, tmp_path / "proj", tmp_path / "temp")
    (tmp_path / "a" / "kick.wav").unlink()
    (tmp_path / "b" / "kick.wav").unlink()
    loaded = load_compressed(saved, tmp_path / "temp", tmp_path / "out")
    restored = [Path(a["path"]) for a in loaded["assets"]]
    assert restored[0] != restored[1]
    assert restored[0].read_bytes() == b"AAAA"
    assert restored[1].read_bytes() == b"BBBB"
    assert all("archive_name" not in a for a in loaded["assets"])
