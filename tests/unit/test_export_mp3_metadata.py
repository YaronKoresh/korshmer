import struct
from pathlib import Path

import pytest


def _make_minimal_mp3(path: Path) -> None:
    with open(path, "wb") as f:
        for _ in range(4):
            f.write(struct.pack(">I", 0xFFFB9000))
            f.write(b"\x00" * 413)


@pytest.fixture()
def tagless_mp3(tmp_path):
    p = tmp_path / "tagless.mp3"
    _make_minimal_mp3(p)
    return p


def test_embed_metadata_creates_tags_for_tagless_mp3(tagless_mp3):
    from korshmer.project_management.export_manager import ExportManager

    em = ExportManager.__new__(ExportManager)
    em._embed_metadata(
        tagless_mp3,
        {"title": "Test Song", "artist": "Test Artist"},
    )

    from mutagen.mp3 import MP3

    audio = MP3(str(tagless_mp3))
    assert audio.tags is not None


def test_embed_metadata_does_not_crash_on_tagless_mp3(tagless_mp3):
    from korshmer.project_management.export_manager import ExportManager

    em = ExportManager.__new__(ExportManager)
    em._embed_metadata(tagless_mp3, {"title": "Hello"})
