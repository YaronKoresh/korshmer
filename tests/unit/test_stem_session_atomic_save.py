from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

from korshmer.audio_processing.stem_session import StemSession


def test_stem_session_save_uses_atomic_replace():
    source = inspect.getsource(StemSession.save)
    assert "os.replace" in source, (
        "StemSession.save should use os.replace for atomic write"
    )
    assert ".tmp" in source


def test_stem_session_save_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        session = StemSession(
            source_path="/tmp/song.wav",
            output_dir=td,
            tempo=128.0,
            root="A",
            scale="Minor",
            stems={"vocals": "/tmp/vocals.wav"},
            midi_files=["/tmp/melody.mid"],
        )
        path = session.save()
        assert Path(path).exists()

        data = json.loads(Path(path).read_text())
        assert data["tempo"] == 128.0
        assert data["root"] == "A"
        assert data["stems"]["vocals"] == "/tmp/vocals.wav"

        tmp_path = Path(td) / "stems_session.json.tmp"
        assert not tmp_path.exists(), "Temp file should be cleaned up"
