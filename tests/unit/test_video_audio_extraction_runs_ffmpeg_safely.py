import logging
import subprocess
from types import SimpleNamespace

import pytest

from korshmer.project_management.import_manager import ImportManager


@pytest.fixture
def commands(monkeypatch):
    seen = []

    def fake_run(cmd, **options):
        seen.append((list(cmd), options))
        cmd[-1].write_bytes(b"RIFF" + b"\0" * 60)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def video_in(folder, name="clip.mp4"):
    video = folder / name
    video.write_bytes(b"\0" * 16)
    return video


def test_a_video_name_starting_with_a_dash_is_not_read_as_an_option(
    tmp_path, commands, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    video_in(tmp_path, "-version.mp4")
    ImportManager(tmp_path).extract_audio_from_video("-version.mp4")
    cmd, _ = commands[0]
    assert cmd[cmd.index("-i") + 1] == "file:" + str(tmp_path / "-version.mp4")
    assert not any(str(part).startswith("-version") for part in cmd)


def test_ffmpeg_cannot_wait_for_keyboard_input(tmp_path, commands):
    ImportManager(tmp_path).extract_audio_from_video(video_in(tmp_path))
    assert "-nostdin" in commands[0][0]


def test_ffmpeg_is_given_a_time_limit(tmp_path, commands):
    ImportManager(tmp_path).extract_audio_from_video(video_in(tmp_path))
    assert (commands[0][1].get("timeout") or 0) > 0


def test_an_extraction_that_takes_too_long_gives_up_cleanly(
    tmp_path, monkeypatch, caplog
):
    def slow(cmd, **options):
        cmd[-1].write_bytes(b"partial")
        raise subprocess.TimeoutExpired(cmd, options.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", slow)
    manager = ImportManager(tmp_path)
    with caplog.at_level(logging.ERROR):
        assert manager.extract_audio_from_video(video_in(tmp_path)) is None
    assert any("took too long" in r.getMessage() for r in caplog.records)
    assert list((manager.imports_dir / "extracted_audio").iterdir()) == []


def test_a_missing_ffmpeg_is_reported_clearly(tmp_path, monkeypatch, caplog):
    def missing(cmd, **options):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    with caplog.at_level(logging.ERROR):
        result = ImportManager(tmp_path).extract_audio_from_video(
            video_in(tmp_path)
        )
    assert result is None
    assert any(
        "FFmpeg is not installed" in r.getMessage() for r in caplog.records
    )
