import logging
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from korshmer.project_management.asset_manager import AssetManager
from korshmer.project_management.export_manager import ExportManager
from korshmer.project_management.file_io_log import LOGGER_NAME
from korshmer.project_management.import_manager import ImportManager


def _messages(caplog, level=None):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER_NAME and (level is None or r.levelno == level)
    ]


@pytest.fixture
def source_wav(tmp_path):
    path = tmp_path / "kick.wav"
    sf.write(path, np.zeros(4410), 44100)
    return path


def test_successful_export_logs_source_and_output(tmp_path, source_wav, caplog):
    manager = ExportManager(tmp_path)
    output = manager.exports_dir / "out.wav"
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    assert manager._export_file(source_wav, output, "Mastering", {})

    done = _messages(caplog, logging.INFO)
    assert any(
        m.startswith(f"Read export source done: {source_wav}") for m in done
    )
    assert any(
        m.startswith(f"Export audio done: {output}")
        and f"{output.stat().st_size} bytes" in m
        for m in done
    )
    assert f"Export audio started: {output}" in _messages(caplog, logging.DEBUG)


def test_export_with_missing_source_names_the_file(tmp_path, caplog):
    manager = ExportManager(tmp_path)
    missing = tmp_path / "missing.wav"
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    assert not manager._export_file(
        missing, manager.exports_dir / "o.wav", "Mastering", {}
    )

    errors = _messages(caplog, logging.ERROR)
    assert any(
        m.startswith(f"Read export source failed: {missing}") for m in errors
    )
    assert not any("Export audio" in m for m in _messages(caplog))


def test_export_write_error_names_the_output(tmp_path, source_wav, caplog):
    manager = ExportManager(tmp_path)
    output = tmp_path / "no_such_dir" / "out.wav"
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    assert not manager._export_file(source_wav, output, "Mastering", {})

    errors = _messages(caplog, logging.ERROR)
    assert any(m.startswith(f"Export audio failed: {output}") for m in errors)


def test_export_that_leaves_no_file_warns(
    tmp_path, source_wav, caplog, monkeypatch
):
    manager = ExportManager(tmp_path)
    monkeypatch.setattr(manager, "_convert_to_mp3", lambda *a, **k: None)
    output = manager.exports_dir / "out.mp3"
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    assert manager._export_file(source_wav, output, "MP3 High", {})

    assert f"Export audio left no file: {output}" in _messages(
        caplog, logging.WARNING
    )


def test_preset_save_and_load_are_logged(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    manager = ExportManager(tmp_path)
    manager.save_preset("Mine", {"format": "wav", "bit_depth": 24})
    ExportManager(tmp_path)

    done = _messages(caplog, logging.INFO)
    presets = manager.presets_file
    assert any(
        m.startswith(f"Save export presets done: {presets}") for m in done
    )
    assert any(
        m.startswith(f"Load export presets done: {presets}") for m in done
    )


def test_broken_presets_file_is_named_in_error(tmp_path, caplog):
    presets = tmp_path / "saved_data" / "export_presets.json"
    presets.parent.mkdir(parents=True)
    presets.write_text("{bad", encoding="utf-8")
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    manager = ExportManager(tmp_path)

    assert "CD Quality" in manager.export_presets
    errors = _messages(caplog, logging.ERROR)
    assert any(
        m.startswith(f"Load export presets failed: {presets}") for m in errors
    )


def test_chord_analysis_copy_is_logged(tmp_path, source_wav, caplog):
    (tmp_path / "kick_Chords.txt").write_text("C G Am F", encoding="utf-8")
    out_dir = tmp_path / "batch"
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    manager = ExportManager(tmp_path)
    assert manager.batch_export([source_wav], output_dir=out_dir) == 1

    dest = out_dir / "kick_Chords.txt"
    assert any(
        m.startswith(f"Copy chord analysis done: {dest}")
        for m in _messages(caplog, logging.INFO)
    )


def test_import_logs_target_and_size(tmp_path, source_wav, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    imported = ImportManager(tmp_path).import_audio_files([source_wav])

    assert len(imported) == 1
    target = imported[0]
    assert any(
        m.startswith(f"Import file done: {target}")
        and f"{target.stat().st_size} bytes" in m
        for m in _messages(caplog, logging.INFO)
    )


def test_import_copy_error_names_target(
    tmp_path, source_wav, caplog, monkeypatch
):
    import shutil

    def broken_copy(src, dst):
        raise PermissionError("denied")

    monkeypatch.setattr(shutil, "copy2", broken_copy)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    assert ImportManager(tmp_path).import_audio_files([source_wav]) == []
    assert any(
        m.startswith("Import file failed: ") and "PermissionError" in m
        for m in _messages(caplog, logging.ERROR)
    )


def test_video_extraction_logs_result(tmp_path, caplog, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\0" * 16)
    manager = ImportManager(tmp_path)

    def fake_ffmpeg(cmd, capture_output, text):
        output = cmd[-1]
        output.write_bytes(b"RIFF" + b"\0" * 60)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_ffmpeg)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    output = manager.extract_audio_from_video(video)

    assert output is not None
    assert f"Extract audio done: {output} (64 bytes)" in _messages(
        caplog, logging.INFO
    )


def test_failed_video_extraction_names_output(tmp_path, caplog, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\0" * 16)
    manager = ImportManager(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, capture_output, text: SimpleNamespace(
            returncode=1, stderr="bad input"
        ),
    )
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    assert manager.extract_audio_from_video(video) is None
    errors = _messages(caplog, logging.ERROR)
    assert any(
        m.startswith("Extract audio failed: ") and "exit code 1" in m
        for m in errors
    )


def test_asset_add_and_remove_are_logged(tmp_path, source_wav, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    manager = AssetManager(tmp_path)

    asset_id = manager.add_asset(source_wav)
    assert asset_id is not None
    stored = manager.asset_library[asset_id]["path"]
    assert manager.remove_asset(asset_id, delete_file=True)

    done = _messages(caplog, logging.INFO)
    assert any(m.startswith(f"Copy asset done: {stored}") for m in done)
    assert any(m.startswith(f"Delete asset done: {stored}") for m in done)
    assert any(
        m.startswith(f"Save asset library done: {manager.library_file}")
        for m in done
    )
    assert any(
        m.startswith(f"Save favorites done: {manager.favorites_file}")
        for m in done
    )


def test_asset_library_reload_is_logged(tmp_path, source_wav, caplog):
    first = AssetManager(tmp_path)
    first.add_to_favorites(first.add_asset(source_wav))
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    AssetManager(tmp_path)

    done = _messages(caplog, logging.INFO)
    assert any(
        m.startswith(f"Load asset library done: {first.library_file}")
        for m in done
    )
    assert any(
        m.startswith(f"Load favorites done: {first.favorites_file}")
        for m in done
    )


def test_corrupt_asset_library_is_named_in_error(tmp_path, caplog):
    library = tmp_path / "saved_data" / "asset_library.json"
    library.parent.mkdir(parents=True)
    library.write_text("{bad", encoding="utf-8")
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    manager = AssetManager(tmp_path)

    assert manager.asset_library == {}
    assert any(
        m.startswith(f"Load asset library failed: {library}")
        and "JSONDecodeError" in m
        for m in _messages(caplog, logging.ERROR)
    )
