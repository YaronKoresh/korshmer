import json
import zipfile

import pytest

from korshmer.project_management import project_manager as manager_module
from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.project_saver import save_compressed


def manifest_of(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


@pytest.fixture
def manager(tmp_path):
    made = ProjectManager(tmp_path)
    made.start_song("Song")
    assert made.save_project()
    return made


def counting(monkeypatch, name):
    seen = []
    real = getattr(manager_module, name)

    def counted(*args, **kwargs):
        seen.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(manager_module, name, counted)
    return seen


@pytest.fixture
def writes(monkeypatch):
    return counting(monkeypatch, "save_compressed")


def set_tempo(manager, tempo):
    manager.current_project_data["metadata"]["bpm"] = tempo


def test_the_automatic_backup_holds_work_that_was_not_saved_yet(manager):
    set_tempo(manager, 141)
    manager.last_backup_time = 0
    assert manager.create_backup()
    (backup,) = manager.list_backups("Song")
    assert manifest_of(backup)["metadata"]["bpm"] == 141
    assert manifest_of(manager.current_project_path)["metadata"]["bpm"] == 120


def test_saving_an_unchanged_song_again_does_not_rewrite_the_file(
    manager, writes
):
    assert manager.save_project_incremental()
    assert manager.save_project_incremental()
    assert writes == []


def test_saving_after_a_change_rewrites_the_file_once(manager, writes):
    set_tempo(manager, 90)
    assert manager.save_project_incremental()
    assert manager.save_project_incremental()
    assert len(writes) == 1
    assert manifest_of(manager.current_project_path)["metadata"]["bpm"] == 90


def test_going_back_to_an_earlier_state_after_a_plain_save_is_written(
    manager, writes
):
    assert manager.save_project_incremental()
    set_tempo(manager, 90)
    assert manager.save_project()
    set_tempo(manager, 120)
    assert manager.save_project_incremental()
    assert len(writes) == 2
    assert manifest_of(manager.current_project_path)["metadata"]["bpm"] == 120


def test_a_song_that_was_just_opened_is_not_written_back_unchanged(
    manager, writes
):
    opened = ProjectManager(manager.base_dir)
    assert opened.load_project(manager.current_project_path) is not None
    assert opened.save_project_incremental()
    assert writes == []
    set_tempo(opened, 77)
    assert opened.save_project_incremental()
    assert len(writes) == 1


def test_an_unchanged_song_is_written_again_when_its_file_is_gone(
    manager, writes
):
    manager.current_project_path.unlink()
    assert manager.save_project_incremental()
    assert len(writes) == 1
    assert manager.current_project_path.exists()


def test_an_unchanged_song_is_still_written_to_a_new_place(
    manager, writes, tmp_path
):
    elsewhere = tmp_path / "saved_data" / "Copy" / "Copy.kproj"
    elsewhere.parent.mkdir(parents=True)
    assert manager.save_project_incremental(elsewhere)
    assert manager.save_project_incremental(elsewhere)
    assert len(writes) == 1
    assert elsewhere.exists()


def test_the_song_is_turned_into_text_once_to_tell_if_it_changed(
    manager, monkeypatch
):
    set_tempo(manager, 90)
    calls = counting(monkeypatch, "project_digest")
    assert manager.save_project_incremental()
    assert len(calls) == 1


def test_unchanged_work_is_not_written_to_the_recovery_file_again(
    manager, monkeypatch
):
    kept = counting(monkeypatch, "save_recovery_file")
    assert manager.save_recovery_state()
    assert manager.save_recovery_state()
    assert len(kept) == 1
    set_tempo(manager, 90)
    assert manager.save_recovery_state()
    assert len(kept) == 2


def test_recovery_work_is_written_again_after_the_file_was_cleared(
    manager, monkeypatch
):
    kept = counting(monkeypatch, "save_recovery_file")
    assert manager.save_recovery_state()
    assert manager.clear_recovery_state()
    assert manager.save_recovery_state()
    assert len(kept) == 2


def archive_with_assets(tmp_path, names):
    assets = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"audio " * 4000)
        assets.append({"path": str(path)})
    saved = save_compressed(
        {"name": "Song", "assets": assets},
        tmp_path / "Song.kproj",
        tmp_path / "temp",
    )
    with zipfile.ZipFile(saved) as archive:
        return {
            info.filename: info.compress_type for info in archive.infolist()
        }


def test_audio_that_is_already_compressed_is_stored_as_it_is(tmp_path):
    names = ["a.mp3", "b.ogg", "c.flac", "d.m4a", "e.OPUS", "take.wav"]
    kinds = archive_with_assets(tmp_path, names)
    for name in names[:-1]:
        assert kinds[f"assets/{name}"] == zipfile.ZIP_STORED
    assert kinds["assets/take.wav"] == zipfile.ZIP_DEFLATED
    assert kinds["manifest.json"] == zipfile.ZIP_DEFLATED
