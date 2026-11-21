import pytest

from korshmer.project_management.cloud_sync import CloudSyncManager


@pytest.fixture
def base(tmp_path):
    (tmp_path / "app" / "saved_data").mkdir(parents=True)
    return tmp_path / "app"


@pytest.fixture
def manager(base, tmp_path):
    m = CloudSyncManager(base)
    assert m.connect("folder", {"path": str(tmp_path / "Shared")})
    yield m
    m.disconnect()


def save_song(base, name, content=b"song"):
    folder = base / "saved_data" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.kproj").write_bytes(content)
    return folder


def test_sync_all_uploads_songs_saved_in_their_own_folders(
    base, manager, tmp_path
):
    save_song(base, "Ballad", b"one")
    save_song(base, "Dance Track", b"two")
    assert manager.sync_all() == {
        "Ballad.kproj": True,
        "Dance Track.kproj": True,
    }
    assert (tmp_path / "Shared" / "Ballad.kproj").read_bytes() == b"one"
    assert (tmp_path / "Shared" / "Dance Track.kproj").read_bytes() == b"two"


def test_versions_and_backups_inside_a_song_folder_are_not_uploaded(
    base, manager
):
    folder = save_song(base, "Ballad")
    (folder / "Versions").mkdir()
    (folder / "Versions" / "v1.kproj").write_bytes(b"old")
    (folder / "Backups").mkdir()
    (folder / "Backups" / "Ballad_backup.kproj").write_bytes(b"old")
    assert list(manager.sync_all()) == ["Ballad.kproj"]


def test_songs_kept_in_the_old_projects_folder_are_still_uploaded(
    base, manager
):
    old = base / "saved_data" / "projects"
    old.mkdir()
    (old / "legacy.kproj").write_bytes(b"old")
    save_song(base, "Ballad")
    assert sorted(manager.sync_all()) == ["Ballad.kproj", "legacy.kproj"]


def test_a_saved_song_counts_as_waiting_for_upload(base, manager):
    save_song(base, "Ballad")
    assert manager.get_sync_status()["pending_uploads"] == 1
    manager.sync_all()
    assert manager.get_sync_status()["pending_uploads"] == 0


def test_folders_without_a_song_file_are_skipped_as_a_safeguard(base, manager):
    (base / "saved_data" / "logs").mkdir()
    (base / "saved_data" / "logs" / "korshmer.log").write_text("x")
    assert manager.sync_all() == {}


def test_an_unsupported_service_is_named_in_the_error(base, caplog):
    m = CloudSyncManager(base)
    with caplog.at_level("WARNING"):
        assert m.connect("dropbox", {"token": "x"}) is False
    status = m.get_sync_status()
    assert "dropbox" in str(status.get("last_error"))
    assert "folder" in str(status.get("last_error"))
    messages = " ".join(r.getMessage() for r in caplog.records).lower()
    assert "not yet implemented" not in messages
    assert "unsupported" in messages


def test_a_successful_connection_clears_the_error(base, tmp_path):
    m = CloudSyncManager(base)
    m.connect("dropbox", {})
    assert m.connect("folder", {"path": str(tmp_path / "Shared")}) is True
    status = m.get_sync_status()
    m.disconnect()
    assert status.get("last_error", "missing") is None
