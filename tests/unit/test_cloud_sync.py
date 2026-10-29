import threading

import pytest

from korshmer.project_management.cloud_sync import CloudSyncManager


@pytest.fixture
def env(tmp_path):
    base = tmp_path / "app"
    projects = base / "saved_data" / "projects"
    projects.mkdir(parents=True)
    remote = tmp_path / "Dropbox"
    return base, projects, remote


@pytest.fixture
def manager(env):
    base, _projects, remote = env
    m = CloudSyncManager(base)
    assert m.connect("folder", {"path": str(remote)})
    yield m
    m.disconnect()


def test_unknown_provider_is_refused(env):
    base, _p, remote = env
    m = CloudSyncManager(base)
    assert m.connect("dropbox", {"path": str(remote)}) is False
    assert m.get_sync_status()["enabled"] is False


@pytest.mark.parametrize("creds", [{}, {"path": ""}, {"path": "a\x00b"}])
def test_connect_rejects_bad_credentials(env, creds):
    m = CloudSyncManager(env[0])
    assert m.connect("folder", creds) is False
    assert m.enabled is False


def test_sync_copies_project_and_lists_it(env, manager):
    _b, projects, remote = env
    song = projects / "song.kproj"
    song.write_bytes(b"one")
    assert manager.sync_project(song) is True
    assert (remote / "song.kproj").read_bytes() == b"one"
    listing = manager.list_cloud_projects()
    assert [p["id"] for p in listing] == ["song.kproj"]
    assert listing[0]["size"] == 3
    status = manager.get_sync_status()
    assert status["last_sync"] is not None
    assert status["pending_uploads"] == 0
    assert status["conflicts"] == []


def test_local_change_is_uploaded_again(env, manager):
    _b, projects, remote = env
    song = projects / "song.kproj"
    song.write_bytes(b"one")
    manager.sync_project(song)
    song.write_bytes(b"two")
    assert manager.get_sync_status()["pending_uploads"] == 1
    assert manager.sync_project(song) is True
    assert (remote / "song.kproj").read_bytes() == b"two"


def test_newer_cloud_copy_is_not_overwritten(env, manager):
    _b, projects, remote = env
    song = projects / "song.kproj"
    song.write_bytes(b"one")
    manager.sync_project(song)
    (remote / "song.kproj").write_bytes(b"from other pc")
    assert manager.sync_project(song) is False
    assert (remote / "song.kproj").read_bytes() == b"from other pc"
    assert manager.get_sync_status()["conflicts"] == []


def test_both_sides_changed_is_a_conflict(env, manager):
    _b, projects, remote = env
    song = projects / "song.kproj"
    song.write_bytes(b"one")
    manager.sync_project(song)
    song.write_bytes(b"mine")
    (remote / "song.kproj").write_bytes(b"theirs")
    assert manager.sync_project(song) is False
    assert manager.get_sync_status()["conflicts"] == ["song.kproj"]
    assert (remote / "song.kproj").read_bytes() == b"theirs"
    assert song.read_bytes() == b"mine"


def test_resolve_conflict_keeps_chosen_side(env, manager):
    _b, projects, remote = env
    song = projects / "song.kproj"
    song.write_bytes(b"one")
    manager.sync_project(song)
    song.write_bytes(b"mine")
    (remote / "song.kproj").write_bytes(b"theirs")
    manager.sync_project(song)
    assert manager.resolve_conflict(song, "bogus") is False
    assert manager.resolve_conflict(song, "remote") is True
    assert song.read_bytes() == b"theirs"
    assert manager.get_sync_status()["conflicts"] == []

    song.write_bytes(b"mine again")
    (remote / "song.kproj").write_bytes(b"theirs again")
    manager.sync_project(song)
    assert manager.resolve_conflict(song, "local") is True
    assert (remote / "song.kproj").read_bytes() == b"mine again"


def test_existing_cloud_file_without_history_is_a_conflict(env, manager):
    _b, projects, remote = env
    remote.mkdir(exist_ok=True)
    (remote / "song.kproj").write_bytes(b"theirs")
    song = projects / "song.kproj"
    song.write_bytes(b"mine")
    assert manager.sync_project(song) is False
    assert manager.get_sync_status()["conflicts"] == ["song.kproj"]


def test_download_project(env, manager):
    _b, projects, remote = env
    (remote / "other.kproj").write_bytes(b"data")
    assert manager.get_sync_status()["pending_downloads"] == 1
    target = projects / "other.kproj"
    assert manager.download_project("other.kproj", target) is True
    assert target.read_bytes() == b"data"
    assert manager.get_sync_status()["pending_downloads"] == 0


@pytest.mark.parametrize(
    "cloud_id", ["../x.kproj", "a/b.kproj", "..", "x.txt", "", None, ".h.kproj"]
)
def test_download_rejects_unsafe_ids(env, manager, cloud_id):
    _b, projects, _r = env
    assert manager.download_project(cloud_id, projects / "x.kproj") is False
    assert not (projects / "x.kproj").exists()


def test_sync_rejects_non_project_and_missing_files(env, manager):
    _b, projects, _r = env
    note = projects / "note.txt"
    note.write_text("x")
    assert manager.sync_project(note) is False
    assert manager.sync_project(projects / "missing.kproj") is False
    assert manager.sync_project(projects) is False


def test_operations_fail_when_disconnected(env):
    m = CloudSyncManager(env[0])
    song = env[1] / "song.kproj"
    song.write_bytes(b"x")
    assert m.sync_project(song) is False
    assert m.download_project("song.kproj", song) is False
    assert m.list_cloud_projects() == []
    assert m.resolve_conflict(song, "local") is False
    assert m.enable_auto_sync(5) is False


def test_state_survives_a_new_manager(env, manager):
    base, projects, remote = env
    song = projects / "song.kproj"
    song.write_bytes(b"one")
    manager.sync_project(song)
    again = CloudSyncManager(base)
    assert again.connect("folder", {"path": str(remote)})
    assert again.get_sync_status()["pending_uploads"] == 0
    song.write_bytes(b"two")
    assert again.sync_project(song) is True


def test_corrupt_state_file_is_ignored(env):
    base, projects, remote = env
    sync_dir = base / "saved_data" / "cloud_sync"
    sync_dir.mkdir(parents=True)
    (sync_dir / "sync_state.json").write_text("[1, 2")
    m = CloudSyncManager(base)
    assert m.connect("folder", {"path": str(remote)})
    assert m.get_sync_status()["last_sync"] is None


def test_folder_inside_state_dir_is_refused(env):
    base = env[0]
    m = CloudSyncManager(base)
    inside = base / "saved_data" / "cloud_sync" / "x"
    assert m.connect("folder", {"path": str(inside)}) is False


def test_auto_sync_runs_and_stops(env, manager, monkeypatch):
    _b, projects, remote = env
    (projects / "song.kproj").write_bytes(b"one")
    called = threading.Event()
    original = manager.sync_all

    def tracked():
        result = original()
        called.set()
        return result

    monkeypatch.setattr(manager, "sync_all", tracked)
    monkeypatch.setattr(
        "korshmer.project_management.cloud_sync.MIN_AUTO_SYNC_MINUTES", 0
    )
    assert manager.enable_auto_sync(0.001) is True
    assert manager.auto_sync_active
    assert called.wait(5)
    assert (remote / "song.kproj").read_bytes() == b"one"
    assert manager.disable_auto_sync() is True
    assert not manager.auto_sync_active


@pytest.mark.parametrize("minutes", [0, -1, "5", None, True, float("nan")])
def test_auto_sync_rejects_bad_interval(manager, minutes):
    assert manager.enable_auto_sync(minutes) is False
    assert not manager.auto_sync_active
