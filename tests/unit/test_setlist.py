import json
import logging
import threading

from korshmer.network.scene_crossfade import CrossfadeMixer
from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.setlist import Setlist


def _make(tmp_path, count=3):
    pm = ProjectManager(tmp_path)
    paths = []
    for i in range(count):
        p = pm.projects_dir / f"song{i}.json"
        p.write_text(json.dumps({"name": f"song{i}"}))
        paths.append(p)
    return pm, paths


def test_next_twice_reaches_third(tmp_path):
    pm, paths = _make(tmp_path)
    s = Setlist(pm)
    for p in paths:
        assert s.add(p)
    s.start()
    s.next()
    s.next()
    assert s.current_item.path == str(paths[2])
    assert s.current_data["name"] == "song2"
    s.previous()
    assert s.current_data["name"] == "song1"


def test_missing_item_skipped_and_logged(tmp_path, caplog):
    pm, paths = _make(tmp_path)
    s = Setlist(pm)
    s.add(paths[0])
    s.add(pm.projects_dir / "gone.json")
    s.add(paths[2])
    s.start()
    with caplog.at_level(logging.ERROR):
        data = s.next()
    assert data["name"] == "song2"
    assert "missing" in caplog.text


def test_path_outside_saved_data_rejected(tmp_path):
    pm, _ = _make(tmp_path)
    outside = tmp_path / "evil.json"
    outside.write_text("{}")
    s = Setlist(pm)
    assert not s.add(outside)
    assert not s.add(pm.projects_dir / ".." / ".." / "evil.json")
    assert s.items == []


def test_save_load_roundtrip(tmp_path):
    pm, paths = _make(tmp_path)
    s = Setlist(pm, "gig")
    s.add(paths[0], 1000)
    s.add(paths[1])
    assert s.save()
    loaded = Setlist.load(pm, "gig")
    assert [i.to_dict() for i in loaded.items] == [i.to_dict() for i in s.items]


def test_preload_and_crossfade(tmp_path):
    pm, paths = _make(tmp_path, 2)
    mixer = CrossfadeMixer(initial=str(paths[0]))
    s = Setlist(pm, crossfader=mixer)
    for p in paths:
        s.add(p, 500)
    s.start()
    s.preload_next()
    assert s.next()["name"] == "song1"
    assert mixer.current == str(paths[1])
    assert mixer.transitioning


def test_corrupt_item_loads_backup_instead_of_skipping(tmp_path):
    pm, paths = _make(tmp_path)
    pm.load_project(paths[1])
    pm.auto_backup_interval = 0
    pm.last_backup_time = 0
    assert pm.create_backup() is True
    paths[1].write_text("{broken")
    s = Setlist(pm)
    s.add(paths[0])
    s.add(paths[1])
    s.start()
    data = s.next()
    assert data is not None
    assert data["name"] == "song1"


def test_preload_finished_after_remove_is_not_used_for_the_next_item(tmp_path):
    pm, paths = _make(tmp_path)
    started = threading.Event()
    release = threading.Event()
    real_load = pm.load_project_with_fallback

    def slow_load(path):
        if path == paths[1]:
            started.set()
            release.wait(5)
        return real_load(path)

    pm.load_project_with_fallback = slow_load
    s = Setlist(pm)
    for p in paths:
        s.add(p)
    s.start()
    s.preload_next()
    assert started.wait(5)
    s.remove(1)
    release.set()
    data = s.next()
    assert data["name"] == "song2"
    assert s.current_item.path == str(paths[2])
