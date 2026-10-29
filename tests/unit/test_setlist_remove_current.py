import json

from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.setlist import Setlist


def _make(tmp_path, count=3):
    pm = ProjectManager(tmp_path)
    paths = []
    for i in range(count):
        p = pm.projects_dir / f"song{i}.json"
        p.write_text(json.dumps({"name": f"song{i}"}))
        paths.append(p)
    setlist = Setlist(pm)
    for p in paths:
        assert setlist.add(p)
    return setlist, paths


def test_next_after_removing_the_playing_item_plays_the_following_one(
    tmp_path,
):
    s, _ = _make(tmp_path)
    assert s.start()["name"] == "song0"
    assert s.remove(0) is True
    assert s.next()["name"] == "song1"
    assert s.current_item.path.endswith("song1.json")
    assert s.next()["name"] == "song2"


def test_removed_playing_item_is_no_longer_the_current_item(tmp_path):
    s, _ = _make(tmp_path)
    s.start()
    s.next()
    assert s.remove(1) is True
    assert s.current_item is None
    assert s.next()["name"] == "song2"


def test_previous_after_removing_the_playing_item_plays_the_one_before(
    tmp_path,
):
    s, _ = _make(tmp_path)
    s.start()
    s.next()
    s.remove(1)
    assert s.previous()["name"] == "song0"


def test_next_after_removing_the_last_playing_item_returns_nothing(tmp_path):
    s, _ = _make(tmp_path)
    s.start()
    s.next()
    s.next()
    assert s.remove(2) is True
    assert s.current_item is None
    assert s.next() is None
    assert s.previous()["name"] == "song1"


def test_removing_an_item_before_the_playing_one_keeps_it_current(tmp_path):
    s, _ = _make(tmp_path)
    s.start()
    s.next()
    assert s.remove(0) is True
    assert s.current_item.path.endswith("song1.json")
    assert s.next()["name"] == "song2"


def test_removing_an_item_after_the_playing_one_keeps_it_current(tmp_path):
    s, _ = _make(tmp_path)
    s.start()
    assert s.remove(2) is True
    assert s.current_item.path.endswith("song0.json")
    assert s.next()["name"] == "song1"


def test_removing_the_only_item_leaves_an_empty_setlist(tmp_path):
    s, _ = _make(tmp_path, count=1)
    s.start()
    assert s.remove(0) is True
    assert s.current_item is None
    assert s.next() is None
    assert s.previous() is None


def test_adding_after_removing_the_last_playing_item_plays_the_new_one(
    tmp_path,
):
    s, paths = _make(tmp_path)
    s.start()
    s.next()
    s.next()
    s.remove(2)
    extra = paths[0].parent / "song9.json"
    extra.write_text(json.dumps({"name": "song9"}))
    assert s.add(extra)
    assert s.next()["name"] == "song9"
