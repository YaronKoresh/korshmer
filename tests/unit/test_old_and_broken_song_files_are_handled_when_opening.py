import copy
import json
import zipfile

import pytest
from hypothesis import given, settings, strategies as st

from korshmer.exceptions import ProjectCorruptedError, ProjectVersionError
from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.project_migration import (
    PROJECT_FORMAT_VERSION,
    opened_project,
)
from korshmer.ui.viewmodels.editor_vm import EditorModel


def song_file(tmp_path, manifest, name="Old"):
    folder = tmp_path / "saved_data" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.kproj"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    return path


def old_song():
    return {
        "name": "Old",
        "version": "1.0.0",
        "editor": {
            "tempo": 100,
            "tracks": [{"id": 1, "name": "Keys", "program": 4}],
            "notes": [{"id": 1, "track_id": 1, "pitch": 60, "start": 0.0}],
            "clips": [{"id": 1, "path": "take.wav", "duration": 1.0}],
        },
    }


def editor_of_a_real_song():
    model = EditorModel()
    track = model.add_track("Keys", program=4)
    model.add_note(track.id, 60, 0.0, 0.5)
    model.add_clip("take.wav", duration=1.0)
    saved = model.to_dict()
    for entry in saved["tracks"] + saved["clips"]:
        entry.pop("pan", None)
    return saved


def blank_song(tmp_path):
    manager = ProjectManager(tmp_path / "blank")
    made = manager.create_project("Demo")
    made.pop("format_version", None)
    return made


def shapes_older_versions_wrote(tmp_path):
    blank = blank_song(tmp_path)
    return [
        {},
        {"bpm": 120},
        {"name": "p"},
        {"name": "x", "assets": []},
        {"name": "Song"},
        {"metadata": {"author": "me", "bpm": 98}},
        blank,
        dict(blank, marker="value"),
        dict(blank, snapshot_info={"timestamp": "2020-01-01T00:00:00"}),
        {
            k: v
            for k, v in dict(blank, is_template=True).items()
            if k != "created"
        },
        dict(blank, editor=editor_of_a_real_song(), session_dir=""),
        dict(blank, editor={"tracks": [{"id": "1", "gain": "loud"}]}),
        dict(blank, editor=None, metadata=None, settings=None),
        dict(blank, audio_processing={"effects_chain": []}),
        old_song(),
    ]


def test_a_saved_song_records_its_file_format(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.start_song("New")
    assert manager.save_project()
    with zipfile.ZipFile(manager.current_project_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["format_version"] == PROJECT_FORMAT_VERSION


def test_a_song_saved_as_plain_json_records_its_file_format(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.start_song("New")
    assert manager.save_project(compress=False)
    manifest = json.loads(manager.current_project_path.read_text("utf-8"))
    assert manifest["format_version"] == PROJECT_FORMAT_VERSION


def test_an_old_song_is_brought_up_to_the_current_format(tmp_path):
    manager = ProjectManager(tmp_path)
    data = manager.load_project(song_file(tmp_path, old_song()))
    assert data["format_version"] == PROJECT_FORMAT_VERSION
    assert data["editor"]["tracks"][0]["pan"] == 0.0
    assert data["editor"]["clips"][0]["pan"] == 0.0


def test_an_old_song_gets_the_sections_newer_songs_have(tmp_path):
    manager = ProjectManager(tmp_path)
    data = manager.load_project(song_file(tmp_path, old_song()))
    assert data["metadata"] == {}
    assert data["tracks"] == [] and data["markers"] == []
    assert data["assets"] == [] and data["settings"] == {}


def test_every_kind_of_older_song_still_opens(tmp_path):
    manager = ProjectManager(tmp_path)
    for number, shape in enumerate(shapes_older_versions_wrote(tmp_path)):
        path = song_file(tmp_path, shape, name=f"Song {number}")
        data = manager.load_project(path)
        assert data is not None, (shape, manager.last_load_error)
        assert data["format_version"] == PROJECT_FORMAT_VERSION
        if isinstance(data.get("editor"), dict):
            EditorModel.from_dict(data["editor"])


def test_bringing_an_old_song_up_to_date_twice_changes_nothing_more(tmp_path):
    for shape in shapes_older_versions_wrote(tmp_path):
        once = opened_project(copy.deepcopy(shape), "Song")
        twice = opened_project(copy.deepcopy(once), "Song")
        assert twice == once


plain_values = st.one_of(
    st.none(), st.booleans(), st.integers(), st.text(max_size=5)
)
track_entries = st.fixed_dictionaries(
    {"id": st.integers(1, 50)},
    optional={
        "name": st.text(max_size=5),
        "program": st.integers(0, 127),
        "gain": st.floats(0, 2),
        "muted": st.booleans(),
    },
)
old_songs = st.fixed_dictionaries(
    {},
    optional={
        "name": st.text(max_size=8),
        "metadata": st.dictionaries(st.text(max_size=4), plain_values),
        "tracks": st.lists(plain_values, max_size=3),
        "settings": st.dictionaries(st.text(max_size=4), plain_values),
        "editor": st.fixed_dictionaries(
            {}, optional={"tracks": st.lists(track_entries, max_size=3)}
        ),
        "extra": plain_values,
    },
)


@settings(max_examples=60, deadline=None)
@given(old_songs)
def test_any_well_formed_old_song_opens_and_stays_the_same_when_reopened(song):
    once = opened_project(copy.deepcopy(song), "Song")
    assert once["format_version"] == PROJECT_FORMAT_VERSION
    assert opened_project(copy.deepcopy(once), "Song") == once
    for key, value in song.items():
        if key != "editor":
            assert once[key] == value


def test_a_song_from_a_newer_version_is_refused_with_a_clear_reason(tmp_path):
    manager = ProjectManager(tmp_path)
    newer = dict(old_song(), format_version=PROJECT_FORMAT_VERSION + 1)
    assert manager.load_project(song_file(tmp_path, newer)) is None
    assert "newer version" in manager.last_load_error


def test_opening_a_newer_song_raises_the_project_version_error():
    with pytest.raises(ProjectVersionError, match="newer version"):
        opened_project({"name": "x", "format_version": 99}, "x")


def test_a_refused_song_leaves_the_open_song_alone(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.start_song("Current")
    open_path = manager.current_project_path
    manager.load_project(song_file(tmp_path, {"name": "x", "tracks": 5}))
    assert "tracks" in manager.last_load_error
    assert manager.current_project_path == open_path
    assert manager.current_project_data["name"] == "Current"


def test_a_newer_song_is_not_replaced_by_an_older_backup(tmp_path):
    manager = ProjectManager(tmp_path)
    newer = dict(old_song(), format_version=PROJECT_FORMAT_VERSION + 1)
    song = song_file(tmp_path, newer)
    backups = song.parent / "Backups"
    backups.mkdir()
    backup = backups / "Old_backup_20200101_000000.kproj"
    with zipfile.ZipFile(backup, "w") as archive:
        archive.writestr("manifest.json", json.dumps(old_song()))
    assert manager.load_project_with_fallback(song) is None
    assert manager.last_load_backup is None
    assert "newer version" in manager.last_load_error


@pytest.mark.parametrize(
    ("manifest", "named"),
    [
        (["not", "a", "song"], "object"),
        ({"name": 7}, "name"),
        ({"name": "x", "tracks": "none"}, "tracks"),
        ({"name": "x", "metadata": []}, "metadata"),
        ({"name": "x", "editor": []}, "editor"),
        ({"name": "x", "editor": {"tracks": {}}}, "tracks"),
        ({"name": "x", "editor": {"tracks": ["Keys"]}}, "track 1"),
        ({"name": "x", "editor": {"tracks": [{"id": "one"}]}}, "id"),
        ({"name": "x", "editor": {"notes": [{"track_id": None}]}}, "track_id"),
        ({"name": "x", "editor": {"clips": [{"start": "soon"}]}}, "start"),
        ({"name": "x", "format_version": "two"}, "format version"),
    ],
)
def test_a_malformed_song_is_refused_and_the_reason_names_the_part(
    tmp_path, manifest, named
):
    manager = ProjectManager(tmp_path)
    assert manager.load_project(song_file(tmp_path, manifest)) is None
    assert "damaged" in manager.last_load_error
    assert named in manager.last_load_error


def test_a_current_song_missing_a_section_is_refused(tmp_path):
    manager = ProjectManager(tmp_path)
    manifest = {"name": "x", "format_version": PROJECT_FORMAT_VERSION}
    assert manager.load_project(song_file(tmp_path, manifest)) is None
    assert "metadata" in manager.last_load_error


def test_a_malformed_song_raises_the_corrupted_project_error():
    with pytest.raises(ProjectCorruptedError, match="tracks"):
        opened_project({"name": "x", "tracks": 3}, "x")


def test_a_good_open_clears_the_earlier_reason(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.load_project(song_file(tmp_path, ["broken"], name="Bad"))
    assert "object" in manager.last_load_error
    assert manager.load_project(song_file(tmp_path, old_song())) is not None
    assert manager.last_load_error == ""


def test_a_song_saved_now_opens_again_with_its_pan(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.start_song("Round")
    model = EditorModel()
    track = model.add_track("Keys")
    model.set_track_pan(track.id, -0.5)
    manager.current_project_data["editor"] = model.to_dict()
    assert manager.save_project()
    data = ProjectManager(tmp_path).load_project(manager.current_project_path)
    assert data["editor"]["tracks"][0]["pan"] == -0.5


def test_recovered_work_from_an_old_version_is_brought_up_to_date(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.current_project_data = old_song()
    assert manager.save_recovery_state()
    recovered = ProjectManager(tmp_path).enable_auto_recovery()
    project = recovered["project_data"]
    assert project["editor"]["tracks"][0]["pan"] == 0.0
    assert project["format_version"] == PROJECT_FORMAT_VERSION


def test_recovered_work_this_version_cannot_read_is_set_aside_once(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.current_project_data = dict(
        old_song(), format_version=PROJECT_FORMAT_VERSION + 1
    )
    assert manager.save_recovery_state()
    recovery = tmp_path / "saved_data" / ".recovery.gz"

    starting = ProjectManager(tmp_path)
    assert starting.enable_auto_recovery() is None
    assert "newer version" in starting.last_load_error
    assert not recovery.exists()
    assert recovery.with_name(".recovery.gz.refused").exists()

    again = ProjectManager(tmp_path)
    assert again.enable_auto_recovery() is None
    assert again.last_load_error == ""
