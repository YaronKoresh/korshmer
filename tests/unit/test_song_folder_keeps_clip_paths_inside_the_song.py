import os
from pathlib import Path

import pytest

from korshmer.event_bus import Event, subscribe, unsubscribe
from korshmer.project_management.song_folder import (
    opened_editor,
    stored_editor,
)


def editor_with(path):
    return {
        "clips": [
            {
                "path": path,
                "source": [path, 0.0, 1.0],
                "tempo_base": [path, 120.0],
            }
        ]
    }


def opened_paths(path, song):
    clip = opened_editor(editor_with(path), song)["clips"][0]
    return [clip["path"], clip["source"][0], clip["tempo_base"][0]]


@pytest.fixture
def song(tmp_path):
    folder = tmp_path / "Song"
    folder.mkdir()
    return folder


@pytest.fixture
def status_lines():
    lines = []

    def remember(text="", **_other):
        lines.append(text)

    subscribe(Event.STATUS_UPDATE, remember)
    yield lines
    unsubscribe(Event.STATUS_UPDATE, remember)


def test_a_song_cannot_open_a_file_above_its_own_folder(tmp_path, song):
    private = tmp_path / "private.wav"
    private.write_bytes(b"x")
    for opened in opened_paths(os.path.join("..", "private.wav"), song):
        assert not os.path.isfile(opened)
        assert song.resolve() in Path(opened).resolve().parents


def test_a_refused_clip_is_still_in_the_song_after_saving_again(tmp_path, song):
    (tmp_path / "private.wav").write_bytes(b"x")
    crafted = editor_with(os.path.join("..", "private.wav"))
    saved_again = stored_editor(opened_editor(crafted, song), song)
    assert saved_again == crafted


def test_the_user_is_told_which_clip_was_not_opened(
    tmp_path, song, status_lines
):
    (tmp_path / "private.wav").write_bytes(b"x")
    opened_editor(editor_with(os.path.join("..", "private.wav")), song)
    assert any("private.wav" in line for line in status_lines)


def test_a_link_inside_the_song_cannot_lead_outside_it(tmp_path, song):
    outside = tmp_path / "Elsewhere"
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"x")
    try:
        os.symlink(outside, song / "Stems", target_is_directory=True)
    except OSError:
        pytest.skip("links cannot be created here")
    stored = os.path.join("Stems", "secret.wav")
    assert not os.path.isfile(opened_paths(stored, song)[0])
    assert stored_editor(
        opened_editor(editor_with(stored), song), song
    ) == editor_with(stored)


def test_links_that_point_at_each_other_do_not_stop_the_song_opening_as_a_safeguard(
    song,
):
    try:
        os.symlink(song / "b", song / "a")
        os.symlink(song / "a", song / "b")
    except OSError:
        pytest.skip("links cannot be created here")
    stored = os.path.join("a", "take.wav")
    assert not os.path.isfile(opened_paths(stored, song)[0])


def test_a_roundabout_path_inside_the_song_opens_as_a_safeguard(song):
    (song / "Stems").mkdir()
    wanted = song / "Stems" / "bass.wav"
    wanted.write_bytes(b"x")
    roundabout = os.path.join("Stems", "..", "Stems", "bass.wav")
    assert opened_paths(roundabout, song)[0] == str(wanted)


def test_a_clip_from_elsewhere_on_disk_keeps_its_full_path_as_a_safeguard(
    tmp_path, song
):
    outside = tmp_path / "Samples" / "kick.wav"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    assert opened_paths(str(outside), song)[0] == str(outside)
    assert stored_editor(editor_with(str(outside)), song) == editor_with(
        str(outside)
    )
