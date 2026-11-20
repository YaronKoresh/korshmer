import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QMessageBox

from korshmer.audio_processing import clip_level, clip_polish
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets import titlebar
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100
BEAT_SECONDS = 0.5


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


class _Studio(ProjectManagementActionsMixin, EditorActionsMixin):
    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.arranger.btn_snap.setChecked(False)
        self.current_session = None


def _tone(path, seconds=4, pitch=330.0, level=0.3):
    time = np.arange(int(RATE * seconds)) / RATE
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        (level * np.sin(2 * np.pi * pitch * time)).astype(np.float32),
        RATE,
    )
    return str(path)


def _studio_with_song(tmp_path):
    studio = _Studio(tmp_path / "app")
    track = studio.arranger.model.add_track("Lead")
    for step in range(4):
        studio.arranger.model.add_note(
            track.id, 60 + step, 2.0 + step * 0.5, 0.5, 90
        )
    studio.arranger.piano_roll.load_model()
    return studio


def _song_file(studio, name="My song"):
    return studio.project_manager.song_path_for(name)


def _files_in(folder):
    return sorted(
        str(path.relative_to(folder)).replace(os.sep, "/")
        for path in Path(folder).rglob("*")
        if path.is_file()
    )


def test_changing_the_tempo_keeps_audio_clips_on_the_same_beat_as_the_notes(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    timeline = studio.arranger
    clip = timeline.add_audio_clip(_tone(tmp_path / "take.wav"), 2.0)
    timeline.piano_roll.set_tempo(120.0)

    timeline.piano_roll.set_tempo(150.0)

    first_note = min(note.start for note in timeline.model.notes)
    assert first_note == pytest.approx(1.6)
    assert clip.start == pytest.approx(1.6)
    shown = timeline.piano_roll._clip_items[clip.id]
    assert shown.x() / shown.pixels_per_second == pytest.approx(1.6, abs=0.01)


def test_changing_the_tempo_never_changes_the_original_recording(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    timeline = studio.arranger
    recording = _tone(tmp_path / "take.wav")
    before = Path(recording).read_bytes()
    timeline.add_audio_clip(recording, 2.0)

    timeline.piano_roll.set_tempo(150.0)
    timeline.stretch_clips_to_tempo()
    timeline.wait_for_clip_edits()
    QApplication.processEvents()

    assert Path(recording).read_bytes() == before


def test_stepping_back_after_a_tempo_change_keeps_notes_and_clips_together(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    timeline = studio.arranger
    clip = timeline.add_audio_clip(_tone(tmp_path / "take.wav"), 2.0)
    timeline.model.snapshot()
    timeline.model.clip_by_id(clip.id).start = 3.0
    timeline.piano_roll.load_model()

    timeline.piano_roll.set_tempo(150.0)
    timeline.undo()

    restored = timeline.model.clip_by_id(clip.id)
    first_note = min(note.start for note in timeline.model.notes)
    assert first_note == pytest.approx(1.6)
    assert restored.start == pytest.approx(1.6)


def test_returning_to_the_first_tempo_restores_every_position(qapp, tmp_path):
    studio = _studio_with_song(tmp_path)
    timeline = studio.arranger
    clip = timeline.add_audio_clip(_tone(tmp_path / "take.wav"), 2.5)

    timeline.piano_roll.set_tempo(93.0)
    timeline.piano_roll.set_tempo(120.0)

    assert clip.start == pytest.approx(2.5)
    assert sorted(note.start for note in timeline.model.notes) == pytest.approx(
        [2.0, 2.5, 3.0, 3.5]
    )


def test_saving_a_song_copies_its_audio_clips_into_the_song_folder(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    outside = _tone(tmp_path / "desktop" / "take.wav")
    studio.arranger.add_audio_clip(outside, 2.0)
    song = _song_file(studio)

    assert studio._save_song_to(song)

    assert _files_in(song.parent) == ["Clips/take.wav", "My song.kproj"]
    assert os.path.isfile(outside)
    kept = studio.arranger.model.clips[0].path
    assert Path(kept).parent == song.parent / "Clips"


def test_a_saved_song_reopens_with_every_clip_in_place_after_the_originals_are_gone(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    timeline = studio.arranger
    clip = timeline.add_audio_clip(
        _tone(tmp_path / "desktop" / "take.wav"), 2.0, gain=1.4
    )
    timeline.model.trim_clip(clip.id, 3.0, 5.0)
    timeline.model.set_clip_mute(clip.id, True)
    timeline.piano_roll.load_model()
    timeline.piano_roll.set_tempo(150.0)
    song = _song_file(studio)
    studio._save_song_to(song)
    saved = timeline.to_project_dict()
    shutil.rmtree(tmp_path / "desktop")

    later = _Studio(tmp_path / "app")
    with mock.patch.object(QMessageBox, "warning") as warned:
        assert later.open_song_file(song)

    reopened = later.arranger.model.clips[0]
    assert not warned.called
    assert os.path.isfile(reopened.path)
    assert (
        reopened.start,
        reopened.offset,
        reopened.duration,
    ) == pytest.approx((2.4, 1.0, 2.0))
    assert (reopened.gain, reopened.muted) == (1.4, True)
    assert later.arranger.to_project_dict() == saved


def test_a_song_folder_can_be_moved_and_still_opens_with_its_clips(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    studio.arranger.add_audio_clip(
        _tone(tmp_path / "desktop" / "take.wav"), 2.0
    )
    song = _song_file(studio)
    studio._save_song_to(song)
    moved = song.parent.with_name("Renamed folder")
    shutil.move(str(song.parent), str(moved))
    shutil.rmtree(tmp_path / "desktop")

    later = _Studio(tmp_path / "app")
    with mock.patch.object(QMessageBox, "warning") as warned:
        assert later.open_song_file(moved / song.name)

    assert not warned.called
    assert (
        Path(later.arranger.model.clips[0].path) == moved / "Clips" / "take.wav"
    )


def test_two_different_recordings_with_the_same_name_are_both_kept(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    studio.arranger.add_audio_clip(
        _tone(tmp_path / "monday" / "take.wav", pitch=220), 0.0
    )
    studio.arranger.add_audio_clip(
        _tone(tmp_path / "tuesday" / "take.wav", pitch=440), 5.0
    )
    studio.arranger.add_audio_clip(
        _tone(tmp_path / "monday" / "take.wav", pitch=220), 9.0
    )
    song = _song_file(studio)

    studio._save_song_to(song)

    assert _files_in(song.parent / "Clips") == ["take.wav", "take_2.wav"]
    first, second, third = (clip.path for clip in studio.arranger.model.clips)
    assert first == third != second


def test_saving_twice_does_not_copy_the_clips_again(qapp, tmp_path):
    studio = _studio_with_song(tmp_path)
    studio.arranger.add_audio_clip(
        _tone(tmp_path / "desktop" / "take.wav"), 2.0
    )
    song = _song_file(studio)

    studio._save_song_to(song)
    studio._save_song_to(song)

    assert _files_in(song.parent / "Clips") == ["take.wav"]


def test_a_song_saved_beside_other_songs_gets_a_folder_of_its_own(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    loose = studio.project_manager.saved_data_dir / "Ballad.kproj"

    assert studio._save_song_to(loose)

    home = studio.project_manager.saved_data_dir / "Ballad"
    assert studio.project_manager.current_project_path == home / "Ballad.kproj"
    assert not loose.exists()
    second = home / "Remix.kproj"
    assert studio._save_song_to(second)
    assert (
        studio.project_manager.current_project_path
        == home / "Remix" / "Remix.kproj"
    )


def test_versions_and_backups_of_a_song_are_kept_inside_its_folder(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    song = _song_file(studio)
    studio._save_song_to(song)
    manager = studio.project_manager

    assert manager.create_snapshot("before the chorus")
    manager.last_backup_time = 0
    assert manager.create_backup()

    files = _files_in(song.parent)
    assert any(name.startswith("Versions/My song_snapshot_") for name in files)
    assert any(name.startswith("Backups/My song_backup_") for name in files)
    assert (
        manager.list_snapshots("My song")[0].parent == song.parent / "Versions"
    )
    assert not (manager.saved_data_dir / "snapshots").exists()
    assert not (manager.saved_data_dir / "backups").exists()


def test_a_version_from_another_song_cannot_be_restored_into_this_one(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    studio._save_song_to(_song_file(studio, "First"))
    studio.project_manager.create_snapshot()
    other_version = studio.project_manager.list_snapshots()[0]
    studio._save_song_to(_song_file(studio, "Second"))

    assert not studio.project_manager.restore_snapshot(other_version)


def test_recordings_and_exports_of_a_saved_song_default_to_its_folder(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    song = _song_file(studio)
    studio._save_song_to(song)
    names = studio.arranger.vm

    assert Path(names.recording_filename()).parent == song.parent / "Recordings"
    assert Path(names.export_midi_filename()).parent == song.parent / "Exports"
    assert Path(studio._song_exports_folder()) == song.parent / "Exports"


def test_a_clip_whose_audio_is_missing_is_named_when_the_song_opens(
    qapp, tmp_path
):
    studio = _studio_with_song(tmp_path)
    studio.arranger.add_audio_clip(
        _tone(tmp_path / "desktop" / "take.wav"), 2.0, name="Verse"
    )
    song = _song_file(studio)
    studio._save_song_to(song)
    (song.parent / "Clips" / "take.wav").unlink()

    later = _Studio(tmp_path / "app")
    with mock.patch.object(QMessageBox, "warning") as warned:
        later.open_song_file(song)

    assert "Verse" in warned.call_args.args[2]


def test_the_file_menu_no_longer_offers_to_update_an_old_song_file():
    labels = [
        entry[0]
        for entries in vars(titlebar).values()
        if isinstance(entries, (list, tuple))
        for entry in entries
        if isinstance(entry, tuple) and entry and isinstance(entry[0], str)
    ]

    assert labels
    assert not [label for label in labels if "Old Song" in label]
    assert not hasattr(ProjectManager, "migrate_project")
    assert not hasattr(ProjectManagementActionsMixin, "migrate_project_action")


def _two_takes(path, first_late=0.12, second_late=0.2, second_level=0.05):
    audio = np.zeros(RATE * 24, dtype=np.float32)
    burst = np.hanning(2000) * np.sin(np.arange(2000) * 0.2)
    for start, late, level in (
        (1.0, first_late, 0.4),
        (14.0, second_late, second_level),
    ):
        for beat in range(16):
            first = int((start + beat * BEAT_SECONDS + late) * RATE)
            audio[first : first + 2000] += burst * level
    sf.write(str(path), audio, RATE)
    return str(path)


@pytest.fixture
def fixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / clip_polish.POLISHED_FOLDER


def _fixed(qapp, recording, start=0.0):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = 120.0
    timeline.notes_audio = lambda _song, _tempo, _rate: None
    clip = timeline.add_audio_clip(recording, start, name="Takes")
    said = []
    timeline.clip_edit_status.connect(said.append)
    timeline.fix_clip_automatically(clip.id)
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()
    return timeline, said


def _first_sound(clip):
    data, rate = sf.read(clip.path, dtype="float32")
    loud = np.flatnonzero(np.abs(data) > 0.3 * np.abs(data).max())
    return clip.start + loud[0] / rate


def test_a_clip_with_two_separate_takes_is_split_and_each_take_is_fixed_alone(
    qapp, tmp_path, fixed_files
):
    timeline, said = _fixed(qapp, _two_takes(tmp_path / "takes.wav"))

    first, second = sorted(timeline.model.clips, key=lambda clip: clip.start)
    assert [first.name, second.name] == ["Takes 1", "Takes 2"]
    assert first.path != second.path
    assert first.end < second.start
    for part in (first, second):
        sound = _first_sound(part)
        assert sound / BEAT_SECONDS == pytest.approx(
            round(sound / BEAT_SECONDS), abs=0.08
        )
        data, rate = sf.read(part.path, dtype="float32")
        assert clip_level.active_level_db(data) == pytest.approx(
            clip_level.level_goal_db(), abs=clip_level.LEVEL_TOLERANCE_DB
        )
    assert "split into 2 parts" in said[-1]


def test_each_take_moves_by_its_own_amount(qapp, tmp_path, fixed_files):
    timeline, _said = _fixed(
        qapp,
        _two_takes(tmp_path / "takes.wav", first_late=0.12, second_late=-0.1),
    )

    first, second = sorted(timeline.model.clips, key=lambda clip: clip.start)
    gap = _first_sound(second) - _first_sound(first)
    assert gap == pytest.approx(13.0, abs=0.06)
    assert gap != pytest.approx(13.0 - 0.22, abs=0.06)


def test_one_step_back_returns_the_single_unsplit_clip(
    qapp, tmp_path, fixed_files
):
    recording = _two_takes(tmp_path / "takes.wav")
    timeline, _said = _fixed(qapp, recording, start=4.0)

    timeline.undo()

    assert [
        (clip.name, clip.path, clip.start) for clip in timeline.model.clips
    ] == [("Takes", recording, 4.0)]


def test_a_clip_with_one_continuous_take_is_not_split(
    qapp, tmp_path, fixed_files
):
    audio = np.zeros(RATE * 12, dtype=np.float32)
    burst = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * 0.4
    for beat in range(2, 20):
        first = int((beat * BEAT_SECONDS + 0.1) * RATE)
        audio[first : first + 2000] += burst
    sf.write(str(tmp_path / "take.wav"), audio, RATE)

    timeline, said = _fixed(qapp, str(tmp_path / "take.wav"))

    assert [clip.name for clip in timeline.model.clips] == ["Takes"]
    assert "split" not in said[-1]


def test_only_pauses_longer_than_the_splitting_pause_split_a_clip():
    time = np.arange(RATE * 12) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    short_pause = 0.5 * clip_polish.MIN_PART_GAP_S
    resume = int(RATE * (4.0 + short_pause))
    audio = np.zeros_like(tone)
    audio[RATE : RATE * 4] = tone[RATE : RATE * 4]
    audio[resume : RATE * 9] = tone[resume : RATE * 9]

    assert len(clip_polish.clip_parts(audio, RATE)) == 1
    audio[RATE * 4 : RATE * 9] = 0.0
    audio[RATE * 9 : RATE * 11] = tone[RATE * 9 : RATE * 11]
    assert [
        (round(begin), round(end))
        for begin, end in clip_polish.clip_parts(audio, RATE)
    ] == [(1, 4), (9, 11)]


def test_split_takes_leave_only_the_files_the_new_clips_use(
    qapp, tmp_path, fixed_files
):
    timeline, _said = _fixed(qapp, _two_takes(tmp_path / "takes.wav"))

    used = sorted(clip.path for clip in timeline.model.clips)
    assert sorted(str(path) for path in fixed_files.iterdir()) == used
