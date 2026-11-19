import sys

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer.audio_processing import clip_polish
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.project_management import song_folder
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100
BEAT_SECONDS = 0.5


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def fixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / clip_polish.POLISHED_FOLDER


def _held_note(seconds, level=0.3):
    time = np.arange(int(RATE * seconds)) / RATE
    wobble = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time)
    return (level * np.sin(2 * np.pi * 220 * time) * wobble).astype(np.float32)


def _joined(noise_under_sound_db=None, levels=(0.3, 0.3), pause=4.0):
    pieces = [
        np.zeros(RATE * 2, dtype=np.float32),
        _held_note(5, levels[0]),
        np.zeros(int(RATE * pause), dtype=np.float32),
        _held_note(4, levels[1]),
        np.zeros(RATE * 3, dtype=np.float32),
    ]
    audio = np.concatenate(pieces)
    if noise_under_sound_db is not None:
        hiss = np.random.default_rng(0).normal(0, 1, audio.shape)
        audio = audio + 0.3 * 10 ** (noise_under_sound_db / 20.0) * hiss
    return audio.astype(np.float32)


def _spans(audio):
    return clip_polish.clip_parts(audio, RATE)


def _two_sounds_at_their_places(spans):
    return len(spans) == 2 and np.allclose(
        spans, [(1.95, 7.05), (10.95, 15.05)], atol=0.1
    )


def test_two_sounds_joined_by_silence_are_found_as_two_parts():
    assert _two_sounds_at_their_places(_spans(_joined()))


@pytest.mark.parametrize("noise", [-50.0, -35.0, -25.0])
def test_two_sounds_joined_by_background_noise_are_found_as_two_parts(noise):
    assert _two_sounds_at_their_places(_spans(_joined(noise)))


def test_noise_nearly_as_loud_as_the_sound_does_not_split_the_clip():
    assert len(_spans(_joined(-12.0))) == 1


def test_a_much_quieter_second_sound_is_still_found_over_background_noise():
    assert _two_sounds_at_their_places(
        _spans(_joined(-45.0, levels=(0.3, 0.03)))
    )


def test_a_soft_passage_inside_continuous_music_is_not_cut_out():
    time = np.arange(RATE * 20) / RATE
    soft = np.where((time > 8) & (time < 12), 0.12, 1.0)
    music = (0.3 * np.sin(2 * np.pi * 220 * time) * soft).astype(np.float32)

    assert np.allclose(_spans(music), [(0.0, 20.0)], atol=0.1)


def test_a_slow_fade_in_and_out_stays_one_part():
    time = np.arange(RATE * 20) / RATE
    fade = np.clip(np.minimum(time / 6.0, (20.0 - time) / 6.0), 0.0, 1.0)
    music = (0.3 * np.sin(2 * np.pi * 220 * time) * fade).astype(np.float32)

    assert len(_spans(music)) == 1


def test_a_short_pause_keeps_the_sounds_together():
    assert len(_spans(_joined(-40.0, pause=1.0))) == 1


def _clicks(start, late, count=12, level=0.4):
    click = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * level
    return [
        (int((start + index * BEAT_SECONDS + late) * RATE), click)
        for index in range(count)
    ]


def _recording(path, takes, seconds, noise=0.0):
    audio = np.zeros(int(RATE * seconds), dtype=np.float32)
    for take in takes:
        for first, click in _clicks(*take):
            audio[first : first + click.shape[0]] += click
    if noise:
        audio += np.random.default_rng(2).normal(0, noise, audio.shape)
    sf.write(str(path), audio.astype(np.float32), RATE)
    return str(path)


def _fixed(qapp, recording, start=0.0):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = 120.0
    timeline.notes_audio = lambda _song, _tempo, _rate: None
    clip = timeline.add_audio_clip(recording, start, name="Take")
    timeline.wait_for_waveforms()
    said = []
    timeline.clip_edit_status.connect(said.append)
    timeline.fix_clip_automatically(clip.id)
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()
    return timeline, said


def _first_sound(clip):
    data, rate = sf.read(clip.path, dtype="float32")
    first = int(clip.offset * rate)
    kept = data[first : first + int(clip.duration * rate)]
    loud = np.flatnonzero(np.abs(kept) > 0.3 * np.abs(kept).max())
    return clip.start + loud[0] / rate


def _on_a_beat(seconds):
    return seconds / BEAT_SECONDS == pytest.approx(
        round(seconds / BEAT_SECONDS), abs=0.1
    )


def test_sounds_joined_by_background_noise_become_clips_that_each_sit_on_the_beat(
    qapp, tmp_path, fixed_files
):
    recording = _recording(
        tmp_path / "takes.wav", [(1.0, 0.13), (12.0, -0.11)], 20, noise=0.004
    )

    timeline, said = _fixed(qapp, recording)

    first, second = sorted(timeline.model.clips, key=lambda clip: clip.start)
    assert _on_a_beat(_first_sound(first))
    assert _on_a_beat(_first_sound(second))
    assert "split into 2 parts" in said[-1]


def test_each_separated_sound_takes_only_the_room_it_needs(
    qapp, tmp_path, fixed_files
):
    recording = _recording(
        tmp_path / "takes.wav", [(3.0, 0.0), (14.0, 0.0)], 24
    )

    timeline, _said = _fixed(qapp, recording)

    lengths = sorted(clip.duration for clip in timeline.model.clips)
    assert lengths == pytest.approx([5.65, 5.65], abs=0.15)
    assert sum(lengths) < 12.0


def test_a_clip_with_silence_at_both_ends_is_trimmed_to_its_sound(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "take.wav", [(4.0, 0.1)], 16)

    timeline, said = _fixed(qapp, recording, start=2.0)

    clip = timeline.model.clips[0]
    assert len(timeline.model.clips) == 1
    assert clip.duration == pytest.approx(5.65, abs=0.15)
    assert clip.offset == pytest.approx(4.05, abs=0.1)
    assert _on_a_beat(_first_sound(clip))
    assert abs(clip.start - (2.0 + 4.05)) <= BEAT_SECONDS / 2 + 0.05
    assert "trimmed" in said[-1]


def test_trimming_the_silence_keeps_the_recording_whole_for_stretching_back_out(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "take.wav", [(4.0, 0.0)], 16)
    timeline, _said = _fixed(qapp, recording, start=5.0)
    clip = timeline.model.clips[0]

    timeline.model.trim_clip(clip.id, clip.start - 20.0, clip.end + 20.0)

    assert clip.duration == pytest.approx(16.0, abs=0.01)
    assert sf.info(clip.path).duration == pytest.approx(16.0, abs=0.01)


def test_one_step_back_restores_the_untrimmed_clip(qapp, tmp_path, fixed_files):
    recording = _recording(tmp_path / "take.wav", [(4.0, 0.1)], 16)
    timeline, _said = _fixed(qapp, recording, start=2.0)

    timeline.undo()

    clip = timeline.model.clips[0]
    assert (clip.path, clip.start, clip.offset, clip.duration) == (
        recording,
        2.0,
        0.0,
        pytest.approx(16.0),
    )


def test_a_clip_that_is_all_sound_is_not_trimmed(qapp, tmp_path, fixed_files):
    recording = _recording(tmp_path / "take.wav", [(0.0, 0.05, 22)], 10.65)

    timeline, said = _fixed(qapp, recording, start=2.0)

    clip = timeline.model.clips[0]
    assert clip.duration == pytest.approx(10.65, abs=0.05)
    assert "trimmed" not in said[-1]


class _Studio(ProjectManagementActionsMixin, EditorActionsMixin):
    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.current_session = None


def test_saving_a_song_leaves_files_in_the_shared_folders_alone(qapp, tmp_path):
    studio = _Studio(tmp_path / "app")
    shared = studio.project_manager.saved_data_dir / "snapshots"
    shared.mkdir(parents=True)
    earlier = shared / "Ballad_snapshot_20260101_000000.kproj"
    earlier.write_bytes(b"x")
    song = studio.project_manager.song_path_for("Ballad")

    assert studio._save_song_to(song)

    assert earlier.is_file()
    assert not (song.parent / "Versions").exists()
    assert not hasattr(song_folder, "history_kept_in_song")
