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


def _joined_by_rumble(under_sound_db):
    audio = _joined()
    drift = np.cumsum(np.random.default_rng(6).normal(0, 1, audio.shape))
    drift -= np.convolve(drift, np.ones(2000) / 2000, "same")
    drift /= drift.std()
    return (audio + 0.21 * 10 ** (under_sound_db / 20.0) * drift).astype(
        np.float32
    )


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
    assert len(_spans(_joined(-4.0))) == 1


@pytest.mark.parametrize("noise", [-18.0, -14.0, -12.0])
def test_two_tones_joined_by_loud_hiss_are_still_found_as_two_parts(noise):
    spans = _spans(_joined(noise))

    assert len(spans) == 2
    assert np.allclose(spans, [(1.95, 7.0), (10.95, 15.05)], atol=0.2)


@pytest.mark.parametrize("rumble", [-35.0, -25.0, -18.0, -12.0])
def test_two_tones_joined_by_unsteady_low_rumble_are_found_as_two_parts(rumble):
    spans = _spans(_joined_by_rumble(rumble))

    assert len(spans) == 2
    assert np.allclose(spans, [(1.95, 7.0), (10.95, 15.05)], atol=0.2)


def test_rumble_almost_as_loud_as_the_sound_leaves_the_clip_whole():
    assert np.allclose(_spans(_joined_by_rumble(-4.0)), [(0.0, 18.0)], atol=0.1)


def test_a_wavering_note_with_a_soft_passage_is_not_cut_out():
    time = np.arange(RATE * 20) / RATE
    waver = 3.0 * np.sin(2 * np.pi * 5.5 * time)
    passage = np.where((time > 8) & (time < 12), 0.2, 1.0)
    music = 0.3 * np.sin(2 * np.pi * 220 * time + waver) * passage

    assert np.allclose(
        _spans(music.astype(np.float32)), [(0.0, 20.0)], atol=0.1
    )


@pytest.mark.parametrize("soft", [0.12, 0.25, 0.4])
def test_a_soft_passage_of_any_depth_is_not_mistaken_for_hiss(soft):
    time = np.arange(RATE * 20) / RATE
    passage = np.where((time > 8) & (time < 12), soft, 1.0)
    music = (0.3 * np.sin(2 * np.pi * 220 * time) * passage).astype(np.float32)

    assert np.allclose(_spans(music), [(0.0, 20.0)], atol=0.1)


def test_a_softer_stretch_of_a_noise_like_sound_is_not_cut_out():
    time = np.arange(RATE * 20) / RATE
    softer = np.where((time > 8) & (time < 12), 0.25, 1.0)
    wash = np.random.default_rng(4).normal(0, 0.1, time.shape) * softer

    assert np.allclose(_spans(wash.astype(np.float32)), [(0.0, 20.0)], atol=0.1)


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


def _with_inaudible_movement(kind):
    audio = _joined()
    time = np.arange(audio.shape[0]) / RATE
    if kind == "steady offset":
        return audio + 0.02
    if kind == "slow drift":
        return audio + 0.05 * np.sin(2 * np.pi * 8.0 * time)
    wander = 2.0 * np.sin(2 * np.pi * 0.3 * time)
    return audio + 0.03 * np.sin(2 * np.pi * 3.0 * time + wander)


@pytest.mark.parametrize(
    "kind", ["steady offset", "slow drift", "wandering drift"]
)
def test_movement_too_low_to_hear_still_counts_as_silence(kind):
    spans = _spans(_with_inaudible_movement(kind).astype(np.float32))

    assert _two_sounds_at_their_places(spans)


def test_a_pause_of_about_half_a_second_keeps_the_sounds_together():
    assert len(_spans(_joined(-40.0, pause=0.4))) == 1


def test_a_pause_of_one_second_is_enough_to_split_the_sounds():
    assert len(_spans(_joined(-40.0, pause=1.0))) == 2


def _hits(times, seconds):
    audio = np.zeros(int(RATE * seconds), dtype=np.float32)
    click = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * 0.4
    for moment in times:
        first = int(moment * RATE)
        audio[first : first + 2000] += click
    return audio


def test_a_slow_steady_pulse_is_not_broken_into_single_hits():
    pulse = _hits(np.arange(1.0, 21.0, 0.67), 22)

    assert len(_spans(pulse)) == 1


def test_a_slow_pulse_with_a_long_break_is_split_only_at_the_break():
    before = list(np.arange(1.0, 7.0, 0.67))
    after = list(np.arange(10.0, 16.0, 0.67))

    spans = _spans(_hits(before + after, 17))

    assert len(spans) == 2
    assert spans[0][1] < 7.5 and spans[1][0] > 9.5


def test_phrases_less_than_a_second_apart_become_separate_parts():
    time = np.arange(RATE * 14) / RATE
    audio = np.zeros_like(time)
    for begin, end in ((1.0, 3.0), (3.8, 6.0), (6.8, 9.0), (9.8, 12.0)):
        sounding = (time >= begin) & (time < end)
        audio += sounding * 0.3 * np.sin(2 * np.pi * 220 * time)

    assert len(_spans(audio.astype(np.float32))) == 4


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


def _fill(start, late, hits=2, level=0.4):
    click = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * level
    return [
        (int((start + index * BEAT_SECONDS / 2 + late) * RATE), click)
        for index in range(hits)
    ]


def _takes_with_a_short_fill(path, late):
    audio = np.zeros(RATE * 30, dtype=np.float32)
    hits = _clicks(1.0, late) + _fill(9.25, late) + _clicks(12.0, late)
    for first, click in hits:
        audio[first : first + click.shape[0]] += click
    sf.write(str(path), audio, RATE)
    return str(path)


def test_a_very_short_part_moves_with_the_part_next_to_it_and_keeps_its_place(
    qapp, tmp_path, fixed_files
):
    recording = _takes_with_a_short_fill(tmp_path / "takes.wav", late=0.12)

    timeline, said = _fixed(qapp, recording)

    first, fill, last = sorted(
        timeline.model.clips, key=lambda clip: clip.start
    )
    assert fill.duration < 1.0
    assert _on_a_beat(_first_sound(first))
    assert _on_a_beat(_first_sound(last))
    off_beat = _first_sound(fill) / BEAT_SECONDS
    assert off_beat - np.floor(off_beat) == pytest.approx(0.5, abs=0.1)
    assert "together with the part next to it" in said[-1]


def test_a_very_short_part_with_no_rhythmic_part_nearby_starts_on_a_beat(
    qapp, tmp_path, fixed_files
):
    audio = np.zeros(RATE * 12, dtype=np.float32)
    for first, click in _fill(2.37, 0.0) + _fill(7.61, 0.0):
        audio[first : first + click.shape[0]] += click
    sf.write(str(tmp_path / "fills.wav"), audio, RATE)

    timeline, _said = _fixed(qapp, str(tmp_path / "fills.wav"))

    assert len(timeline.model.clips) == 2
    for clip in timeline.model.clips:
        assert _on_a_beat(_first_sound(clip))


def test_quiet_ends_that_hold_only_inaudible_drift_are_trimmed_off(
    qapp, tmp_path, fixed_files
):
    recording = _recording(tmp_path / "take.wav", [(4.0, 0.1)], 16)
    audio, rate = sf.read(recording, dtype="float32")
    time = np.arange(audio.shape[0]) / rate
    sf.write(recording, audio + 0.05 * np.sin(2 * np.pi * 6.0 * time), rate)

    timeline, said = _fixed(qapp, recording, start=2.0)

    clip = timeline.model.clips[0]
    assert clip.duration == pytest.approx(5.65, abs=0.2)
    assert "trimmed" in said[-1]


def test_a_clip_holding_only_inaudible_drift_is_reported_as_silent(
    qapp, tmp_path, fixed_files
):
    time = np.arange(RATE * 6) / RATE
    drift = (0.05 * np.sin(2 * np.pi * 6.0 * time)).astype(np.float32)
    sf.write(str(tmp_path / "drift.wav"), drift, RATE)
    timeline = ArrangerTimeline()
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.model.tempo = 120.0
    clip = timeline.add_audio_clip(str(tmp_path / "drift.wav"), 1.1)
    said = []
    timeline.clip_edit_status.connect(said.append)

    timeline.align_clip_to_beat(clip.id)
    timeline.wait_for_clip_edits()
    QApplication.processEvents()

    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(1.1)
    assert "silent" in said[-1]
