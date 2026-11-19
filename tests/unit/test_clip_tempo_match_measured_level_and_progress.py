import os
import sys

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer.audio import loader
from korshmer.audio_processing import clip_level, clip_polish
from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100
SONG_BPM = 120.0
BEAT_SECONDS = 60.0 / SONG_BPM


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def fixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / clip_polish.POLISHED_FOLDER


def _sung_phrases(path, level=0.3, quiet_phrase=1.0, seconds=16):
    time = np.arange(RATE * seconds) / RATE
    voice = sum(
        np.sin(
            2
            * np.pi
            * pitch
            * time
            * (1 + 0.004 * np.sin(2 * np.pi * 5 * time))
        )
        / part
        for part, pitch in enumerate((220, 440, 660, 880, 1760), 1)
    )
    phrase = np.clip(np.sin(2 * np.pi * time / 4.0) * 3.0, 0.0, 1.0)
    audio = level * voice * phrase / np.abs(voice).max()
    audio[int(8 * RATE) : int(12 * RATE)] *= quiet_phrase
    audio += np.random.default_rng(2).normal(0, level * 0.002, audio.shape)
    sf.write(str(path), audio.astype(np.float32), RATE)
    return str(path)


def _drum_loop(path, bpm, seconds=20, late_seconds=0.0):
    rng = np.random.default_rng(4)
    audio = np.zeros(int(RATE * seconds), dtype=np.float32)
    hit = np.exp(-np.arange(3000) / 400.0) * rng.normal(0, 0.5, 3000)
    step = 60.0 / bpm / 2.0
    position, count = late_seconds, 0
    while position + 0.1 < seconds:
        first = int(position * RATE)
        audio[first : first + 3000] += hit * (1.0 if count % 2 == 0 else 0.5)
        position += step
        count += 1
    sf.write(str(path), audio, RATE)
    return str(path)


def _timeline(qapp, tempo=SONG_BPM):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = tempo
    return timeline


def _finish(timeline):
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()


def _with_clip(qapp, recording, start=0.0, **options):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(recording, start, **options)
    said = []
    timeline.clip_edit_status.connect(said.append)
    return timeline, clip, said


def _readings(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return clip_level.measure(data, rate)


def _tone_balance(path):
    data, _rate = sf.read(path, dtype="float32")
    spectrum = np.abs(np.fft.rfft(data[: RATE * 4] * np.hanning(RATE * 4)))
    bands = [
        spectrum[low * 4 : high * 4].sum()
        for low, high in ((150, 300), (300, 600), (600, 1200), (1200, 2400))
    ]
    return 20 * np.log10(np.array(bands) / bands[0])


def _hit_times(path):
    from korshmer.audio_processing import clip_tempo

    data, rate = sf.read(path, dtype="float32")
    seconds, weights = clip_tempo.onsets(data, rate)
    return seconds[weights > 0.5 * weights.max()]


def test_improving_a_quiet_recording_brings_it_to_the_stated_level_and_checks_it(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, said = _with_clip(qapp, original, gain=1.8)

    timeline.improve_clip_quality(clip.id)
    _finish(timeline)

    clip = timeline.model.clip_by_id(clip.id)
    after = _readings(clip.path)
    assert clip.path != original
    assert after[clip_level.LEVEL] == pytest.approx(
        clip_level.level_goal_db(), abs=clip_level.LEVEL_TOLERANCE_DB
    )
    assert after[clip_level.PEAK] <= clip_level.PEAK_LIMIT_DB + 0.1
    assert clip.gain == 1.0
    assert "reached" in said[-1] and "not reached" not in said[-1]
    assert f"goal {clip_level.level_goal_db():.1f}" in said[-1]


def test_improving_a_recording_changes_its_level_but_not_its_tone_or_length(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.05)
    timeline, clip, _said = _with_clip(qapp, original)

    timeline.improve_clip_quality(clip.id)
    _finish(timeline)

    improved = timeline.model.clip_by_id(clip.id).path
    assert _tone_balance(improved) == pytest.approx(
        _tone_balance(original), abs=0.5
    )
    assert sf.info(improved).frames == sf.info(original).frames
    before, _rate = sf.read(original, dtype="float32")
    after, _rate = sf.read(improved, dtype="float32")
    quietest = np.abs(before) < 0.25 * np.abs(before).max()
    ratio = after[~quietest] / before[~quietest]
    assert np.percentile(ratio, 95) / np.percentile(ratio, 5) < 1.1


def test_a_recording_that_already_meets_the_goals_is_left_untouched(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, said = _with_clip(qapp, original)
    timeline.improve_clip_quality(clip.id)
    _finish(timeline)
    improved = timeline.model.clip_by_id(clip.id).path

    timeline.improve_clip_quality(clip.id)
    _finish(timeline)
    timeline.undo()

    assert "already meets every goal" in said[-1]
    assert timeline.model.clip_by_id(clip.id).path == original
    assert os.path.isfile(improved)


def test_improving_evens_out_a_phrase_that_is_much_quieter_than_the_rest(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(
        tmp_path / "voice.wav", level=0.2, quiet_phrase=0.15
    )
    timeline, clip, said = _with_clip(qapp, original)
    assert _readings(original)[clip_level.PHRASES] > 10.0

    timeline.improve_clip_quality(clip.id)
    _finish(timeline)

    after = _readings(timeline.model.clip_by_id(clip.id).path)
    assert after[clip_level.PHRASES] <= (
        clip_level.PHRASE_SPREAD_GOAL_DB + clip_level.PHRASE_SPREAD_TOLERANCE_DB
    )
    assert "phrase spread" in said[-1]


def test_improving_an_overloaded_recording_lowers_it_under_the_peak_limit(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.999)
    timeline, clip, _said = _with_clip(qapp, original)

    timeline.improve_clip_quality(clip.id)
    _finish(timeline)

    after = _readings(timeline.model.clip_by_id(clip.id).path)
    assert after[clip_level.PEAK] <= clip_level.PEAK_LIMIT_DB + 0.1
    assert after[clip_level.LEVEL] == pytest.approx(
        clip_level.level_goal_db(), abs=clip_level.LEVEL_TOLERANCE_DB
    )


def test_a_goal_that_cannot_be_reached_is_reported_as_not_reached():
    readings = [
        clip_level.Reading(clip_level.LEVEL, -40.0, -17.5, -30.0, False, False)
    ]

    assert "not reached" in clip_level.summary(readings)


def test_aligning_a_drum_loop_recorded_slower_than_the_song_matches_its_tempo(
    qapp, tmp_path, fixed_files
):
    loop = _drum_loop(tmp_path / "loop.wav", bpm=104.0, late_seconds=0.08)
    timeline, clip, said = _with_clip(qapp, loop, start=2.0)
    timeline.wait_for_waveforms()
    length = clip.duration

    timeline.align_clip_to_beat(clip.id)
    _finish(timeline)

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.duration == pytest.approx(length * 104.0 / SONG_BPM, rel=0.005)
    hits = clip.start + _hit_times(clip.path)
    off_grid = np.abs(
        hits / (BEAT_SECONDS / 2) - np.round(hits / (BEAT_SECONDS / 2))
    )
    assert np.max(off_grid) * BEAT_SECONDS / 2 < 0.035
    assert "tempo matched from 104.0 to 120.0 BPM" in said[-1]

    timeline.undo()
    restored = timeline.model.clip_by_id(clip.id)
    assert (restored.path, restored.start) == (loop, 2.0)
    assert restored.duration == pytest.approx(length)


def test_matching_the_tempo_keeps_the_same_part_of_a_cut_clip(
    qapp, tmp_path, fixed_files
):
    loop = _drum_loop(tmp_path / "loop.wav", bpm=104.0)
    timeline, clip, _said = _with_clip(qapp, loop, start=0.0)
    timeline.wait_for_waveforms()
    timeline.model.trim_clip(clip.id, 4.0, 16.0)
    timeline.piano_roll.load_model()

    timeline.align_clip_to_beat(clip.id)
    _finish(timeline)

    clip = timeline.model.clip_by_id(clip.id)
    ratio = 104.0 / SONG_BPM
    assert clip.offset == pytest.approx(4.0 * ratio, rel=0.005)
    assert clip.duration == pytest.approx(12.0 * ratio, rel=0.005)


def test_a_drum_loop_already_at_the_song_tempo_is_not_stretched(
    qapp, tmp_path, fixed_files
):
    loop = _drum_loop(tmp_path / "loop.wav", bpm=SONG_BPM, late_seconds=0.1)
    timeline, clip, said = _with_clip(qapp, loop, start=2.0)

    timeline.align_clip_to_beat(clip.id)
    _finish(timeline)

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.path == loop
    assert clip.start == pytest.approx(1.9, abs=0.03)
    assert "tempo matched" not in said[-1]


def test_a_recording_without_a_clear_tempo_is_never_stretched(
    qapp, tmp_path, fixed_files
):
    voice = _sung_phrases(tmp_path / "voice.wav")
    timeline, clip, said = _with_clip(qapp, voice, start=1.1)

    timeline.align_clip_to_beat(clip.id)
    _finish(timeline)

    assert timeline.model.clip_by_id(clip.id).path == voice
    assert "tempo matched" not in said[-1]


def test_tempo_matching_is_skipped_while_the_song_itself_changes_tempo(
    qapp, tmp_path, fixed_files
):
    loop = _drum_loop(tmp_path / "loop.wav", bpm=104.0)
    timeline, clip, said = _with_clip(qapp, loop, start=2.0)
    timeline.model.tempo_map = TempoMap((0.0, 8.0), (120.0, 126.0))

    timeline.align_clip_to_beat(clip.id)
    _finish(timeline)

    assert timeline.model.clip_by_id(clip.id).path == loop
    assert "tempo matched" not in said[-1]


def test_a_running_fix_shows_its_steps_and_how_far_it_is(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, _said = _with_clip(qapp, original)
    steps = []

    worker = timeline.fix_clip_automatically(clip.id)
    worker.edit_progress.connect(
        lambda _clip, _label, stage, percent: steps.append((stage, percent))
    )
    shown_while_running = timeline.clip_edit_bar.isVisible()
    _finish(timeline)

    stages = [stage for stage, _percent in steps]
    percents = [percent for _stage, percent in steps]
    assert shown_while_running
    assert stages[0] == "Looking for separate parts"
    assert "Removing background noise" in stages
    assert "Measuring and improving the level" in stages
    assert stages[-1] == "Finding the beat"
    assert percents == sorted(percents) and percents[-1] < 100
    assert not timeline.clip_edit_bar.isVisible()


def test_cancelling_a_fix_leaves_the_clip_and_the_folder_as_they_were(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, said = _with_clip(qapp, original, start=1.0)

    timeline.fix_clip_automatically(clip.id)
    timeline.btn_cancel_clip_edit.click()
    _finish(timeline)

    clip = timeline.model.clip_by_id(clip.id)
    assert (clip.path, clip.start) == (original, 1.0)
    assert "cancelled" in said[-1]
    assert not fixed_files.exists() or not list(fixed_files.iterdir())
    assert not timeline.clip_edit_bar.isVisible()


def test_repeated_fixes_keep_one_short_file_name_ending(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, _said = _with_clip(qapp, original)

    for _ in range(2):
        timeline.remove_clip_noise(clip.id)
        _finish(timeline)
        timeline.improve_clip_quality(clip.id)
        _finish(timeline)

    name = os.path.basename(timeline.model.clip_by_id(clip.id).path)
    assert name.startswith("voice_")
    assert name.count("clean") + name.count("improved") == 1


def test_a_full_automatic_fix_leaves_no_half_finished_files_behind(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, _said = _with_clip(qapp, original)

    timeline.fix_clip_automatically(clip.id)
    _finish(timeline)

    final = timeline.model.clip_by_id(clip.id).path
    assert [str(path) for path in fixed_files.iterdir()] == [final]


def test_a_fixed_file_is_removed_once_no_step_back_can_reach_it(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, _said = _with_clip(qapp, original)
    timeline.improve_clip_quality(clip.id)
    _finish(timeline)
    improved = timeline.model.clip_by_id(clip.id).path

    timeline.undo()
    still_there_for_redo = os.path.isfile(improved)
    timeline.add_audio_clip(original, 20.0)

    assert still_there_for_redo
    assert not os.path.exists(improved)
    assert os.path.isfile(original)


def test_a_fixed_file_that_was_saved_with_the_song_is_never_removed(
    qapp, tmp_path, fixed_files
):
    original = _sung_phrases(tmp_path / "voice.wav", level=0.02)
    timeline, clip, _said = _with_clip(qapp, original)
    timeline.improve_clip_quality(clip.id)
    _finish(timeline)
    improved = timeline.model.clip_by_id(clip.id).path
    timeline.to_project_dict()

    timeline.undo()
    timeline.add_audio_clip(original, 20.0)

    assert os.path.isfile(improved)


def test_two_unreadable_recordings_with_the_same_name_get_separate_converted_files(
    tmp_path, monkeypatch
):
    for folder in ("monday", "tuesday"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "take.ogg").write_bytes(folder.encode())
    made = []

    def convert(command, **_options):
        target = command[-1]
        with open(target, "wb") as converted:
            converted.write(b"converted")
        made.append(target)

    monkeypatch.setattr(loader.shutil, "which", lambda _name: "converter")
    monkeypatch.setattr(loader.subprocess, "run", convert)

    first = loader.usable_audio_path(
        tmp_path / "monday" / "take.ogg", tmp_path / "out"
    )
    second = loader.usable_audio_path(
        tmp_path / "tuesday" / "take.ogg", tmp_path / "out"
    )

    assert first != second
    assert len(made) == 2
