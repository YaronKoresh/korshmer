import sys

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer.audio_processing import clip_level, clip_notes, clip_polish
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def fixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / clip_polish.POLISHED_FOLDER


def _singing(level=0.3, seconds=16, pitches=(220, 440, 660, 880, 1760)):
    time = np.arange(RATE * seconds) / RATE
    voice = sum(
        np.sin(2 * np.pi * pitch * time) / part
        for part, pitch in enumerate(pitches, 1)
    )
    phrase = np.clip(np.sin(2 * np.pi * time / 4.0) * 3.0, 0.0, 1.0)
    return (level * voice * phrase / np.abs(voice).max()).astype(np.float32)


def _hiss(below_singing_db, singing, seed=7):
    hiss = np.random.default_rng(seed).normal(0, 1, singing.shape)
    wanted = clip_level.active_level_db(singing) - below_singing_db
    hiss *= 10 ** (wanted / 20.0) / np.sqrt(np.mean(hiss**2))
    return hiss.astype(np.float32)


def _chords(level, seconds=16, pitches=(220, 440, 660, 880)):
    time = np.arange(RATE * seconds) / RATE
    chord = sum(
        np.sin(2 * np.pi * pitch * time + part)
        for part, pitch in enumerate(pitches)
    )
    return (level * chord / np.abs(chord).max()).astype(np.float32)


def _reading(readings, name):
    return next(reading for reading in readings if reading.name == name)


def _names(readings):
    return [reading.name for reading in readings]


def _pause(audio):
    return audio[int(2.6 * RATE) : int(3.6 * RATE)]


def _sung(audio):
    return audio[int(0.6 * RATE) : int(1.6 * RATE)]


def test_improving_a_hissy_recording_quiets_the_hiss_to_its_goal_and_keeps_the_singing():
    singing = _singing()
    recording = singing + _hiss(25.0, singing)

    improved, readings = clip_level.improve(recording, RATE)

    noise = _reading(readings, clip_level.NOISE)
    assert not noise.was_reached and noise.reached
    assert (
        noise.after <= clip_level.NOISE_GOAL_DB + clip_level.NOISE_TOLERANCE_DB
    )
    lift = np.sqrt(
        np.mean(_sung(improved) ** 2) / np.mean(_sung(recording) ** 2)
    )
    hiss_left = np.sqrt(
        np.mean(_pause(improved) ** 2) / np.mean(_pause(recording) ** 2)
    )
    assert hiss_left < 0.35 * lift
    assert "noise in pauses" in clip_level.summary(readings)


def test_the_hiss_is_cut_by_the_measured_gap_and_not_by_more():
    singing = _singing()
    slightly = singing + _hiss(35.0, singing)
    badly = singing + _hiss(24.0, singing)

    after_slight, _readings = clip_level.improve(slightly, RATE)
    after_bad, _readings = clip_level.improve(badly, RATE)

    def cut_db(before, after):
        lift = np.mean(_sung(after) ** 2) / np.mean(_sung(before) ** 2)
        left = np.mean(_pause(after) ** 2) / np.mean(_pause(before) ** 2)
        return 10 * np.log10(lift / left)

    assert 3.0 < cut_db(slightly, after_slight) < 10.0
    assert cut_db(badly, after_bad) > cut_db(slightly, after_slight) + 6.0


def test_a_clean_recording_with_silent_pauses_is_not_treated_as_noisy():
    _improved, readings = clip_level.improve(_singing(level=0.05), RATE)

    assert _reading(readings, clip_level.NOISE).was_reached
    assert "noise" not in clip_level.summary(readings)


def test_music_without_pauses_is_not_mistaken_for_noise():
    music = _chords(0.3) * (
        0.6 + 0.4 * np.sin(np.arange(RATE * 16) / RATE) ** 2
    )

    improved, readings = clip_level.improve(music.astype(np.float32), RATE)

    assert _reading(readings, clip_level.NOISE).was_reached
    ratio = improved[RATE : RATE * 15] / music[RATE : RATE * 15]
    steady = np.abs(music[RATE : RATE * 15]) > 0.05
    assert (
        np.percentile(ratio[steady], 95) / np.percentile(ratio[steady], 5) < 1.3
    )


def test_a_clip_is_raised_above_loud_notes_that_play_at_the_same_time():
    singing = _singing(level=0.05)
    notes = _chords(0.95, pitches=(110, 165, 2500, 3300))

    alone, _readings = clip_level.improve(singing, RATE)
    beside, readings = clip_level.improve(singing, RATE, notes)

    notes_level = clip_level.notes_levels(singing, notes)[0]
    assert (
        clip_level.active_level_db(beside)
        > clip_level.active_level_db(alone) + 3.0
    )
    assert clip_level.active_level_db(beside) == pytest.approx(
        notes_level + clip_level.VOCAL_ABOVE_MIDI_DB,
        abs=clip_level.LEVEL_TOLERANCE_DB,
    )
    assert _reading(readings, clip_level.LEVEL).reached


def test_a_clip_is_not_lowered_because_the_notes_beside_it_are_quiet():
    singing = _singing(level=0.05)

    alone, _readings = clip_level.improve(singing, RATE)
    beside, _readings = clip_level.improve(singing, RATE, _chords(0.01))

    assert clip_level.active_level_db(beside) == pytest.approx(
        clip_level.active_level_db(alone), abs=0.1
    )


def test_a_clip_hidden_by_notes_in_its_own_pitch_range_gets_an_extra_lift():
    singing = _singing(level=0.05)
    same_range = _chords(0.4, pitches=(220, 440, 660, 880))
    other_range = _chords(0.4, pitches=(55, 82, 5000, 6000))

    hidden, readings = clip_level.improve(singing, RATE, same_range)
    clear, _readings = clip_level.improve(singing, RATE, other_range)

    over = _reading(readings, clip_level.OVER_NOTES)
    assert not over.was_reached
    assert over.after > over.before
    assert (
        clip_level.active_level_db(hidden)
        > clip_level.active_level_db(clear) + 1.0
    )
    assert "over the notes" in clip_level.summary(readings)


def test_the_extra_lift_over_the_notes_is_limited_and_never_breaks_the_peak_limit():
    singing = _singing(level=0.05)
    drowning = _chords(0.9, pitches=(220, 440, 660, 880))

    improved, readings = clip_level.improve(singing, RATE, drowning)

    level = _reading(readings, clip_level.LEVEL)
    notes_level = clip_level.notes_levels(singing, drowning)[0]
    assert level.goal <= (
        notes_level
        + clip_level.VOCAL_ABOVE_MIDI_DB
        + clip_level.MAX_LIFT_OVER_NOTES_DB
        + 0.01
    )
    assert (
        np.abs(improved).max() <= 10 ** (clip_level.PEAK_LIMIT_DB / 20.0) + 1e-3
    )
    assert level.goal > notes_level + clip_level.VOCAL_ABOVE_MIDI_DB


def test_a_clip_already_balanced_against_the_notes_is_left_untouched():
    singing = _singing(level=0.05)
    notes = _chords(0.4, pitches=(110, 165, 2500, 3300))
    improved, _readings = clip_level.improve(singing, RATE, notes)

    again, readings = clip_level.improve(improved, RATE, notes)

    assert clip_level.unchanged(readings)
    assert np.array_equal(again, improved)


def test_without_any_notes_no_balance_with_notes_is_reported():
    _improved, readings = clip_level.improve(_singing(level=0.05), RATE)

    assert clip_level.OVER_NOTES not in _names(readings)


def test_only_the_notes_that_sound_while_the_clip_plays_are_compared():
    song = np.zeros(RATE * 40, dtype=np.float32)
    song[RATE * 10 : RATE * 26] = _chords(0.5)

    during = clip_notes.beside_clip(song, RATE, 10.0, RATE * 16)
    after = clip_notes.beside_clip(song, RATE, 30.0, RATE * 16)
    early = clip_notes.beside_clip(song, RATE, -2.0, RATE * 16)

    assert np.abs(during).max() > 0.4
    assert np.abs(after).max() == 0.0
    assert np.abs(early[: RATE * 2]).max() == 0.0
    assert early.shape == during.shape == (RATE * 16,)


def _timeline_with_clip(qapp, tmp_path, notes, start=0.0):
    recording = tmp_path / "voice.wav"
    sf.write(str(recording), _singing(level=0.05), RATE)
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.notes_audio = lambda _song, _tempo, _rate: notes
    clip = timeline.add_audio_clip(str(recording), start)
    said = []
    timeline.clip_edit_status.connect(said.append)
    return timeline, clip, said


def _improve(timeline, clip):
    steps = []
    worker = timeline.improve_clip_quality(clip.id)
    worker.edit_progress.connect(
        lambda _clip, _label, stage, _percent: steps.append(stage)
    )
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()
    data, _rate = sf.read(
        timeline.model.clip_by_id(clip.id).path, dtype="float32"
    )
    return clip_level.active_level_db(data), steps


def test_improving_a_clip_in_the_editor_listens_to_the_notes_under_it(
    qapp, tmp_path, fixed_files
):
    loud_notes = np.zeros(RATE * 40, dtype=np.float32)
    loud_notes[: RATE * 16] = _chords(0.95, pitches=(110, 165, 2500, 3300))

    timeline, clip, said = _timeline_with_clip(qapp, tmp_path, loud_notes)
    level, steps = _improve(timeline, clip)

    assert steps[0] == "Listening to the notes of the song"
    assert level > clip_level.level_goal_db() + 3.0
    assert "over the notes" in said[-1]
    assert timeline.model.clip_by_id(clip.id).gain == 1.0


def test_a_clip_placed_where_no_notes_play_gets_the_usual_level(
    qapp, tmp_path, fixed_files
):
    loud_notes = np.zeros(RATE * 60, dtype=np.float32)
    loud_notes[: RATE * 16] = _chords(0.5)

    timeline, clip, said = _timeline_with_clip(
        qapp, tmp_path, loud_notes, start=30.0
    )
    level, _steps = _improve(timeline, clip)

    assert level == pytest.approx(
        clip_level.level_goal_db(), abs=clip_level.LEVEL_TOLERANCE_DB
    )
    assert "over the notes" not in said[-1]


def test_a_song_without_notes_or_without_an_instrument_sound_still_improves_the_clip(
    qapp, tmp_path, fixed_files
):
    timeline, clip, said = _timeline_with_clip(qapp, tmp_path, None)
    level, _steps = _improve(timeline, clip)

    assert level == pytest.approx(
        clip_level.level_goal_db(), abs=clip_level.LEVEL_TOLERANCE_DB
    )
    assert "reached" in said[-1]


def test_an_empty_song_gives_no_notes_to_compare_with():
    import pretty_midi

    assert clip_notes.notes_audio(pretty_midi.PrettyMIDI(), 120.0, RATE) is None
    assert clip_notes.notes_audio(None, 120.0, RATE) is None
