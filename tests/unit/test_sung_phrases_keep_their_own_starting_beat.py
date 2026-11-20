import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer.audio_processing import clip_polish
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

RATE = 44100
BEAT_SECONDS = 0.5
SUNG_AT_BEATS = (0.0, 0.75, 1.25, 2.5, 3.0, 4.25, 5.0, 5.75, 7.5, 8.25)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def edited_clips_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


def _sung_phrase(path):
    audio = np.zeros(int(RATE * 6), dtype=np.float32)
    for index, beat in enumerate(SUNG_AT_BEATS):
        length = int(RATE * 0.2)
        time = np.arange(length) / RATE
        pitch = 220.0 + 30.0 * (index % 4)
        syllable = 0.4 * np.sin(2 * np.pi * pitch * time)
        syllable *= np.minimum(1.0, time / 0.004) * np.exp(-time * 9.0)
        syllable *= np.minimum(1.0, (0.2 - time) / 0.05)
        first = int((1.0 + beat) * BEAT_SECONDS * RATE)
        audio[first : first + length] += syllable.astype(np.float32)
    sf.write(str(path), audio, RATE)
    return str(path)


def _timeline(qapp):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.model.tempo = 120.0
    return timeline


def _choose(timeline, clip, label):
    item = timeline.piano_roll._clip_items[clip.id]
    spot = timeline.piano_roll.mapFromScene(item.sceneBoundingRect().center())
    done = dict(timeline.clip_menu_actions(spot))[label]()
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    return done


def _aligned_from(qapp, path, start):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(path, start)
    said = []
    timeline.clip_edit_status.connect(said.append)
    _choose(timeline, clip, "Align Clip to the Beat")
    return timeline, timeline.model.clip_by_id(clip.id), said


@pytest.mark.parametrize("starting_beat", [3.0, 3.5, 5.75, 10.25])
@pytest.mark.parametrize("late_seconds", [0.04, -0.035])
def test_a_phrase_dropped_slightly_off_keeps_the_beat_it_was_dropped_on(
    qapp, tmp_path, edited_clips_folder, starting_beat, late_seconds
):
    phrase = _sung_phrase(tmp_path / "phrase.wav")
    chosen = starting_beat * BEAT_SECONDS

    _timeline_used, clip, _said = _aligned_from(
        qapp, phrase, chosen + late_seconds
    )

    assert clip.start == pytest.approx(chosen, abs=0.012)


def test_a_phrase_is_never_moved_more_than_half_a_grid_step(
    qapp, tmp_path, edited_clips_folder
):
    phrase = _sung_phrase(tmp_path / "phrase.wav")

    for late_seconds in np.linspace(-0.24, 0.24, 9):
        timeline, clip, _said = _aligned_from(qapp, phrase, 2.0 + late_seconds)
        grid = BEAT_SECONDS / timeline.current_grid_divisions()

        assert abs(clip.start - (2.0 + late_seconds)) <= grid / 2 + 0.012
        steps = clip.start / grid
        assert steps == pytest.approx(round(steps), abs=0.1)


def test_a_clip_can_be_moved_one_beat_later_and_earlier_and_undone(
    qapp, tmp_path, edited_clips_folder
):
    phrase = _sung_phrase(tmp_path / "phrase.wav")
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(phrase, 2.0)
    said = []
    timeline.clip_edit_status.connect(said.append)

    assert _choose(timeline, clip, "Move Clip One Beat Later")
    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(2.5)
    assert "one beat later" in said[-1]

    assert _choose(timeline, clip, "Move Clip One Beat Earlier")
    assert _choose(timeline, clip, "Move Clip One Beat Earlier")
    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(1.5)
    assert "one beat earlier" in said[-1]

    timeline.undo()
    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(2.0)


def test_a_clip_at_the_start_of_the_song_does_not_move_earlier(
    qapp, tmp_path, edited_clips_folder
):
    phrase = _sung_phrase(tmp_path / "phrase.wav")
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(phrase, 0.2)
    said = []
    timeline.clip_edit_status.connect(said.append)

    assert not _choose(timeline, clip, "Move Clip One Beat Earlier")

    assert timeline.model.clip_by_id(clip.id).start == pytest.approx(0.2)
    assert "cannot move earlier" in said[-1]
