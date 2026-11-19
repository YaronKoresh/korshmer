import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from korshmer import midi_algorithms
from korshmer.audio_processing import clip_polish, clip_tempo
from korshmer.audio_processing.stem_session import StemSession
from korshmer.midi.tempo_map import TempoMap
from korshmer.mixins.actions import editor as editor_actions
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.material.length_box import (
    LengthPicker,
    clock_text,
    minutes_from_text,
)

RATE = 44100


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def fixed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_polish, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / clip_polish.POLISHED_FOLDER


class _Studio(ProjectManagementActionsMixin, EditorActionsMixin):
    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.arranger.btn_snap.setChecked(False)
        self.arranger.btn_even_bpm.setChecked(False)
        self.current_session = None


def _click_loop(path, bpm=120.0, seconds=16):
    audio = np.zeros(int(RATE * seconds), dtype=np.float32)
    click = np.hanning(2000) * np.sin(np.arange(2000) * 0.2) * 0.4
    beat = 60.0 / bpm
    for index in range(int(seconds / beat) - 1):
        first = int(index * beat * RATE)
        audio[first : first + 2000] += click
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, RATE)
    return str(path)


def _click_beats(clip, bpm):
    data, rate = sf.read(clip.path, dtype="float32")
    loud = np.abs(data) > 0.05
    starts = np.flatnonzero(loud[1:] & ~loud[:-1]) / rate
    firsts = starts[np.insert(np.diff(starts) > 0.1, 0, True)]
    return (clip.start - clip.offset + firsts) * bpm / 60.0


def _timeline(qapp, tempo=120.0):
    timeline = ArrangerTimeline()
    timeline.resize(1100, 600)
    timeline.show()
    qapp.processEvents()
    timeline.btn_snap.setChecked(False)
    timeline.btn_even_bpm.setChecked(False)
    timeline.piano_roll.set_tempo(tempo)
    return timeline


def _change_tempo(timeline, bpm):
    timeline.piano_roll.set_tempo(bpm)
    timeline.stretch_clips_to_tempo()
    timeline.wait_for_clip_edits()
    QApplication.processEvents()
    QApplication.processEvents()


def test_a_clip_is_stretched_so_its_beats_stay_on_the_beats_at_a_new_tempo(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    recording = _click_loop(tmp_path / "loop.wav")
    clip = timeline.add_audio_clip(recording, 2.0)
    timeline.wait_for_waveforms()

    _change_tempo(timeline, 150.0)

    clip = timeline.model.clip_by_id(clip.id)
    beats = _click_beats(clip, 150.0)
    assert clip.path != recording
    assert clip.duration == pytest.approx(16.0 * 120.0 / 150.0, rel=0.005)
    assert np.abs(beats - np.round(beats)).max() < 0.08
    assert beats[0] == pytest.approx(4.0, abs=0.08)


def test_a_cut_clip_keeps_the_same_musical_part_after_a_tempo_change(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_click_loop(tmp_path / "loop.wav"), 2.0)
    timeline.wait_for_waveforms()
    timeline.model.trim_clip(clip.id, 4.0, 10.0)
    timeline.piano_roll.load_model()

    _change_tempo(timeline, 100.0)

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.start == pytest.approx(4.8)
    assert clip.offset == pytest.approx(2.4, rel=0.005)
    assert clip.duration == pytest.approx(7.2, rel=0.005)


def test_returning_to_the_recorded_tempo_plays_the_untouched_recording_again(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    recording = _click_loop(tmp_path / "loop.wav")
    clip = timeline.add_audio_clip(recording, 2.0)
    timeline.wait_for_waveforms()

    _change_tempo(timeline, 150.0)
    _change_tempo(timeline, 90.0)
    _change_tempo(timeline, 120.0)

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.path == recording
    assert (clip.start, clip.duration) == pytest.approx((2.0, 16.0))
    assert not fixed_files.exists() or not list(fixed_files.iterdir())


def test_several_tempo_changes_always_stretch_from_the_recording_itself(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    recording = _click_loop(tmp_path / "loop.wav")
    clip = timeline.add_audio_clip(recording, 0.0)
    timeline.wait_for_waveforms()

    _change_tempo(timeline, 140.0)
    _change_tempo(timeline, 160.0)

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.tempo_base == (recording, 120.0)
    assert clip.duration == pytest.approx(16.0 * 120.0 / 160.0, rel=0.005)
    assert len(list(fixed_files.iterdir())) == 1


def test_stepping_back_after_a_tempo_change_keeps_the_stretched_clip(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    clip = timeline.add_audio_clip(_click_loop(tmp_path / "loop.wav"), 2.0)
    timeline.wait_for_waveforms()
    timeline.model.snapshot()
    timeline.model.clip_by_id(clip.id).start = 6.0
    timeline.piano_roll.load_model()

    _change_tempo(timeline, 150.0)
    stretched = timeline.model.clip_by_id(clip.id).path
    timeline.undo()

    restored = timeline.model.clip_by_id(clip.id)
    assert restored.path == stretched
    assert restored.start == pytest.approx(1.6)
    assert restored.duration == pytest.approx(12.8, rel=0.005)


def test_a_tempo_too_far_from_the_recording_leaves_the_clip_speed_and_says_so(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    recording = _click_loop(tmp_path / "loop.wav")
    clip = timeline.add_audio_clip(recording, 2.0)
    said = []
    timeline.clip_edit_status.connect(said.append)

    _change_tempo(timeline, 290.0)

    clip = timeline.model.clip_by_id(clip.id)
    assert clip.path == recording
    assert clip.duration == pytest.approx(16.0)
    assert "too far" in said[-1]


def test_clips_are_not_stretched_while_the_song_itself_changes_tempo(
    qapp, tmp_path, fixed_files
):
    timeline = _timeline(qapp)
    recording = _click_loop(tmp_path / "loop.wav")
    clip = timeline.add_audio_clip(recording, 2.0)
    timeline.model.tempo_map = TempoMap((0.0, 8.0), (120.0, 126.0))

    _change_tempo(timeline, 150.0)

    assert timeline.model.clip_by_id(clip.id).path == recording


def test_a_saved_song_remembers_the_recording_so_later_tempo_changes_still_follow(
    qapp, tmp_path, fixed_files
):
    studio = _Studio(tmp_path / "app")
    timeline = studio.arranger
    timeline.piano_roll.set_tempo(120.0)
    clip = timeline.add_audio_clip(
        _click_loop(tmp_path / "desk" / "loop.wav"), 2.0
    )
    timeline.wait_for_waveforms()
    _change_tempo(timeline, 150.0)
    song = studio.project_manager.song_path_for("Follow")
    assert studio._save_song_to(song)

    later = _Studio(tmp_path / "app")
    assert later.open_song_file(song)
    reopened = later.arranger.model.clip_by_id(clip.id)
    kept_recording = song.parent / "Clips" / "loop.wav"
    assert reopened.tempo_base == (str(kept_recording), 120.0)

    _change_tempo(later.arranger, 120.0)
    assert later.arranger.model.clip_by_id(clip.id).path == str(kept_recording)


def _sparse_rhythm(seed, bpm=104.0, kept=95, stray=400, seconds=150.0):
    rng = np.random.default_rng(seed)
    step = 60.0 / bpm / 2.0
    on_grid = rng.choice(np.arange(0.0, seconds, step), kept, replace=False)
    on_grid = on_grid + rng.normal(0.0, 0.008, kept)
    times = np.sort(np.concatenate([on_grid, rng.uniform(0.0, seconds, stray)]))
    return times, np.ones_like(times)


def test_a_faint_rhythm_is_found_when_separate_sections_agree_on_its_tempo():
    times, weights = _sparse_rhythm(seed=4)

    assert clip_tempo._clear_tempo(times, weights, 120.0) is None
    tempo, _certainty = clip_tempo.own_tempo(times, weights, 120.0)
    assert tempo == pytest.approx(104.0, abs=0.1)


def test_sounds_without_any_rhythm_are_not_given_a_tempo():
    accepted = 0
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        times = np.sort(rng.uniform(0.0, 150.0, 500))
        accepted += (
            clip_tempo.own_tempo(times, rng.uniform(0.2, 1.0, 500), 120.0)
            is not None
        )

    assert accepted == 0


def test_a_short_clip_needs_a_clear_rhythm_to_be_given_a_tempo():
    times, weights = _sparse_rhythm(seed=4)
    first_minute = times < 60.0

    assert (
        clip_tempo.own_tempo(times[first_minute], weights[first_minute], 120.0)
        is None
    )


def _melody_file(path):
    song = pretty_midi.PrettyMIDI()
    lead = pretty_midi.Instrument(program=0)
    lead.notes.append(pretty_midi.Note(90, 60, 0.0, 0.5))
    song.instruments.append(lead)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    song.write(str(path))
    return str(path)


def test_edited_notes_of_an_open_song_are_saved_in_that_song_folder(
    qapp, tmp_path
):
    studio = _Studio(tmp_path / "app")
    song = studio.project_manager.song_path_for("Ballad")
    assert studio._save_song_to(song)

    saved = midi_algorithms._save_modified_midi(
        pretty_midi.PrettyMIDI(_melody_file(tmp_path / "in" / "tune.mid")),
        str(tmp_path / "in" / "tune.mid"),
        "louder",
    )

    assert Path(saved).parent == song.parent / "MIDI"
    assert Path(editor_actions._layers_dir()) == song.parent / "MIDI"


def test_each_song_keeps_its_own_edited_notes(qapp, tmp_path):
    studio = _Studio(tmp_path / "app")
    first = studio.project_manager.song_path_for("First")
    second = studio.project_manager.song_path_for("Second")
    tune = _melody_file(tmp_path / "in" / "tune.mid")

    studio._save_song_to(first)
    midi_algorithms._save_modified_midi(pretty_midi.PrettyMIDI(tune), tune, "a")
    studio._save_song_to(second)
    midi_algorithms._save_modified_midi(pretty_midi.PrettyMIDI(tune), tune, "b")

    assert [path.name for path in (first.parent / "MIDI").iterdir()] == [
        "tune_a.mid"
    ]
    assert [path.name for path in (second.parent / "MIDI").iterdir()] == [
        "tune_b.mid"
    ]


def test_without_an_open_song_edited_notes_go_to_the_shared_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(midi_algorithms, "DATA_DIR", tmp_path / "data")
    tune = _melody_file(tmp_path / "in" / "tune.mid")

    saved = midi_algorithms._save_modified_midi(
        pretty_midi.PrettyMIDI(tune), tune, "x"
    )

    assert Path(saved).parent == tmp_path / "data" / "Processed_MIDI"


def _converted_song(folder):
    stems = folder / "Stems"
    stems.mkdir(parents=True)
    vocals = stems / "vocals.wav"
    sf.write(str(vocals), np.zeros(RATE, dtype=np.float32) + 0.1, RATE)
    notes = _melody_file(folder / "MIDI" / "bass.mid")
    session = StemSession(
        source_path=str(folder / "song.mp3"),
        output_dir=str(folder),
        stems={"vocals": str(vocals)},
        midi_files=[notes],
    )
    session.save()
    return session


def test_saving_a_converted_song_elsewhere_takes_its_stems_and_notes_along(
    qapp, tmp_path
):
    studio = _Studio(tmp_path / "app")
    converted = (
        studio.project_manager.saved_data_dir / "song" / "2026_11_19__10_00_00"
    )
    studio.current_session = _converted_song(converted)
    studio.arranger.add_audio_clip(str(converted / "Stems" / "vocals.wav"), 0.0)
    song = studio.project_manager.song_path_for("My version")

    assert studio._save_song_to(song)

    kept = song.parent / "Conversion"
    assert (kept / "Stems" / "vocals.wav").is_file()
    assert (kept / "MIDI" / "bass.mid").is_file()
    assert (
        Path(studio.arranger.model.clips[0].path)
        == kept / "Stems" / "vocals.wav"
    )
    assert Path(studio.current_session.output_dir) == kept
    assert not (song.parent / "Clips").exists()
    assert (converted / "Stems" / "vocals.wav").is_file()

    later = _Studio(tmp_path / "app")
    assert later.open_song_file(song)
    assert Path(later.current_session.output_dir) == kept


def test_a_song_saved_inside_its_conversion_folder_is_not_copied(
    qapp, tmp_path
):
    studio = _Studio(tmp_path / "app")
    converted = (
        studio.project_manager.saved_data_dir / "song" / "2026_11_19__10_00_00"
    )
    studio.current_session = _converted_song(converted)

    assert studio.save_project_action()

    assert not (converted / "Conversion").exists()
    assert list(converted.glob("*.kproj"))


def test_the_length_is_shown_and_typed_as_minutes_and_seconds(qapp):
    picker = LengthPicker(0.5, 120.0, 3.0)

    assert picker.box.text() == "3:00"
    for typed, minutes in (
        ("2:30", 2.5),
        ("1:05", 65 / 60),
        ("2.5", 2.5),
        ("90s", 1.5),
    ):
        picker.box.lineEdit().setText(typed)
        picker.box.interpretText()
        assert picker.box.value() == pytest.approx(minutes)
    assert clock_text(2.5) == "2:30"
    assert minutes_from_text("soon") is None


def test_the_length_arrows_move_in_quarter_minute_steps(qapp):
    picker = LengthPicker(0.5, 120.0, 3.0)

    picker.box.stepUp()
    assert picker.box.text() == "3:15"
    picker.box.stepDown()
    picker.box.stepDown()
    assert picker.box.text() == "2:45"


def test_quick_length_buttons_set_the_length_and_show_which_one_is_chosen(qapp):
    picker = LengthPicker(0.5, 120.0, 3.0)

    assert [button.text() for button in picker.quick.values()] == [
        "0:30",
        "1:00",
        "1:30",
        "2:00",
        "3:00",
        "4:00",
        "5:00",
    ]
    picker.quick[90].click()
    assert picker.box.value() == pytest.approx(1.5)
    assert [
        seconds
        for seconds, button in picker.quick.items()
        if button.isChecked()
    ] == [90]
    picker.box.setValue(1.75)
    assert not any(button.isChecked() for button in picker.quick.values())


def test_the_length_shows_how_many_bars_it_gives_at_the_song_tempo(qapp):
    picker = LengthPicker(0.5, 120.0, 2.0, tempo=120.0)

    assert picker.bar_count() == 60
    assert "60 bars" in picker.bars.text()
    picker.set_tempo(90.0)
    assert picker.bar_count() == 45
    assert "45 bars at 90 BPM" in picker.bars.text()


def test_a_length_outside_the_allowed_range_is_brought_back_inside(qapp):
    picker = LengthPicker(0.5, 120.0, 3.0)

    picker.box.lineEdit().setText("0:10")
    picker.box.interpretText()

    assert picker.box.text() == "0:30"


def test_turning_the_length_off_also_turns_off_its_box(qapp):
    picker = LengthPicker(0.5, 120.0, 3.0)

    picker.setEnabled(False)

    assert not picker.box.isEnabled()
