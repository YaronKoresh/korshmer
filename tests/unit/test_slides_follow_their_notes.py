import json
import sys

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer import audio_processors, constants
from korshmer.midi.tempo_map import TempoMap
from korshmer.midi_generation.composer import ComposeOptions, compose
from korshmer.mixins.actions.editor import EditorActionsMixin
from korshmer.ui.managers.context_menu import ContextMenuManager
from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

BPM = 120.0
SUB_BASS = 87
WIDE_RANGE = 12
UNITS_PER_SEMITONE = 8192 / WIDE_RANGE
PLAIN, GLIDING, FALLING, LAST = 33, 38, 36, 43
GLIDE_FROM = -5
FALL_TO = -12
GLIDE_SECONDS = 0.25
FALL_SECONDS = 0.5
TICK = 60.0 / BPM / 220
JUST_AFTER = 1e-6


def _declare_wide_range(inst):
    inst.control_changes += [
        pretty_midi.ControlChange(101, 0, 0.0),
        pretty_midi.ControlChange(100, 0, 0.0),
        pretty_midi.ControlChange(6, WIDE_RANGE, 0.01),
    ]


def _ramp(inst, start, end, first, last, steps=10):
    for k in range(steps + 1):
        semitones = first + (last - first) * k / steps
        inst.pitch_bends.append(
            pretty_midi.PitchBend(
                round(semitones * UNITS_PER_SEMITONE),
                start + (end - start) * k / steps,
            )
        )


def _bass_line():
    bass = pretty_midi.Instrument(SUB_BASS, name="Sub")
    bass.notes += [
        pretty_midi.Note(100, PLAIN, 0.0, 1.0),
        pretty_midi.Note(100, GLIDING, 1.0, 2.0),
        pretty_midi.Note(100, FALLING, 3.0, 4.0),
        pretty_midi.Note(100, LAST, 5.0, 6.0),
    ]
    _declare_wide_range(bass)
    _ramp(bass, 1.0, 1.0 + GLIDE_SECONDS, GLIDE_FROM, 0)
    _ramp(bass, 4.0 - FALL_SECONDS, 4.0, 0, FALL_TO)
    bass.pitch_bends.append(pretty_midi.PitchBend(0, 4.25))
    return bass


def _song(*instruments):
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    midi.instruments.extend(instruments)
    return midi


def _model():
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(_bass_line()), replace=True)
    return model


def _note(model, pitch):
    return next(n for n in model.notes if n.pitch == pitch)


def _exported(model):
    (inst,) = model.to_pretty_midi().instruments
    return inst


def _bends(model):
    return [(b.time, b.pitch) for b in _exported(model).pitch_bends]


def _semitones_at(model, time):
    value = 0
    for when, pitch in _bends(model):
        if when > time + 1e-9:
            break
        value = pitch
    return value / UNITS_PER_SEMITONE


def _same_bends(found, wanted):
    assert [v for _t, v in found] == [v for _t, v in wanted]
    assert [t for t, _v in found] == pytest.approx(
        [t for t, _v in wanted], abs=1e-9
    )
    return True


def _bent_times(model, start, end):
    return [t for t, v in _bends(model) if v and start <= t <= end]


def _glide_is_at(model, start, length=GLIDE_SECONDS):
    assert _semitones_at(model, start) == pytest.approx(GLIDE_FROM, abs=0.01)
    assert _semitones_at(model, start + length / 2) == pytest.approx(
        GLIDE_FROM / 2, abs=0.3
    )
    assert _semitones_at(model, start + length + JUST_AFTER) == 0
    return True


def _fall_ends_at(model, end, length=FALL_SECONDS):
    assert _semitones_at(model, end - length) == 0
    assert _semitones_at(model, end - length / 2) == pytest.approx(
        FALL_TO / 2, abs=0.7
    )
    assert _semitones_at(model, end + JUST_AFTER) == pytest.approx(
        FALL_TO, abs=0.01
    )
    return True


def _roll():
    roll = InteractivePianoRoll()
    roll.load_notes(_song(_bass_line()))
    return roll


def _item(roll, pitch):
    return roll._note_items[_note(roll.model, pitch).id]


def _select(roll, *pitches):
    roll.scene.clearSelection()
    for pitch in pitches:
        _item(roll, pitch).setSelected(True)


def test_a_loaded_song_gives_each_slide_to_the_one_note_it_bends():
    model = _model()

    slid = sorted(n.pitch for n in model.notes if n.slide)

    assert slid == [FALLING, GLIDING]
    assert all(not track.bends for track in model.tracks.values())
    assert _same_bends(
        _bends(model), [(b.time, b.pitch) for b in _bass_line().pitch_bends]
    )


def test_moving_a_sliding_note_in_time_moves_its_slide_and_leaves_none_behind():
    model = _model()

    model.shift_notes([_note(model, GLIDING).id], 1.0)

    assert _glide_is_at(model, 2.0)
    assert _bent_times(model, 0.0, 1.99) == []


def test_dragging_a_sliding_note_in_the_piano_roll_takes_its_slide_along():
    roll = _roll()
    item = _item(roll, GLIDING)
    pps = roll.vm.pixels_per_second
    roll.set_snap(False)

    item.setPos(2.0 * pps, item.y() - 2 * item.grid_y)
    roll.sync_model_from_scene()

    assert _note(roll.model, GLIDING + 2).start == pytest.approx(2.0)
    assert _glide_is_at(roll.model, 2.0)
    assert _bent_times(roll.model, 0.0, 1.99) == []


def test_deleting_a_sliding_note_in_the_piano_roll_deletes_its_slide():
    roll = _roll()
    _select(roll, GLIDING)

    roll.delete_selected()

    assert _bent_times(roll.model, 0.0, 3.0) == []
    assert _fall_ends_at(roll.model, 4.0)


def test_deleting_every_sliding_note_leaves_a_track_without_any_bend():
    roll = _roll()
    _select(roll, GLIDING, FALLING)

    roll.delete_selected()

    assert _exported(roll.model).pitch_bends == []


def test_a_duplicated_sliding_note_gets_its_own_copy_of_the_slide():
    roll = _roll()
    _select(roll, GLIDING)

    (copy,) = roll.duplicate_selected()
    roll.model.remove_slides([_note(roll.model, GLIDING).id])

    assert copy.start == pytest.approx(2.0)
    assert _glide_is_at(roll.model, 2.0)
    assert _bent_times(roll.model, 0.0, 1.99) == []


def test_a_copied_and_pasted_sliding_note_carries_its_slide_to_the_new_place():
    roll = _roll()
    _select(roll, GLIDING)
    roll.copy_selected()

    (pasted,) = roll.paste(at_time=7.0)
    roll.model.shift_notes([pasted.id], 0.5)

    assert _glide_is_at(roll.model, 1.0)
    assert _glide_is_at(roll.model, 7.5)
    assert _bent_times(roll.model, 6.0, 7.49) == []


def test_quantizing_a_sliding_note_moves_its_slide_onto_the_grid_with_it():
    model = _model()
    note = _note(model, GLIDING)
    model.shift_notes([note.id], 0.57)

    model.quantize_notes([note.id], 0.5)

    assert note.start == pytest.approx(1.5)
    assert _glide_is_at(model, 1.5)
    assert _bent_times(model, 0.0, 1.49) == []


def test_humanizing_a_sliding_note_keeps_its_slide_at_the_new_start():
    model = _model()
    note = _note(model, GLIDING)
    model.shift_notes([note.id], 0.5)

    model.humanize_notes([note.id], 0.05, 0, seed=4)

    assert note.start != pytest.approx(1.5)
    assert _glide_is_at(model, note.start)
    assert _bent_times(model, 0.0, note.start - TICK) == []


def test_nudging_a_sliding_note_by_grid_steps_nudges_its_slide():
    roll = _roll()
    _select(roll, GLIDING)

    roll.nudge_selected(2 * roll.grid_divisions)

    assert _note(roll.model, GLIDING).start == pytest.approx(2.0)
    assert _glide_is_at(roll.model, 2.0)


def test_transposing_a_sliding_note_keeps_the_same_slide_on_the_new_pitch():
    roll = _roll()
    before = _bends(roll.model)
    _select(roll, GLIDING)

    roll.transpose_selected(12)

    assert _note(roll.model, GLIDING + 12).slide
    assert _bends(roll.model) == before


def test_a_longer_note_keeps_a_slide_in_at_its_start_and_a_fall_at_its_end():
    model = _model()
    _note(model, GLIDING).duration = 1.5
    _note(model, FALLING).duration = 1.75

    assert _glide_is_at(model, 1.0)
    assert _fall_ends_at(model, 4.75)
    assert _semitones_at(model, 5.0) == 0


def test_a_note_shortened_below_its_slide_keeps_the_whole_slide_inside_it():
    roll = _roll()
    item = _item(roll, GLIDING)
    pps = roll.vm.pixels_per_second

    item.setRect(0, 0, 0.1 * pps, item.rect().height())
    roll.sync_model_from_scene()

    assert _glide_is_at(roll.model, 1.0, 0.1)
    assert _bent_times(roll.model, 1.1, 3.0) == []


def test_moving_the_start_of_a_note_moves_a_slide_in_but_not_a_fall_at_the_end():
    model = _model()
    for pitch in (GLIDING, FALLING):
        note = _note(model, pitch)
        note.start += 0.25
        note.duration -= 0.25

    assert _glide_is_at(model, 1.25)
    assert _fall_ends_at(model, 4.0)


def test_slides_stretch_with_their_notes_when_the_notes_are_stretched():
    roll = _roll()

    roll.stretch_notes(2.0)

    assert _glide_is_at(roll.model, 2.0, 2 * GLIDE_SECONDS)
    assert _fall_ends_at(roll.model, 8.0, 2 * FALL_SECONDS)


def test_slides_follow_a_tempo_change_and_the_undo_history_follows_too():
    model = _model()
    model.snapshot()
    model.shift_notes([_note(model, GLIDING).id], 0.5)

    model.follow_tempo(0.5)
    assert _glide_is_at(model, 0.75, GLIDE_SECONDS / 2)
    model.undo()

    assert _glide_is_at(model, 0.5, GLIDE_SECONDS / 2)


def test_slides_stay_on_their_notes_when_even_bpm_is_turned_on_and_off():
    model = _model()
    model.tempo_map = TempoMap((0.0, 2.0), (120.0, 90.0))

    model.normalize_tempo(lambda clip, tempo_map, bpm: (clip.path, 0.0))
    evened = _note(model, FALLING)
    assert evened.end != pytest.approx(4.0, abs=0.05)
    assert _semitones_at(model, evened.end) == pytest.approx(FALL_TO, abs=0.01)
    assert _semitones_at(model, evened.end - 0.6 * evened.duration) == 0
    model.restore_tempo()

    assert _glide_is_at(model, 1.0)
    assert _fall_ends_at(model, 4.0)


def test_shifting_by_beats_under_a_changing_tempo_moves_the_slide_in_beats():
    model = _model()
    model.tempo_map = TempoMap((0.0, 2.0), (120.0, 60.0))
    note = _note(model, GLIDING)

    model.shift_notes_by_beats([note.id], 2)

    assert note.start == pytest.approx(2.0)
    assert note.duration == pytest.approx(2.0)
    assert _glide_is_at(model, 2.0, 2 * GLIDE_SECONDS)


def test_a_slide_squeezed_into_a_few_ticks_still_ends_at_centre_in_a_midi_file(
    tmp_path,
):
    bass = pretty_midi.Instrument(SUB_BASS)
    bass.notes.append(pretty_midi.Note(100, 40, 1.0, 2.0))
    _declare_wide_range(bass)
    _ramp(bass, 1.0, 1.25, 5, 0, steps=60)
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(bass), replace=True)

    model.follow_tempo(0.1)
    model.to_pretty_midi().write(str(tmp_path / "fast.mid"))
    (saved,) = pretty_midi.PrettyMIDI(str(tmp_path / "fast.mid")).instruments

    heights = [bend.pitch for bend in saved.pitch_bends]
    assert heights[0] > 4 * UNITS_PER_SEMITONE
    assert heights == sorted(heights, reverse=True)
    assert heights[-1] == 0
    assert saved.pitch_bends[-1].time == pytest.approx(0.125, abs=2 * TICK)


def test_undo_and_redo_of_a_move_take_the_slide_back_and_forth():
    roll = _roll()
    _select(roll, GLIDING)
    roll.nudge_selected(2 * roll.grid_divisions)

    roll.undo()
    assert _glide_is_at(roll.model, 1.0)
    assert _bent_times(roll.model, 1.5, 3.0) == []
    roll.redo()

    assert _glide_is_at(roll.model, 2.0)
    assert _bent_times(roll.model, 0.0, 1.99) == []


def test_a_slide_is_silent_while_its_note_overlaps_a_note_already_sounding():
    model = _model()
    note = _note(model, GLIDING)

    model.shift_notes([note.id], -0.5)
    assert _bent_times(model, 0.0, 3.0) == []
    model.shift_notes([note.id], 0.5)

    assert _glide_is_at(model, 1.0)


def test_a_note_that_starts_inside_a_slide_starts_at_centre_pitch():
    model = _model()

    model.shift_notes([_note(model, LAST).id], -3.9)

    assert _semitones_at(model, 1.05) != 0
    assert _semitones_at(model, 1.1) == 0
    assert _bent_times(model, 1.1, 3.0) == []


def test_a_note_placed_right_after_a_falling_note_starts_at_centre_pitch():
    model = _model()

    model.shift_notes([_note(model, LAST).id], -0.95)

    assert _fall_ends_at(model, 4.0)
    assert _semitones_at(model, 4.04) == pytest.approx(FALL_TO, abs=0.01)
    assert _semitones_at(model, 4.05) == 0


def test_a_chord_added_on_a_sliding_note_is_not_bent_by_its_slide():
    model = _model()
    track_id = _note(model, GLIDING).track_id

    model.add_note(track_id, GLIDING + 7, 1.0, 1.0)

    assert _bent_times(model, 0.0, 3.0) == []
    assert _fall_ends_at(model, 4.0)


def _chord_with_a_wheel_move():
    keys = pretty_midi.Instrument(0, name="Keys")
    keys.notes += [
        pretty_midi.Note(90, 60, 0.0, 2.0),
        pretty_midi.Note(90, 64, 0.0, 2.0),
        pretty_midi.Note(90, 67, 3.0, 4.0),
    ]
    keys.pitch_bends += [
        pretty_midi.PitchBend(2000, 0.5),
        pretty_midi.PitchBend(4000, 1.0),
        pretty_midi.PitchBend(0, 1.5),
    ]
    return keys


def test_a_bend_over_several_notes_stays_where_it_was_and_plays_as_before():
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(_chord_with_a_wheel_move()), replace=True)
    wanted = [(0.5, 2000), (1.0, 4000), (1.5, 0)]

    assert not any(n.slide for n in model.notes)
    model.shift_notes([_note(model, 67).id], 2.0)
    restored = EditorModel.from_dict(json.loads(json.dumps(model.to_dict())))

    assert _bends(model) == wanted
    assert _bends(restored) == wanted


def _song_saved_before_slides_belonged_to_notes():
    model = _model()
    saved = model.to_dict()
    bends = _bends(model)
    saved["version"] = 1
    for note in saved["notes"]:
        del note["slide"]
    for track in saved["tracks"]:
        track["bends"] = [list(point) for point in bends]
    return json.loads(json.dumps(saved))


def test_a_song_saved_with_track_bends_plays_the_same_and_its_slides_follow():
    saved = _song_saved_before_slides_belonged_to_notes()

    model = EditorModel.from_dict(saved)
    assert _same_bends(_bends(model), saved["tracks"][0]["bends"])
    model.shift_notes([_note(model, GLIDING).id], 1.0)

    assert _glide_is_at(model, 2.0)
    assert _bent_times(model, 0.0, 1.99) == []


def test_a_saved_song_keeps_each_slide_on_its_note():
    model = _model()
    model.shift_notes([_note(model, GLIDING).id], 1.0)

    restored = EditorModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert _bends(restored) == _bends(model)
    restored.remove_notes([_note(restored, GLIDING).id])

    assert _bent_times(restored, 0.0, 3.0) == []


def test_a_sliding_note_merged_into_a_track_on_import_keeps_its_slide():
    model = EditorModel(tempo=BPM)
    first = pretty_midi.Instrument(SUB_BASS)
    first.notes.append(pretty_midi.Note(100, 30, 8.0, 9.0))
    model.load_midi(_song(first), replace=True, merge_instruments=True)

    model.load_midi(_song(_bass_line()), replace=False, merge_instruments=True)
    model.shift_notes([_note(model, GLIDING).id], 1.0)

    assert len(model.tracks) == 1
    assert _glide_is_at(model, 2.0)


def test_two_tracks_of_one_sound_export_apart_when_a_moved_slide_would_clash():
    model = _model()
    other = model.add_track("Second sub", program=SUB_BASS)
    model.add_note(other.id, 45, 2.0, 1.0)

    model.shift_notes([_note(model, GLIDING).id], 1.0)
    exported = model.to_pretty_midi().instruments

    assert len(exported) == 2
    plain = next(i for i in exported if not i.pitch_bends)
    assert [n.pitch for n in plain.notes] == [45]


def _phonk(seed):
    composed = compose(
        ComposeOptions(
            "A", "Minor", bpm=140.0, style="Phonk", seed=seed, bars=16
        )
    )
    midi = pretty_midi.PrettyMIDI(initial_tempo=140.0)
    midi.instruments.extend(composed.tracks)
    return midi


def _all_bends(midi):
    return sorted(
        (b.time, b.pitch) for inst in midi.instruments for b in inst.pitch_bends
    )


@pytest.mark.parametrize("seed", (1, 2, 3, 4))
def test_generated_phonk_slides_arrive_on_their_notes_and_export_unchanged(
    seed,
):
    generated = _phonk(seed)
    model = EditorModel(tempo=140.0)

    model.load_midi(generated, replace=True, merge_instruments=True)
    exported = _all_bends(model.to_pretty_midi(tempo=140.0))

    assert sum(1 for n in model.notes if n.slide) > 0
    assert all(not track.bends for track in model.tracks.values())
    assert _same_bends(exported, _all_bends(generated))


def test_moving_a_generated_phonk_slide_note_by_a_bar_moves_its_bends_a_bar():
    model = EditorModel(tempo=140.0)
    model.load_midi(_phonk(3), replace=True)
    note = max((n for n in model.notes if n.slide), key=lambda n: n.start)
    points = note.slide.points(note.start, note.end)
    bar = 240.0 / 140.0

    model.shift_notes([note.id], 4 * bar)
    moved = [
        (b.time, b.pitch)
        for inst in model.to_pretty_midi().instruments
        for b in inst.pitch_bends
        if b.time >= note.start - 1e-9
    ]

    assert [t for t, _v in moved] == pytest.approx(
        [t + 4 * bar for t, _v in points]
    )
    assert any(v for _t, v in moved)


def _picture(roll, pitch):
    image = QImage(200, 40, QImage.Format_ARGB32)
    image.fill(0xFF000000)
    painter = QPainter(image)
    roll.scene.render(
        painter,
        source=_item(roll, pitch).sceneBoundingRect().adjusted(-2, -2, 2, 2),
    )
    painter.end()
    return image


def test_the_piano_roll_marks_sliding_notes_and_names_the_slide_in_the_tooltip():
    roll = _roll()
    with_mark = _picture(roll, GLIDING)
    plain = _picture(roll, PLAIN)

    assert "slides in from -5 st" in _item(roll, GLIDING).toolTip()
    assert "slides out to -12 st" in _item(roll, FALLING).toolTip()
    assert "slide" not in _item(roll, PLAIN).toolTip()
    roll.select_all()
    roll.remove_slide_selected()
    roll.scene.clearSelection()

    assert "slide" not in _item(roll, GLIDING).toolTip()
    assert _picture(roll, GLIDING) != with_mark
    assert _picture(roll, PLAIN) == plain


def test_removing_the_slide_from_selected_notes_keeps_the_notes_and_can_be_undone():
    roll = _roll()
    _select(roll, GLIDING, PLAIN)

    assert roll.selected_slide_count() == 1
    assert roll.remove_slide_selected() == 1

    assert len(roll.model.notes) == 4
    assert _bent_times(roll.model, 0.0, 3.0) == []
    assert _fall_ends_at(roll.model, 4.0)
    roll.undo()
    assert _glide_is_at(roll.model, 1.0)


class _Window(QWidget, EditorActionsMixin):
    def __init__(self):
        super().__init__()
        self.arranger = ArrangerTimeline()


def _menu_entry(menu, label):
    return next(a for a in menu.actions() if a.text() == label)


def test_the_right_click_menu_removes_the_slide_of_the_selected_notes():
    window = _Window()
    window.arranger.load_midi_object(_song(_bass_line()), replace=True)
    roll = window.arranger.piano_roll
    _select(roll, GLIDING)
    menu = ContextMenuManager().create_piano_roll_menu(
        window, (), window.arranger.selected_slide_count()
    )

    _menu_entry(menu, "Remove Slide").trigger()

    assert _note(roll.model, GLIDING).slide is None
    assert _bent_times(roll.model, 0.0, 3.0) == []
    assert _fall_ends_at(roll.model, 4.0)


def test_the_right_click_menu_offers_remove_slide_only_for_notes_that_slide():
    roll = _roll()
    manager = ContextMenuManager()

    _select(roll, PLAIN)
    without = manager.create_piano_roll_menu(
        None, (), roll.selected_slide_count()
    )
    _select(roll, FALLING)
    with_slide = manager.create_piano_roll_menu(
        None, (), roll.selected_slide_count()
    )

    assert not _menu_entry(without, "Remove Slide").isEnabled()
    assert _menu_entry(with_slide, "Remove Slide").isEnabled()


def _render(model, tmp_path, name):
    soundfont = constants.SF3_PATH
    if not soundfont.exists():
        pytest.skip("soundfont missing")
    try:
        from korshmer.core.fluidsynth import FluidSynth

        FluidSynth(lib_dir=str(constants.FLUID_DIR)).close()
    except Exception:
        pytest.skip("FluidSynth library unavailable")
    source = tmp_path / f"{name}.mid"
    model.to_pretty_midi().write(str(source))
    target = tmp_path / f"{name}.wav"
    assert audio_processors.render_midi_to_wav(
        str(source), str(target), str(constants.FLUID_DIR), str(soundfont), BPM
    )
    audio, rate = sf.read(str(target))
    return audio.mean(axis=1), rate


def _heard_pitch(audio, rate, time, window=0.06):
    size = int(window * rate)
    start = max(0, int(time * rate) - size // 2)
    frame = audio[start : start + size]
    frame = (frame - frame.mean()) * np.hanning(len(frame))
    power = np.abs(np.fft.rfft(frame, 2 * size)) ** 2
    lags = np.fft.irfft(power)[: rate // 22]
    lags = lags / lags[0]
    first_dip = int(np.argmax(lags < 0))
    lags[:first_dip] = -1
    lag = int(np.argmax(lags >= 0.9 * lags.max()))
    while lag + 1 < len(lags) and lags[lag + 1] > lags[lag]:
        lag += 1
    before, peak, after = lags[lag - 1 : lag + 2]
    lag += 0.5 * (before - after) / (before - 2 * peak + after)
    return 69 + 12 * np.log2(rate / lag / 440.0)


def _sub_line():
    bass = pretty_midi.Instrument(SUB_BASS, name="Sub")
    bass.notes += [
        pretty_midi.Note(104, 45, 0.0, 1.0),
        pretty_midi.Note(104, 45, 1.0, 2.0),
        pretty_midi.Note(104, 45, 4.0, 5.0),
    ]
    _declare_wide_range(bass)
    _ramp(bass, 1.0, 1.5, -12, 0, steps=40)
    return bass


def test_the_glide_is_heard_on_the_moved_note_and_not_where_the_note_was(
    tmp_path,
):
    model = EditorModel(tempo=BPM)
    model.load_midi(_song(_sub_line()), replace=True)
    sliding = next(n for n in model.notes if n.slide)

    model.shift_notes([sliding.id], 1.5)
    model.shift_notes([n.id for n in model.notes if n.start == 4.0], -3.0)
    audio, rate = _render(model, tmp_path, "moved")

    assert _heard_pitch(audio, rate, 0.5) == pytest.approx(45, abs=0.2)
    assert _heard_pitch(audio, rate, 1.1) == pytest.approx(45, abs=0.2)
    assert _heard_pitch(audio, rate, 1.3) == pytest.approx(45, abs=0.2)
    assert _heard_pitch(audio, rate, 2.56) == pytest.approx(34.4, abs=1.0)
    assert _heard_pitch(audio, rate, 2.75) == pytest.approx(39, abs=0.7)
    assert _heard_pitch(audio, rate, 3.3) == pytest.approx(45, abs=0.2)
