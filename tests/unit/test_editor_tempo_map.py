import sys

import numpy as np
import pretty_midi
import pytest
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.midi.tempo_map import TempoMap
from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.viewmodels.piano_roll_vm import PianoRollViewModel
from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll
from korshmer.utils.tempo import midi_file_tempo

SONG = TempoMap((0.0, 30.0), (100.0, 120.0))


def _song_midi(beats=(0, 10, 49.5, 50, 61.25, 90)):
    pm = SONG.new_pretty_midi()
    inst = pretty_midi.Instrument(0)
    for beat in beats:
        start = SONG.seconds_at(beat)
        inst.notes.append(pretty_midi.Note(90, 60, start, start + 0.2))
    pm.instruments.append(inst)
    return pm


def _view_model():
    vm = PianoRollViewModel()
    vm.tempo = 120.0
    vm.tempo_map = SONG
    return vm


def test_snapping_follows_the_tempo_of_each_section():
    vm = _view_model()
    slow_beat = vm.time_to_x(SONG.seconds_at(7))
    fast_step = vm.time_to_x(SONG.seconds_at(60.25))
    assert vm.snap_to_grid(slow_beat + 5, 4) == pytest.approx(slow_beat)
    assert vm.snap_to_grid(fast_step - 4, 4) == pytest.approx(fast_step)
    assert vm.snap_to_beat(fast_step) == pytest.approx(
        vm.time_to_x(SONG.seconds_at(60))
    )
    assert vm.snap_to_grid(-50.0, 4) == 0.0
    vm.grid_offset = 0.15
    shifted = vm.snap_to_grid(vm.time_to_x(31.0), 1)
    beat = SONG.beats_at(shifted / vm.pixels_per_second) - SONG.beats_at(0.15)
    assert beat == pytest.approx(round(beat))


def test_a_uniform_or_missing_map_keeps_the_plain_grid():
    vm = PianoRollViewModel()
    plain = vm.snap_to_grid(133.0, 4)
    assert vm.mapped_lines(0, 500, 4) is None
    vm.tempo_map = TempoMap.uniform(120.0)
    assert not vm.follows_tempo_map
    assert vm.snap_to_grid(133.0, 4) == plain
    assert vm.mapped_lines(0, 500, 4) is None


def test_grid_lines_are_spaced_by_the_local_tempo():
    vm = _view_model()
    lines = vm.mapped_lines(0, vm.time_to_x(40.0), 1)
    xs = np.array([x for x, _ in lines])
    index = [i for _, i in lines]
    assert index == list(range(len(index)))
    slow = np.diff(xs[xs < vm.time_to_x(29.0)])
    fast = np.diff(xs[xs > vm.time_to_x(30.5)])
    assert slow == pytest.approx(60.0, abs=1e-6)
    assert fast == pytest.approx(50.0, abs=1e-6)
    fine = vm.mapped_lines(0, vm.time_to_x(2.0), 4)
    assert [i for _, i in fine[:5]] == [0, None, None, None, 1]


def test_the_grid_painter_draws_a_mapped_grid_without_errors():
    roll = InteractivePianoRoll()
    roll.load_notes(_song_midi())
    assert roll.vm.follows_tempo_map
    plain = InteractivePianoRoll()
    plain.load_notes(None)
    images = []
    for view in (roll, plain):
        image = QImage(800, 300, QImage.Format_ARGB32)
        image.fill(0xFFFFFFFF)
        painter = QPainter(image)
        view.render(painter)
        painter.end()
        images.append(image)
    assert images[0] != images[1]


def test_the_editor_model_keeps_the_map_through_load_save_and_state(tmp_path):
    model = EditorModel(tempo=120.0)
    model.load_midi(_song_midi(beats=(0, 10, 50, 200)))
    assert model.tempo_map.bpms == pytest.approx(SONG.bpms, rel=1e-4)

    path = tmp_path / "saved.mid"
    model.to_pretty_midi().write(str(path))
    saved = TempoMap.from_pretty_midi(pretty_midi.PrettyMIDI(str(path)))
    assert saved.bpms == pytest.approx(SONG.bpms, rel=1e-4)
    assert saved.starts == pytest.approx(SONG.starts, abs=0.003)
    assert midi_file_tempo(str(path)) == pytest.approx(120.0, abs=0.05)

    restored = EditorModel().load_state(model.to_dict())
    assert restored.tempo_map.bpms == pytest.approx(SONG.bpms, rel=1e-4)
    assert EditorModel().load_state({"tempo_map": "broken"}).tempo_map is None

    forced = model.to_pretty_midi(tempo=90.0)
    assert list(forced.get_tempo_changes()[1]) == pytest.approx([90.0])

    plain = pretty_midi.PrettyMIDI(initial_tempo=97.0)
    plain.instruments.append(pretty_midi.Instrument(0))
    model.load_midi(plain)
    assert model.tempo_map is None
    assert len(model.to_pretty_midi().get_tempo_changes()[1]) == 1


def test_quantize_uses_the_local_grid_and_a_tempo_change_stretches_the_map():
    model = EditorModel(tempo=120.0)
    model.load_midi(_song_midi(beats=(7.1, 60.3)))
    ids = [note.id for note in model.notes]
    model.quantize_notes(ids, 0.125, 1.0, 0.0)
    starts = sorted(note.start for note in model.notes)
    assert model.tempo_map.beats_at(np.array(starts)) == pytest.approx(
        [7.0, 60.25], abs=0.01
    )
    model.scale_note_times(2.0)
    assert model.tempo_map.bpms == pytest.approx((50.0, 60.0), rel=1e-4)
    assert model.tempo_map.beats_at(
        np.array(sorted(n.start for n in model.notes))
    ) == pytest.approx([7.0, 60.25], abs=0.01)


def test_the_piano_roll_snaps_dragged_notes_on_the_mapped_grid():
    roll = InteractivePianoRoll()
    roll.grid_divisions = 4
    roll.load_notes(_song_midi())
    item = next(iter(roll._note_items.values()))
    target = roll.vm.time_to_x(SONG.seconds_at(60.25))
    assert item.snap_with is not None
    assert item._snap_x(target + 3.0, 1.0) == pytest.approx(target)
    old = roll.vm.tempo_map
    roll.set_tempo(roll.tempo * 2)
    assert roll.vm.tempo_map.bpms == pytest.approx(
        tuple(2 * b for b in old.bpms), rel=1e-6
    )


def test_file_tempo_is_still_the_first_one_for_ordinary_files(tmp_path):
    path = tmp_path / "plain.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=97.6)
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(90, 60, 0.0, 1.0))
    pm.instruments.append(inst)
    pm.write(str(path))
    assert midi_file_tempo(str(path)) == pytest.approx(97.6, abs=0.01)
