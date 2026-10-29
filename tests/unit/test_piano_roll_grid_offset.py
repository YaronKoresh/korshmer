import pretty_midi
import pytest

from korshmer.ui.viewmodels.editor_vm import EditorModel
from korshmer.ui.viewmodels.piano_roll_vm import PianoRollViewModel
from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll


def _vm(offset=0.0, tempo=120.0):
    vm = PianoRollViewModel()
    vm.tempo = tempo
    vm.pixels_per_second = 100.0
    vm.grid_offset = offset
    return vm


def test_snap_without_offset_is_unchanged():
    vm = _vm()

    assert vm.snap_to_grid(40.0, 1) == pytest.approx(50.0)
    assert vm.snap_to_beat(80.0) == pytest.approx(100.0)


def test_snap_follows_the_offset():
    vm = _vm(offset=0.35)

    assert vm.snap_to_grid(40.0, 1) == pytest.approx(35.0)
    assert vm.snap_to_grid(70.0, 1) == pytest.approx(85.0)
    assert vm.snap_to_beat(80.0) == pytest.approx(85.0)


def test_snap_never_goes_before_zero():
    vm = _vm(offset=0.35)

    assert vm.snap_to_grid(0.0, 1) >= 0.0
    assert vm.snap_to_grid(0.0, 1) == pytest.approx(35.0)


def test_tempo_change_moves_the_offset_with_the_notes():
    vm = _vm(offset=0.4)

    vm.compute_tempo_change(60.0)

    assert vm.grid_offset == pytest.approx(0.8)


def test_model_quantize_uses_the_offset():
    model = EditorModel()
    track = model.add_track("T")
    note = model.add_note(track.id, 60, 0.4, 0.2, 90)
    note_id = note.id if note is not None else model.notes[0].id

    model.quantize_notes([note_id], 0.5, 1.0, grid_offset=0.35)

    assert model.notes[0].start == pytest.approx(0.35)


def test_model_saves_and_loads_the_offset():
    model = EditorModel()
    model.grid_offset = 0.35

    loaded = EditorModel.from_dict(model.to_dict())

    assert loaded.grid_offset == pytest.approx(0.35)
    assert EditorModel.from_dict({}).grid_offset == 0.0
    assert EditorModel.from_dict({"grid_offset": -3}).grid_offset == 0.0
    assert EditorModel.from_dict({"grid_offset": "x"}).grid_offset == 0.0


class TestPianoRollGridOffset:
    @pytest.fixture(autouse=True)
    def _qt_app(self, qapp):
        return qapp

    def _song(self):
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(0)
        inst.notes.append(pretty_midi.Note(60, 72, 0.0, 0.3))
        for bar in range(4):
            at = 0.5 + 2.0 * bar
            inst.notes.append(pretty_midi.Note(110, 45, at, at + 0.4))
            for k in range(1, 4):
                inst.notes.append(
                    pretty_midi.Note(60, 72, at + 0.5 * k, at + 0.5 * k + 0.4)
                )
        pm.instruments.append(inst)
        return pm

    def test_detect_grid_offset_finds_the_first_bar_line(self):
        roll = InteractivePianoRoll()

        offset = roll.detect_grid_offset(self._song())

        assert offset == pytest.approx(0.5, abs=0.01)
        assert roll.grid_offset == pytest.approx(offset)
        assert roll.model.grid_offset == pytest.approx(offset)

    def test_snap_x_follows_the_detected_offset(self):
        roll = InteractivePianoRoll()
        roll.set_grid_offset(0.35)

        assert roll.snap_x(40.0) == pytest.approx(0.35 * roll.pixels_per_second)

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "x", None])
    def test_invalid_offset_falls_back_to_zero(self, bad):
        roll = InteractivePianoRoll()
        roll.set_grid_offset(0.35)

        roll.set_grid_offset(bad)

        assert roll.grid_offset == 0.0

    def test_background_paints_with_an_offset(self):
        roll = InteractivePianoRoll()
        roll.resize(600, 300)
        roll.set_grid_offset(0.35)

        image = roll.viewport().grab()

        assert not image.isNull()
