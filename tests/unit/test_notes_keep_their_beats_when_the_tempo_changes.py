import sys

import pytest
from PySide6.QtWidgets import QApplication

from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

BEATS = (0.0, 1.0, 1.5, 2.25, 4.0, 7.75, 16.0, 33.5, 64.0)
LENGTH_IN_BEATS = 0.5


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _roll(tempo=120.0, snap=True):
    roll = InteractivePianoRoll()
    roll.set_tempo(tempo)
    roll.snap_enabled = snap
    track = roll.model.add_track("Lead")
    beat = 60.0 / tempo
    for position in BEATS:
        roll.model.add_note(
            track.id, 60, position * beat, LENGTH_IN_BEATS * beat, 90
        )
    roll.load_model()
    return roll


def _stored_beats(roll):
    tempo = roll.model.tempo
    return sorted(note.start * tempo / 60.0 for note in roll.model.notes)


def _shown_beats(roll):
    return sorted(
        item.x() / roll.vm.beat_width for item in roll._note_items.values()
    )


@pytest.mark.parametrize("tempo", [150.0, 97.0, 133.33, 61.5, 240.0])
def test_every_note_keeps_its_exact_beat_at_a_new_tempo(qapp, tempo):
    roll = _roll()

    roll.set_tempo(tempo)
    roll.sync_model_from_scene()

    assert _stored_beats(roll) == pytest.approx(BEATS, abs=1e-9)
    assert _shown_beats(roll) == pytest.approx(BEATS, abs=1e-6)


def test_notes_between_grid_lines_are_not_pulled_onto_the_grid(qapp):
    roll = _roll()

    roll.set_tempo(150.0)
    roll.sync_model_from_scene()

    assert 2.25 in [round(beat, 9) for beat in _stored_beats(roll)]
    assert 7.75 in [round(beat, 9) for beat in _stored_beats(roll)]


def test_note_lengths_keep_their_length_in_beats(qapp):
    roll = _roll()

    roll.set_tempo(97.0)
    roll.sync_model_from_scene()

    tempo = roll.model.tempo
    lengths = {
        round(note.duration * tempo / 60.0, 9) for note in roll.model.notes
    }
    assert lengths == {LENGTH_IN_BEATS}
    shown = {
        round(item.rect().width() / roll.vm.beat_width, 6)
        for item in roll._note_items.values()
    }
    assert shown == {LENGTH_IN_BEATS}


def test_many_tempo_changes_in_a_row_do_not_let_notes_drift(qapp):
    roll = _roll()

    for step in range(60):
        roll.set_tempo(120.0 + step)
    for step in range(60):
        roll.set_tempo(179.0 - step * 1.37)
    roll.set_tempo(120.0)
    roll.sync_model_from_scene()

    assert _stored_beats(roll) == pytest.approx(BEATS, abs=1e-7)
    assert _shown_beats(roll) == pytest.approx(BEATS, abs=1e-6)


def test_notes_keep_their_beats_with_the_grid_snap_turned_off_too(qapp):
    roll = _roll(snap=False)

    roll.set_tempo(87.5)
    roll.sync_model_from_scene()

    assert _stored_beats(roll) == pytest.approx(BEATS, abs=1e-9)


def test_a_note_dragged_just_before_the_tempo_change_keeps_its_new_place(qapp):
    roll = _roll(snap=False)
    item = next(iter(roll._note_items.values()))
    item.setX(3.0 * roll.vm.beat_width)

    roll.set_tempo(150.0)

    assert 3.0 in [round(beat, 6) for beat in _stored_beats(roll)]


def test_notes_can_still_be_snapped_to_the_new_grid_after_the_change(qapp):
    roll = _roll()
    roll.set_tempo(150.0)
    item = next(iter(roll._note_items.values()))

    item.setX(2.1 * roll.vm.beat_width)

    assert item.x() / roll.vm.beat_width == pytest.approx(
        round(item.x() / roll.vm.beat_width * 4) / 4, abs=1e-6
    )
    assert item.grid_x == pytest.approx(roll.vm.beat_width)


def test_stretching_the_notes_scales_every_start_exactly(qapp):
    roll = _roll()

    roll.stretch_notes(1.3)
    roll.sync_model_from_scene()

    assert _stored_beats(roll) == pytest.approx(
        [beat * 1.3 for beat in BEATS], abs=1e-9
    )
    assert _shown_beats(roll) == pytest.approx(
        [beat * 1.3 for beat in BEATS], abs=1e-6
    )
