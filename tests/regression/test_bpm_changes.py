import math
import sys

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from korshmer.utils.tempo import MAX_BPM, MIN_BPM

NOTES = [(60, 0.0, 1.0), (64, 0.5, 0.25), (67, 2.0, 1.5)]
TOL = 1e-9


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def piano_roll(qapp):
    from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

    roll = InteractivePianoRoll()
    roll.set_tempo(120.0)
    roll.model.notes.clear()
    track = roll.model.add_track("t")
    track_id = track.id if hasattr(track, "id") else track
    for pitch, start, duration in NOTES:
        roll.model.add_note(track_id, pitch, start, duration)
    return roll


def _times(roll):
    return [(n.start, n.duration) for n in roll.model.notes]


def _assert_scaled(roll, factor):
    for (start, duration), (_, s0, d0) in zip(
        _times(roll), NOTES, strict=False
    ):
        assert start == pytest.approx(s0 * factor, abs=TOL)
        assert duration == pytest.approx(d0 * factor, abs=TOL)


def test_doubling_tempo_halves_note_times(piano_roll):
    piano_roll.set_tempo(240.0)
    assert piano_roll.tempo == 240.0
    _assert_scaled(piano_roll, 0.5)


def test_round_trip_restores_original_times(piano_roll):
    piano_roll.set_tempo(90.0)
    piano_roll.set_tempo(120.0)
    _assert_scaled(piano_roll, 1.0)


@pytest.mark.parametrize(
    "requested, bound", [(5.0, MIN_BPM), (1000.0, MAX_BPM)]
)
def test_out_of_range_tempo_clamped(piano_roll, requested, bound):
    piano_roll.set_tempo(requested)
    assert piano_roll.tempo == bound
    _assert_scaled(piano_roll, 120.0 / bound)


def test_small_decimal_change_is_not_ignored(piano_roll):
    piano_roll.set_tempo(120.1)
    _assert_scaled(piano_roll, 120.0 / 120.1)


def test_many_small_steps_match_direct_change(piano_roll):
    for i in range(1, 101):
        piano_roll.set_tempo(120.0 + i * 0.1)
    _assert_scaled(piano_roll, 120.0 / 130.0)


def test_view_model_reports_small_change():
    from korshmer.ui.viewmodels.piano_roll_vm import PianoRollViewModel

    vm = PianoRollViewModel()
    vm.tempo = 120.0
    assert vm.compute_tempo_change(120.1) is not None
    assert vm.tempo == 120.1


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.lists(
        st.floats(min_value=MIN_BPM, max_value=MAX_BPM), min_size=1, max_size=8
    )
)
def test_note_time_times_tempo_is_constant(piano_roll, bpms):
    piano_roll.set_tempo(120.0)
    for n, (_, s0, d0) in zip(piano_roll.model.notes, NOTES, strict=False):
        n.start, n.duration = s0, d0
    for bpm in bpms:
        piano_roll.set_tempo(bpm)
    tempo = piano_roll.tempo
    for (start, duration), (_, s0, d0) in zip(
        _times(piano_roll), NOTES, strict=False
    ):
        assert math.isclose(
            start * tempo, s0 * 120.0, rel_tol=1e-6, abs_tol=1e-9
        )
        assert math.isclose(
            duration * tempo, d0 * 120.0, rel_tol=1e-6, abs_tol=1e-9
        )
