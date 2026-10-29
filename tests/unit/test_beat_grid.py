import numpy as np
import pretty_midi
import pytest

from korshmer.midi_algorithms import (
    apply_beat_grid,
    fit_beat_grid,
    fix_legato_and_flutter,
)

TEMPO = 118.3
OFFSET = 0.35


def _onsets(count=200, jitter=0.0, seed=1):
    rng = np.random.default_rng(seed)
    step = 60.0 / TEMPO / 4
    slots = rng.integers(0, 60, size=count)
    return OFFSET + slots * step + rng.normal(0, jitter, size=count)


def test_fit_finds_fractional_tempo_and_start_offset():
    tempo, offset, alignment = fit_beat_grid(_onsets(), 117.0)
    assert tempo == pytest.approx(TEMPO, abs=0.05)
    step = 60.0 / tempo / 4
    assert (offset - OFFSET) / step == pytest.approx(
        round((offset - OFFSET) / step), abs=0.05
    )
    assert alignment > 0.95


def test_fit_tolerates_human_timing_jitter():
    tempo, _, alignment = fit_beat_grid(_onsets(jitter=0.015), 117.0)
    assert tempo == pytest.approx(TEMPO, abs=0.2)
    assert alignment > 0.5


def test_fit_rejects_onsets_that_do_not_sit_on_a_grid():
    times = np.random.default_rng(3).uniform(0, 100, size=300)
    assert fit_beat_grid(times, 117.0) is None


def test_fit_needs_enough_onsets():
    assert fit_beat_grid(_onsets(count=5), 117.0) is None


def _midi(notes):
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(0)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(90, pitch, start, end))
    pm.instruments.append(inst)
    return pm


def test_apply_beat_grid_keeps_note_lengths_and_gaps():
    step = 60.0 / TEMPO / 4
    notes = [
        (40, OFFSET + k * 2 * step + 0.012, OFFSET + k * 2 * step + 0.2)
        for k in range(10)
    ]
    pm = apply_beat_grid(_midi(notes), TEMPO, OFFSET)
    out = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    for k, note in enumerate(out):
        assert note.start == pytest.approx(OFFSET + k * 2 * step, abs=1e-6)
        assert note.end - note.start == pytest.approx(0.188, abs=1e-6)
    assert all(a.end < b.start for a, b in zip(out, out[1:], strict=False))


def test_apply_beat_grid_never_overlaps_a_repeated_pitch():
    step = 60.0 / TEMPO / 4
    notes = [
        (40, OFFSET + 0.4 * step, OFFSET + 3.2 * step),
        (40, OFFSET + 2.7 * step, OFFSET + 5 * step),
    ]
    out = apply_beat_grid(_midi(notes), TEMPO, OFFSET).instruments[0].notes
    first, second = sorted(out, key=lambda n: n.start)
    assert first.end <= second.start


def test_legato_keeps_repeated_notes_of_normal_length():
    pm = _midi([(40, 0.0, 0.25), (40, 0.26, 0.5), (40, 0.5, 0.75)])
    assert (
        len(fix_legato_and_flutter(pm, min_duration=0.06).instruments[0].notes)
        == 3
    )


def test_legato_merges_a_short_fragment_into_its_note():
    pm = _midi([(40, 0.0, 0.3), (40, 0.31, 0.38)])
    notes = fix_legato_and_flutter(pm, min_duration=0.06).instruments[0].notes
    assert len(notes) == 1
    assert notes[0].end == pytest.approx(0.38)


def test_legato_min_duration_is_adjustable():
    pm = _midi([(40, 0.0, 0.09), (41, 1.0, 1.5)])
    assert len(fix_legato_and_flutter(pm).instruments[0].notes) == 1
    pm = _midi([(40, 0.0, 0.09), (41, 1.0, 1.5)])
    assert (
        len(fix_legato_and_flutter(pm, min_duration=0.06).instruments[0].notes)
        == 2
    )
