import numpy as np
import pretty_midi

from korshmer.midi.adaptive_snap import snap_adaptive
from korshmer.midi_algorithms import fit_beat_grid, fit_beat_grid_or_best

TEMPO = 120.0
BEAT = 60.0 / TEMPO


def _melody():
    rng = np.random.default_rng(1)
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(0)
    t = 0.0
    for k in range(40):
        d = float(rng.choice([0.125, 0.25, 0.5, 1.0]))
        start = max(0.0, t + rng.normal(0, 0.02))
        end = start + d * rng.uniform(0.9, 1.1)
        inst.notes.append(pretty_midi.Note(80, 60 + k % 12, start, end))
        t += d
    midi.instruments.append(inst)
    return midi


def _off_grid(times, offset, step):
    ratio = (np.asarray(times) - offset) / step
    return int(np.sum(np.abs(ratio - np.round(ratio)) * step > 0.005))


def test_snap_puts_note_ends_on_the_grid():
    midi = _melody()
    tempo, offset, _ = fit_beat_grid(
        [n.start for n in midi.instruments[0].notes], TEMPO
    )
    snap_adaptive(midi, tempo, offset, finest=4)
    notes = midi.instruments[0].notes
    step = BEAT / 4
    assert _off_grid([n.start for n in notes], offset, step) == 0
    assert _off_grid([n.end for n in notes], offset, step) == 0
    assert all(n.end - n.start >= step - 1e-9 for n in notes)


def test_snap_keeps_drum_hit_length():
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    kit = pretty_midi.Instrument(0, is_drum=True)
    kit.notes.append(pretty_midi.Note(100, 36, 0.01, 0.13))
    midi.instruments.append(kit)
    snap_adaptive(midi, TEMPO, 0.0, finest=4)
    assert abs(kit.notes[0].end - kit.notes[0].start - 0.12) < 1e-9


def test_weak_grid_falls_back_to_best_phase():
    onsets = [k * 0.37 + 0.01 for k in range(10)]
    assert fit_beat_grid(onsets, TEMPO, 4) is None
    grid = fit_beat_grid_or_best(onsets, TEMPO, 4)
    assert grid is not None
    assert np.isfinite(grid).all()
