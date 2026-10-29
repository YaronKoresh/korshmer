import pretty_midi
import pytest

from korshmer.midi_generation.composer import ComposeOptions, compose

BPM = 120.0


def _song(pickup=True):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    inst = pretty_midi.Instrument(0)
    first = 0.5 if pickup else 0.0
    if pickup:
        inst.notes.append(pretty_midi.Note(60, 72, 0.0, 0.3))
    for bar in range(4):
        at = first + 2.0 * bar
        inst.notes.append(pretty_midi.Note(110, 45, at, at + 0.4))
        for k in range(1, 4):
            inst.notes.append(
                pretty_midi.Note(60, 72, at + 0.5 * k, at + 0.5 * k + 0.4)
            )
    for k in range(2):
        at = first + 8.0 + 0.5 * k
        inst.notes.append(pretty_midi.Note(60, 72, at, at + 0.4))
    pm.instruments.append(inst)
    return pm


def _continue(pm):
    options = ComposeOptions(
        root="A",
        scale="Minor",
        bpm=BPM,
        minutes=1.0,
        seed=1,
        mode="Continue editor",
        texture_batches=0,
    )
    return compose(options, existing=pm)


def _first_start(composition):
    return min(n.start for i in composition.instruments for n in i.notes)


def test_continue_starts_on_the_next_real_bar_line_after_a_pickup():
    c = _continue(_song())

    assert _first_start(c) == pytest.approx(10.5, abs=1e-6)


def test_continue_without_a_pickup_starts_on_the_zero_grid():
    pm = _song(pickup=False)
    c = _continue(pm)

    assert _first_start(c) == pytest.approx(c.start_bar * 2.0, abs=1e-6)
    assert _first_start(c) == pytest.approx(10.0, abs=1e-6)
