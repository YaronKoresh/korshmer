import numpy as np
import pretty_midi
import pytest

from korshmer.midi.bar_lines import bar_lines
from korshmer.midi.variation import _bar_starts


def _song(pickup_at=None, first=0.5, bars=4, signatures=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    if pickup_at is not None:
        inst.notes.append(pretty_midi.Note(60, 72, pickup_at, pickup_at + 0.3))
    for bar in range(bars):
        at = first + 2.0 * bar
        inst.notes.append(pretty_midi.Note(110, 36, at, at + 0.4))
        for k in range(1, 4):
            inst.notes.append(
                pretty_midi.Note(60, 72, at + 0.5 * k, at + 0.5 * k + 0.4)
            )
    pm.instruments.append(inst)
    if signatures:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    return pm


def test_bar_lines_follow_a_one_beat_pickup():
    lines = bar_lines(_song(pickup_at=0.0))

    assert lines[:3] == pytest.approx([0.5, 2.5, 4.5])


def test_bar_lines_cover_the_song_to_its_end():
    pm = _song(pickup_at=0.0)

    assert bar_lines(pm)[-1] <= pm.get_end_time()
    assert bar_lines(pm)[-1] + 2.0 > pm.get_end_time()


def test_song_that_starts_on_a_bar_line_is_unchanged():
    pm = _song(first=0.0)

    assert np.allclose(bar_lines(pm), pm.get_downbeats())


def test_explicit_time_signature_wins():
    pm = _song(pickup_at=0.0, signatures=True)

    assert np.allclose(bar_lines(pm), pm.get_downbeats())


def test_tempo_change_keeps_the_default_lines():
    pm = _song(pickup_at=0.0)
    pm._tick_scales = [(0, 0.5 / 220), (480, 0.25 / 220)]

    assert np.allclose(bar_lines(pm), pm.get_downbeats())


def test_empty_midi_has_default_lines():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)

    assert np.allclose(bar_lines(pm), pm.get_downbeats())


def test_variation_uses_the_real_bar_lines():
    starts = _bar_starts(_song(pickup_at=0.0))

    assert starts[:3] == pytest.approx([0.5, 2.5, 4.5])


def test_beat_lines_follow_the_pickup_and_hold_every_bar_line():
    from korshmer.midi.bar_lines import beat_lines

    pm = _song(pickup_at=0.0, first=0.35)
    beats = beat_lines(pm)
    bars = bar_lines(pm)

    assert beats[0] == pytest.approx(0.35, abs=0.01)
    assert np.allclose(np.diff(beats), 0.5)
    for bar in bars:
        assert np.min(np.abs(beats - bar)) < 1e-6


def test_beat_lines_without_a_pickup_are_the_pretty_midi_beats():
    from korshmer.midi.bar_lines import beat_lines

    pm = _song(first=0.0)

    assert np.allclose(beat_lines(pm), pm.get_beats())
