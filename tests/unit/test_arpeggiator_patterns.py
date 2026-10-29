import random

import pytest

from korshmer.midi_processing.pattern_generation import (
    MAX_ARPEGGIO_NOTES,
    Arpeggiator,
    ArpPattern,
)

CHORD = [60, 64, 67]


def _arp(pattern, bpm=120.0):
    arp = Arpeggiator(bpm=bpm)
    arp.pattern = pattern
    return arp


def test_chord_pattern_plays_every_note_together_on_each_step():
    notes = _arp(ArpPattern.CHORD).generate_arpeggio(CHORD, 1.0, 0.25)
    starts = sorted({n.start_time for n in notes})
    assert starts == [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    for start in starts:
        at_step = sorted(n.pitch for n in notes if n.start_time == start)
        assert at_step == CHORD


def test_chord_pattern_keeps_one_octave_when_the_range_is_wider():
    arp = _arp(ArpPattern.CHORD)
    arp.set_octave_range(3)
    notes = arp.generate_arpeggio(CHORD, 0.125, 0.25)
    assert sorted(n.pitch for n in notes) == CHORD


def test_chord_pattern_counts_every_note_against_the_limit():
    arp = _arp(ArpPattern.CHORD, bpm=240.0)
    with pytest.raises(ValueError, match="limit"):
        arp.generate_arpeggio(CHORD, MAX_ARPEGGIO_NOTES, 0.25)


@pytest.mark.parametrize("seed", range(40))
def test_random_pattern_uses_each_note_once_per_cycle(seed):
    random.seed(seed)
    arp = _arp(ArpPattern.RANDOM)
    assert sorted(arp._generate_sequence(CHORD)) == CHORD


def test_random_pattern_with_octaves_uses_each_pitch_once_per_octave():
    random.seed(5)
    arp = _arp(ArpPattern.RANDOM)
    arp.set_octave_range(2)
    sequence = arp._generate_sequence(CHORD)
    assert sorted(sequence) == sorted(CHORD + [n + 12 for n in CHORD])


def test_up_pattern_still_plays_one_note_per_step():
    notes = _arp(ArpPattern.UP).generate_arpeggio(CHORD, 0.75, 0.25)
    assert [n.pitch for n in notes] == [60, 64, 67, 60, 64, 67]


@pytest.mark.parametrize("octaves", [2.5, 1.0, "2", None, True, float("nan")])
def test_octave_range_rejects_non_whole_values(octaves):
    with pytest.raises(ValueError, match="Octave range"):
        Arpeggiator().set_octave_range(octaves)


@pytest.mark.parametrize(("octaves", "expected"), [(0, 1), (2, 2), (9, 4)])
def test_octave_range_is_clamped_to_one_to_four(octaves, expected):
    arp = Arpeggiator()
    arp.set_octave_range(octaves)
    assert arp.octave_range == expected


@pytest.mark.parametrize(
    "length", [float("nan"), float("inf"), -0.5, "0.5", None, True]
)
def test_gate_length_rejects_invalid_values(length):
    with pytest.raises(ValueError, match="Gate length"):
        Arpeggiator().set_gate_length(length)


@pytest.mark.parametrize(
    ("length", "expected"), [(0.0, 0.1), (0.5, 0.5), (4, 1.0)]
)
def test_gate_length_is_clamped_and_stored_as_a_float(length, expected):
    arp = Arpeggiator()
    arp.set_gate_length(length)
    assert arp.gate_length == expected
    assert type(arp.gate_length) is float
