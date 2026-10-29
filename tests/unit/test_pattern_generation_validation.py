import pytest

from korshmer.midi_processing.pattern_generation import (
    Arpeggiator,
    EuclideanRhythmGenerator,
    PolyrhythmGenerator,
    StepSequencer,
)

BAD_NUMBERS = [0, -1, float("nan"), float("inf"), True, None, "1"]


@pytest.mark.parametrize("subdivision", BAD_NUMBERS)
def test_arpeggio_rejects_invalid_subdivision(subdivision):
    with pytest.raises(ValueError):
        Arpeggiator().generate_arpeggio([60, 64], 2.0, subdivision)


@pytest.mark.parametrize(
    "duration", [-2, float("nan"), float("inf"), True, None, "1"]
)
def test_arpeggio_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        Arpeggiator().generate_arpeggio([60, 64], duration)


def test_arpeggio_rejects_huge_note_count():
    with pytest.raises(ValueError):
        Arpeggiator().generate_arpeggio([60], 1e9)


def test_arpeggio_zero_duration_is_empty():
    assert Arpeggiator().generate_arpeggio([60, 64], 0) == []


def test_arpeggio_valid_call_unchanged():
    notes = Arpeggiator().generate_arpeggio([60, 64], 2.0)

    assert [n.pitch for n in notes[:3]] == [60, 64, 60]
    assert len(notes) == 16


@pytest.mark.parametrize("steps", [-1, 4097, 2.5, True, None])
def test_step_sequencer_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError):
        StepSequencer(steps)


def test_step_sequencer_zero_steps_rejected():
    with pytest.raises(ValueError):
        StepSequencer(0)


@pytest.mark.parametrize(
    "pulses, steps", [(-2, 8), (2, -4), (2.5, 8), (2, None)]
)
def test_euclidean_rejects_invalid_counts(pulses, steps):
    with pytest.raises(ValueError):
        EuclideanRhythmGenerator().generate_euclidean(pulses, steps)


def test_euclidean_valid_cases_unchanged():
    gen = EuclideanRhythmGenerator()

    assert gen.generate_euclidean(0, 4) == [False] * 4
    assert gen.generate_euclidean(5, 4) == [True] * 4
    assert gen.generate_euclidean(3, 0) == []
    assert sum(gen.generate_euclidean(3, 8)) == 3


def test_rotate_empty_pattern_returns_empty():
    assert EuclideanRhythmGenerator().rotate_pattern([], 3) == []


@pytest.mark.parametrize("rotation", [1.5, True, None])
def test_rotate_rejects_non_whole_rotation(rotation):
    with pytest.raises(ValueError):
        EuclideanRhythmGenerator().rotate_pattern([True, False], rotation)


@pytest.mark.parametrize("numerator", [0, -3, 2.5, 5000])
def test_polyrhythm_rejects_invalid_numerator(numerator):
    with pytest.raises(ValueError):
        PolyrhythmGenerator().generate_polyrhythm([(numerator, 3)], 4.0, 60)


@pytest.mark.parametrize("duration", [0, -4, float("nan"), float("inf")])
def test_polyrhythm_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        PolyrhythmGenerator().generate_polyrhythm([(3, 2)], duration, 60)


def test_polyrhythm_valid_call_unchanged():
    tracks = PolyrhythmGenerator().generate_polyrhythm(
        [(3, 2), (2, 3)], 4.0, 60
    )

    assert [len(t) for t in tracks] == [3, 2]
