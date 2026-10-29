import pytest

from korshmer.midi_processing.drum_map import (
    FILL_TYPES,
    DrumFillGenerator,
    DrumRollGenerator,
    HiHatVariationGenerator,
)


@pytest.mark.parametrize("roll_type", ["single_stroke", "double_stroke"])
def test_known_roll_types_produce_notes(roll_type):
    notes = DrumRollGenerator().generate_roll(38, 0.0, 1.0, roll_type)
    assert notes


@pytest.mark.parametrize("pattern", ["eighth_notes", "sixteenth_notes"])
def test_known_hihat_patterns_produce_notes(pattern):
    assert HiHatVariationGenerator().generate_variation(2.0, pattern)


def test_unknown_roll_type_is_rejected():
    with pytest.raises(ValueError, match="single_stroke"):
        DrumRollGenerator().generate_roll(38, 0.0, 1.0, "paradiddle")


def test_unknown_hihat_pattern_is_rejected():
    with pytest.raises(ValueError, match="eighth_notes"):
        HiHatVariationGenerator().generate_variation(4.0, "triplets")


@pytest.mark.parametrize("fill_type", FILL_TYPES)
def test_known_fill_types_produce_notes(fill_type):
    assert DrumFillGenerator().generate_fill(2.0, "medium", fill_type)


def test_unknown_fill_type_is_rejected():
    with pytest.raises(ValueError, match="tom_roll"):
        DrumFillGenerator().generate_fill(2.0, "medium", "nonsense")


def test_unknown_fill_intensity_is_rejected():
    with pytest.raises(ValueError, match="medium"):
        DrumFillGenerator().generate_fill(2.0, "loud")


@pytest.mark.parametrize(
    "duration", [float("inf"), float("nan"), -1.0, "2", None, True]
)
def test_invalid_fill_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        DrumFillGenerator().generate_fill(duration)
