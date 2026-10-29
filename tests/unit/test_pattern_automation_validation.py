import pytest

from korshmer.midi_processing.pattern_generation import (
    MAX_RATCHET_REPEATS,
    ArpNote,
    MidiCCAutomation,
    Pattern,
    RatchetGenerator,
    StepSequencer,
)

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("time", [NAN, INF, -1.0, "1", None, True])
def test_cc_point_rejects_invalid_time(time):
    with pytest.raises(ValueError, match="Time"):
        MidiCCAutomation().add_point(1, time, 10)


@pytest.mark.parametrize("value", [NAN, INF, "5", None, True])
def test_cc_point_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="CC value"):
        MidiCCAutomation().add_point(1, 0.0, value)


def test_cc_point_stores_a_whole_clamped_value():
    auto = MidiCCAutomation()
    auto.add_point(1, 0.0, 64.7)
    auto.add_point(1, 1.0, 500)
    auto.add_point(1, 2.0, -9)
    values = [p.value for p in auto.automation_data[1]]
    assert values == [65, 127, 0]
    assert all(type(v) is int for v in values)


@pytest.mark.parametrize("time", [NAN, INF, "1", None])
def test_cc_lookup_rejects_invalid_time(time):
    auto = MidiCCAutomation()
    auto.add_point(1, 0.0, 10)
    with pytest.raises(ValueError, match="Time"):
        auto.get_value_at_time(1, time)


def test_cc_lookup_still_interpolates():
    auto = MidiCCAutomation()
    auto.add_point(1, 0.0, 0)
    auto.add_point(1, 2.0, 100)
    assert auto.get_value_at_time(1, 1.0) == 50
    assert auto.get_value_at_time(1, -5.0) == 0
    assert auto.get_value_at_time(9, 1.0) is None


@pytest.mark.parametrize(
    ("start", "end"), [(NAN, 1.0), (0.0, INF), (-1.0, 1.0), (2.0, 1.0)]
)
def test_curve_rejects_invalid_times(start, end):
    with pytest.raises(ValueError, match="time"):
        MidiCCAutomation().generate_curve(1, start, end, 0, 127)


@pytest.mark.parametrize("value", [NAN, INF, "5", None])
def test_curve_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="value"):
        MidiCCAutomation().generate_curve(1, 0.0, 1.0, value, 127)


def test_curve_rejects_unknown_type():
    with pytest.raises(ValueError, match="curve type"):
        MidiCCAutomation().generate_curve(1, 0.0, 1.0, 0, 127, "bogus")


@pytest.mark.parametrize(
    "kind", ["linear", "exponential", "logarithmic", "s-curve"]
)
def test_curve_runs_from_start_to_end_value(kind):
    auto = MidiCCAutomation()
    auto.generate_curve(1, 0.0, 1.0, 10, 100, kind)
    points = auto.automation_data[1]
    assert len(points) == 33
    assert points[0].value == 10
    assert points[-1].value == 100
    assert points[-1].time == 1.0


def _note():
    return [ArpNote(60, 0.0, 0.5, 100, 0.8)]


@pytest.mark.parametrize("repeats", [2.5, 2.0, "2", None, True, NAN])
def test_ratchet_rejects_non_whole_repeats(repeats):
    with pytest.raises(ValueError, match="Ratchet repeats"):
        RatchetGenerator().apply_ratcheting(_note(), [repeats])


def test_ratchet_rejects_a_huge_repeat_count():
    with pytest.raises(ValueError, match="at most"):
        RatchetGenerator().apply_ratcheting(_note(), [MAX_RATCHET_REPEATS + 1])


def test_ratchet_splits_a_note_and_keeps_small_values_as_is():
    gen = RatchetGenerator()
    assert len(gen.apply_ratcheting(_note(), [4])) == 4
    assert len(gen.apply_ratcheting(_note(), [1])) == 1
    assert len(gen.apply_ratcheting(_note(), [0])) == 1
    assert len(gen.apply_ratcheting(_note(), [-2])) == 1


def test_pattern_with_no_steps_gives_no_notes():
    seq = StepSequencer(4)
    empty = Pattern("e", [], [], [], [])
    assert seq.generate_from_pattern(empty, 60, 1.0) == []


@pytest.mark.parametrize("duration", [NAN, INF, -1.0, "1", None])
def test_pattern_rejects_invalid_duration(duration):
    seq = StepSequencer(4)
    pattern = seq.create_pattern("p")
    with pytest.raises(ValueError, match="Duration"):
        seq.generate_from_pattern(pattern, 60, duration)


@pytest.mark.parametrize("pitch", [-1, 128, 60.5, "60", None])
def test_pattern_rejects_invalid_base_pitch(pitch):
    seq = StepSequencer(4)
    pattern = seq.create_pattern("p")
    with pytest.raises(ValueError, match="base pitch"):
        seq.generate_from_pattern(pattern, pitch, 1.0)


def test_pattern_notes_are_spread_over_the_duration():
    seq = StepSequencer(4)
    pattern = seq.create_pattern("p", [True, False, True, False])
    notes = seq.generate_from_pattern(pattern, 60, 2.0)
    assert [n.start_time for n in notes] == [0.0, 1.0]
