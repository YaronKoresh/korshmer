import numpy as np
import pytest

from korshmer.midi_processing.note_validation import (
    MIDI_CC_MAX,
    MIDI_CC_MIN,
    validate_midi_cc_number,
)
from korshmer.midi_processing.pattern_generation import MidiCCAutomation


class TestValidateMidiCcNumber:
    @pytest.mark.parametrize("value", [0, 11, 127, np.int64(64), 11.0])
    def test_valid(self, value):
        assert validate_midi_cc_number(value) == int(value)
        assert isinstance(validate_midi_cc_number(value), int)

    @pytest.mark.parametrize(
        "value",
        [-1, -5, 128, 200, 3.7, float("nan"), None, True, False, "11", [11]],
    )
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="CC controller number"):
            validate_midi_cc_number(value)

    def test_bounds(self):
        assert (MIDI_CC_MIN, MIDI_CC_MAX) == (0, 127)

    def test_out_of_range_message_mentions_range(self):
        with pytest.raises(
            ValueError, match="must be between 0 and 127, got 200"
        ):
            validate_midi_cc_number(200)


class TestAddPoint:
    @pytest.mark.parametrize("cc_number", [0, 11, 127])
    def test_valid_edges_and_middle(self, cc_number):
        auto = MidiCCAutomation()
        auto.add_point(cc_number, 0.0, 64)
        assert list(auto.automation_data) == [cc_number]
        key = next(iter(auto.automation_data))
        assert isinstance(key, int)

    @pytest.mark.parametrize(
        "cc_number",
        [-1, -5, 128, 200, 3.7, "11", None, True, float("nan")],
    )
    def test_invalid_raises_and_adds_nothing(self, cc_number):
        auto = MidiCCAutomation()
        with pytest.raises(ValueError):
            auto.add_point(cc_number, 0.0, 64)
        assert auto.automation_data == {}

    def test_whole_float_is_normalised_to_int_key(self):
        auto = MidiCCAutomation()
        auto.add_point(11.0, 0.0, 64)
        auto.add_point(11, 1.0, 100)
        assert list(auto.automation_data) == [11]
        assert len(auto.automation_data[11]) == 2

    def test_value_still_clamped(self):
        auto = MidiCCAutomation()
        auto.add_point(64, 1.0, 500)
        assert auto.automation_data[64][0].value == 127
        auto.add_point(64, 2.0, -3)
        assert auto.automation_data[64][-1].value == 0


class TestGenerateCurve:
    def test_invalid_cc_raises_before_any_point(self):
        auto = MidiCCAutomation()
        with pytest.raises(ValueError):
            auto.generate_curve(300, 0, 1, 0, 127)
        assert 300 not in auto.automation_data
        assert auto.automation_data == {}

    def test_valid_cc_creates_points(self):
        auto = MidiCCAutomation()
        auto.generate_curve(1, 0, 1, 0, 127)
        assert list(auto.automation_data) == [1]
        assert len(auto.automation_data[1]) == 33

    def test_get_value_at_time_after_valid_curve(self):
        auto = MidiCCAutomation()
        auto.generate_curve(1, 0.0, 1.0, 0, 127, curve_type="linear")
        mid = auto.get_value_at_time(1, 0.5)
        assert mid is not None
        assert 0 <= mid <= 127

    def test_unknown_cc_returns_none(self):

        auto = MidiCCAutomation()
        assert auto.get_value_at_time(200, 0.5) is None


def test_nan_value_is_rejected_and_adds_nothing():

    auto = MidiCCAutomation()
    with pytest.raises(ValueError, match="CC value"):
        auto.add_point(7, 0.0, float("nan"))
    assert auto.automation_data == {}
