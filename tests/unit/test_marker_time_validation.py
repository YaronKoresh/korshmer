import pytest

from korshmer.midi_processing.midi_file_enhancement import (
    CuePointManager,
    MIDIFileEnhancer,
)
from korshmer.utils.timeline import validate_timeline_time


class TestValidateTimelineTime:
    def test_zero_ok(self):
        assert validate_timeline_time(0.0) == 0.0

    def test_positive_ok(self):
        assert validate_timeline_time(12.5) == 12.5

    def test_negative_clamped(self):
        assert validate_timeline_time(-5.0) == 0.0

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), "abc", None]
    )
    def test_invalid_returns_none(self, value):
        assert validate_timeline_time(value) is None


class TestAddMarker:
    def test_negative_clamped_to_zero(self):
        e = MIDIFileEnhancer()
        e.add_marker(-5.0, "Intro")
        assert len(e.markers) == 1
        assert e.markers[0].time == 0.0
        assert e.markers[0].name == "Intro"

    def test_zero_added(self):
        e = MIDIFileEnhancer()
        e.add_marker(0.0, "Start")
        assert e.markers[0].time == 0.0

    def test_positive_unchanged(self):
        e = MIDIFileEnhancer()
        e.add_marker(12.5, "Verse")
        assert e.markers[0].time == 12.5

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "abc"])
    def test_invalid_not_added(self, value):
        e = MIDIFileEnhancer()
        before = len(e.markers)
        e.add_marker(value, "X")
        assert len(e.markers) == before


class TestAddCuePoint:
    def test_negative_clamped(self):
        m = CuePointManager()
        m.add_cue_point(-3.0, "Chorus")
        assert m.cue_points[0].time == 0.0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "abc"])
    def test_invalid_not_added(self, value):
        m = CuePointManager()
        m.add_cue_point(value, "Y")
        assert len(m.cue_points) == 0

    def test_find_nearest_cue_after_clamp(self):
        m = CuePointManager()
        m.add_cue_point(-3.0, "A")
        m.add_cue_point(2.0, "B")
        nearest = m.find_nearest_cue(0)
        assert nearest.time == 0.0
