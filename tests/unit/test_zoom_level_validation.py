import math

import pytest

from korshmer.ui.widgets.midi.piano_roll import (
    MAX_ZOOM,
    MIN_ZOOM,
    InteractivePianoRoll,
    clamp_zoom,
)


class TestClampZoom:
    def test_in_range_unchanged(self):
        assert clamp_zoom(1.0) == 1.0
        assert clamp_zoom(MIN_ZOOM) == MIN_ZOOM
        assert clamp_zoom(MAX_ZOOM) == MAX_ZOOM

    def test_above_max_clamped(self):
        assert clamp_zoom(1000.0) == MAX_ZOOM

    def test_below_min_clamped(self):
        assert clamp_zoom(0.001) == MIN_ZOOM

    @pytest.mark.parametrize(
        "value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
    )
    def test_invalid_returns_none(self, value):
        assert clamp_zoom(value) is None


class TestPianoRollZoom:
    @pytest.fixture(autouse=True)
    def _qt_app(self, qapp):
        return qapp

    def test_plus_stops_at_max(self):
        roll = InteractivePianoRoll()
        for _ in range(100):
            roll.zoom(1.25, 1.0)
        assert roll.transform().m11() <= MAX_ZOOM + 1e-6
        assert abs(roll.transform().m11() - MAX_ZOOM) < 1e-6

    def test_minus_stops_at_min(self):
        roll = InteractivePianoRoll()
        for _ in range(100):
            roll.zoom(0.8, 1.0)
        assert roll.transform().m11() >= MIN_ZOOM - 1e-6
        assert abs(roll.transform().m11() - MIN_ZOOM) < 1e-6

    def test_scale_direct_clamped(self):
        roll = InteractivePianoRoll()
        roll.scale(1000, 1000)
        assert roll.transform().m11() <= MAX_ZOOM + 1e-6
        assert roll.transform().m22() <= MAX_ZOOM + 1e-6

    @pytest.mark.parametrize("fx", [float("nan"), 0, -2, float("inf")])
    def test_invalid_factor_no_change_no_error(self, fx):
        roll = InteractivePianoRoll()
        before = roll.transform().m11()
        roll.zoom(fx, 1.0)
        assert math.isclose(roll.transform().m11(), before)
