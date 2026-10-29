import math

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from korshmer.ui.utils.throttle import EventThrottle


def test_first_call_immediate_and_burst_coalesced(qtbot):
    calls = []
    t = EventThrottle(lambda v: calls.append(v), interval_ms=30)
    for i in range(50):
        t.submit(i)
    assert calls == [0]
    qtbot.waitUntil(lambda: calls[-1] == 49, timeout=1000)
    assert calls == [0, 49]


def test_cancel_drops_pending(qtbot):
    calls = []
    t = EventThrottle(lambda v: calls.append(v), interval_ms=20)
    t.submit(1)
    t.submit(2)
    t.cancel()
    qtbot.wait(60)
    assert calls == [1]


def test_flush_runs_pending_now(qtbot):
    calls = []
    t = EventThrottle(lambda v: calls.append(v), interval_ms=500)
    t.submit(1)
    t.submit(2)
    t.flush()
    assert calls == [1, 2]
    t.cancel()


def test_piano_roll_wheel_zoom_calls_are_reduced_and_zoom_not_lost(qtbot):
    from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll

    roll = InteractivePianoRoll()
    qtbot.addWidget(roll)
    calls = []
    original = roll.zoom
    roll.zoom = lambda fx=1.0, fy=1.0: (
        calls.append((fx, fy)),
        original(fx, fy),
    )

    def wheel(delta):
        return QWheelEvent(
            QPointF(10, 10),
            QPointF(10, 10),
            QPoint(0, 0),
            QPoint(0, delta),
            Qt.NoButton,
            Qt.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )

    for _ in range(200):
        roll.wheelEvent(wheel(120))
        roll.wheelEvent(wheel(-120))
    qtbot.waitUntil(lambda: roll._pending_zoom_x == 1.0, timeout=1000)
    qtbot.wait(60)
    assert len(calls) < 20
    assert math.isclose(math.prod(c[0] for c in calls), 1.0, rel_tol=1e-9)


def test_spectrogram_hover_is_throttled(qtbot):
    from PySide6.QtGui import QImage

    from korshmer.ui.widgets.audio.spectrogram import SpectrogramWidget

    w = SpectrogramWidget()
    qtbot.addWidget(w)
    w.spectrogram_image = QImage(10, 10, QImage.Format_RGB32)
    seen = []
    w.vm.hover_info = lambda x, y, W, H: seen.append((x, y)) or None

    class Ev:
        def __init__(self, x):
            self._x = x

        def position(self):
            return QPointF(self._x, 5)

    for x in range(100):
        w.mouseMoveEvent(Ev(x))
    qtbot.waitUntil(lambda: seen and seen[-1][0] == 99, timeout=1000)
    assert len(seen) < 10
    assert seen[0][0] == 0
