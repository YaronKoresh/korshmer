import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.ui.widgets.audio import visualizer as visualizer_module
from korshmer.ui.widgets.audio.live_meters import LiveMeters
from korshmer.ui.widgets.audio.visualizer import AudioVisualizer

SR = 44100


class FakeEngine(QObject):
    playbackStateChanged = Signal(int)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.fs = SR
        self.idx = len(data)
        self.state = 0

    def playbackState(self):
        return self.state

    def switch(self, state):
        self.state = state
        self.playbackStateChanged.emit(state)


def quiet_song():
    t = np.arange(SR) / SR
    tone = 0.25 * np.sin(2 * np.pi * 440 * t)
    return np.stack([tone, tone], axis=1)


def loud_song():
    return np.ones((SR, 2))


@pytest.fixture
def meters():
    widget = LiveMeters()
    widget.show()
    yield widget
    widget.close()


@pytest.fixture
def visualizer():
    widget = AudioVisualizer()
    yield widget
    widget.close()


def count_bar_restyles(meters):
    calls = []
    original = meters.bar.setStyleSheet

    def record(sheet):
        calls.append(sheet)
        original(sheet)

    meters.bar.setStyleSheet = record
    return calls


def count_refreshes(meters):
    calls = []
    original = meters.show_levels

    def record(levels):
        calls.append(levels)
        original(levels)

    meters.show_levels = record
    return calls


def start_song(visualizer):
    visualizer.on_analysis_done(np.full(100, 0.9), 10.0)
    visualizer.set_position(5000)
    visualizer.set_playing(True)


def test_meters_do_not_refresh_before_anything_plays(meters):
    meters.set_engine(FakeEngine(quiet_song()))
    assert not meters.timer.isActive()


def test_safeguard_meters_refresh_while_a_song_plays(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    engine.switch(1)
    assert meters.timer.isActive()


def test_meters_stop_refreshing_when_the_song_stops(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    engine.switch(1)
    engine.switch(0)
    assert not meters.timer.isActive()


def test_hidden_meters_do_not_refresh_during_playback(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    engine.switch(1)
    meters.hide()
    assert not meters.timer.isActive()


def test_safeguard_meters_resume_when_shown_again_during_playback(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    meters.hide()
    engine.switch(1)
    meters.show()
    assert meters.timer.isActive()


def test_meters_stop_following_a_player_that_was_replaced(meters):
    old = FakeEngine(quiet_song())
    meters.set_engine(old)
    meters.set_engine(FakeEngine(quiet_song()))
    old.switch(1)
    assert not meters.timer.isActive()


def test_giving_the_meters_the_same_player_twice_connects_it_once(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    meters.set_engine(engine)
    assert engine.receivers("2playbackStateChanged(int)") == 1


def test_the_level_bar_is_not_restyled_while_its_colour_stays_the_same(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    engine.switch(1)
    restyles = count_bar_restyles(meters)
    for _ in range(10):
        meters.timer.timeout.emit()
    assert restyles == []


def test_the_level_bar_is_restyled_once_when_the_song_starts_clipping(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    engine.switch(1)
    meters.timer.timeout.emit()
    restyles = count_bar_restyles(meters)
    engine.data = loud_song()
    for _ in range(5):
        meters.timer.timeout.emit()
    assert len(restyles) == 1


def test_safeguard_meters_show_levels_on_every_tick_while_playing(meters):
    engine = FakeEngine(quiet_song())
    meters.set_engine(engine)
    engine.switch(1)
    refreshes = count_refreshes(meters)
    for _ in range(3):
        meters.timer.timeout.emit()
    assert len(refreshes) == 3


def test_the_visualizer_does_not_animate_while_hidden(visualizer):
    start_song(visualizer)
    assert not visualizer.timer.isActive()


def test_the_visualizer_does_not_animate_while_nothing_plays(visualizer):
    visualizer.show()
    assert not visualizer.timer.isActive()


def test_safeguard_the_visualizer_animates_while_visible_and_playing(
    visualizer,
):
    visualizer.show()
    start_song(visualizer)
    assert visualizer.timer.isActive()


def test_the_visualizer_bars_fall_back_and_then_rest_after_stop(visualizer):
    visualizer.show()
    start_song(visualizer)
    for _ in range(5):
        visualizer.timer.timeout.emit()
    raised = max(visualizer.values)
    visualizer.set_playing(False)
    for _ in range(200):
        visualizer.timer.timeout.emit()
    assert raised > 0.05 and max(visualizer.values) < 0.01
    assert not visualizer.timer.isActive()


def test_the_visualizer_pauses_when_hidden_and_resumes_when_shown(visualizer):
    visualizer.show()
    start_song(visualizer)
    visualizer.hide()
    paused = visualizer.timer.isActive()
    visualizer.show()
    assert (paused, visualizer.timer.isActive()) == (False, True)


def test_painting_the_visualizer_reuses_its_bar_gradient(
    visualizer, monkeypatch
):
    built = []
    real = visualizer_module.QLinearGradient

    def counting(*args):
        built.append(args)
        return real(*args)

    monkeypatch.setattr(visualizer_module, "QLinearGradient", counting)
    visualizer.resize(200, 80)
    for _ in range(4):
        visualizer.grab()
    assert len(built) <= 1
