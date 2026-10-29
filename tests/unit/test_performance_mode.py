from PySide6.QtWidgets import QSpinBox, QVBoxLayout, QWidget

from korshmer.mixins.performance_mode import PerformanceModeMixin
from korshmer.ui.managers.keyboard_shortcuts import KeyboardShortcutManager
from korshmer.ui.widgets.performance_panel import (
    PERFORMANCE_BUTTON_MIN_HEIGHT,
    PerformancePanel,
)


class _Host(QWidget, PerformanceModeMixin):
    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout(self)
        self.title_bar = QWidget()
        self.splitter = QWidget()
        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.splitter, 1)
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(20, 300)
        self.spin_bpm.setValue(120)
        self.calls = []

    def _handle_play_pause(self):
        self.calls.append("play_pause")


def test_toggle_twice_restores_layout_and_tempo(mock_qt_app):
    host = _Host()
    host.show()
    assert host.splitter.isVisible()

    assert host.toggle_performance_mode() is True
    assert host.is_performance_mode()
    assert not host.splitter.isVisible()
    assert host._performance_panel.isVisible()

    assert host.toggle_performance_mode() is False
    assert not host.is_performance_mode()
    assert host.splitter.isVisible()
    assert not host._performance_panel.isVisible()
    assert host.spin_bpm.value() == 120


def test_buttons_are_at_least_64_pixels_high(mock_qt_app):
    panel = PerformancePanel()
    assert PERFORMANCE_BUTTON_MIN_HEIGHT >= 64
    assert len(panel.buttons()) == 6
    assert all(b.minimumHeight() >= 64 for b in panel.buttons())


def test_tempo_buttons_adjust_spinbox(mock_qt_app):
    host = _Host()
    host.toggle_performance_mode()
    panel = host._performance_panel
    panel.btn_tempo_up.click()
    assert host.spin_bpm.value() == 125
    panel.btn_tempo_down.click()
    panel.btn_tempo_down.click()
    assert host.spin_bpm.value() == 115


def test_play_pause_button_uses_play_pause_handler(mock_qt_app):
    host = _Host()
    host.toggle_performance_mode()
    host._performance_panel.btn_play_pause.click()
    assert host.calls == ["play_pause"]


def test_scene_buttons_use_setlist_when_present(mock_qt_app):
    host = _Host()

    class FakeSetlist:
        def __init__(self):
            self.moves = []

        def next(self):
            self.moves.append("next")

        def previous(self):
            self.moves.append("previous")

    host.setlist = FakeSetlist()
    host.toggle_performance_mode()
    host._performance_panel.btn_next_scene.click()
    host._performance_panel.btn_previous_scene.click()
    assert host.setlist.moves == ["next", "previous"]


def test_scene_buttons_without_setlist_do_nothing(mock_qt_app):
    host = _Host()
    host.toggle_performance_mode()
    host._performance_panel.btn_next_scene.click()


def test_performance_shortcut_does_not_collide_with_fullscreen():
    shortcuts = KeyboardShortcutManager()
    assert shortcuts.get("performance_mode") == "F10"
    assert shortcuts.get("fullscreen") == "F11"
    values = list(shortcuts.get_all().values())
    assert len(values) == len(set(values))
