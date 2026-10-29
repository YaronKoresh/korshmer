import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.ui.widgets.statusbar import EnhancedStatusBar


def _hosted():
    window = QMainWindow()
    bar = EnhancedStatusBar(window)
    host = QStatusBar()
    host.addWidget(bar, 1)
    window.setStatusBar(host)
    host.setVisible(False)
    window.show()
    return window, bar, host


def test_bar_has_no_metric_labels():
    bar = EnhancedStatusBar()
    texts = {label.text() for label in bar.findChildren(QLabel)}
    assert texts == {""}
    for name in ("position_label", "chord_label", "scale_label"):
        assert not hasattr(bar, name)
    for name in ("cpu_label", "memory_label", "update_performance"):
        assert not hasattr(bar, name)


def test_message_shows_the_host_and_clearing_hides_it():
    window, bar, host = _hosted()
    try:
        assert not host.isVisible()
        bar.update_action("Undo: editor edit")
        assert host.isVisible()
        assert bar.action_label.text() == "Undo: editor edit"
        bar.clear_action()
        assert not host.isVisible()
        assert bar.action_label.text() == ""
    finally:
        window.close()


def test_message_clears_itself_after_a_timeout():
    window, bar, host = _hosted()
    try:
        bar.update_action("MIDI device connected")
        assert bar._clear_timer.isActive()
        bar._clear_timer.timeout.emit()
        assert not host.isVisible()
        assert not bar._clear_timer.isActive()
    finally:
        window.close()


def test_bar_without_a_status_bar_host_still_stores_the_text():
    bar = EnhancedStatusBar()
    bar.update_action("Ready")
    assert bar.action_label.text() == "Ready"
