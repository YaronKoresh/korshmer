import sys
import time

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from korshmer.ui.widgets.material.mode_tool import ModeToolCard, ToolMode


def _card():
    return ModeToolCard("Tool", [ToolMode("One", lambda: None)])


def _wait(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_highlight_clears_itself():
    app = QApplication.instance() or QApplication(sys.argv)
    card = _card()
    card.highlight(20)
    assert card.title_label.styleSheet() != ""
    _wait(app, 0.2)
    assert card.title_label.styleSheet() == ""


def test_highlight_timer_is_dropped_when_the_card_is_deleted(capsys):
    app = QApplication.instance() or QApplication(sys.argv)
    card = _card()
    card.highlight(20)
    card.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _wait(app, 0.2)
    assert "already deleted" not in capsys.readouterr().err
