import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QWidget,
)

if not QApplication.instance():
    QApplication(sys.argv)

try:
    from korshmer.__main__ import MainWindow
except Exception as import_error:
    MainWindow = None
    MAIN_IMPORT_ERROR = str(import_error)
else:
    MAIN_IMPORT_ERROR = ""

from korshmer.core.window_initializer import initial_window_size
from korshmer.ui import layout_config
from tests.fixtures.qt_fonts import ensure_real_fonts

FONTS_AVAILABLE = ensure_real_fonts()

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def window():
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    win = MainWindow()
    win.show()
    QApplication.processEvents()
    yield win
    win.close()
    QApplication.processEvents()


def test_minimum_window_is_much_smaller_than_the_default():
    assert layout_config.MAIN_WINDOW_MIN_WIDTH <= 900
    assert layout_config.MAIN_WINDOW_MIN_HEIGHT <= 650
    assert layout_config.MAIN_WINDOW_MIN_WIDTH < layout_config.MAIN_WINDOW_WIDTH
    assert (
        layout_config.MAIN_WINDOW_MIN_HEIGHT < layout_config.MAIN_WINDOW_HEIGHT
    )


def test_window_minimum_matches_the_configured_minimum(window):
    assert window.minimumWidth() == layout_config.MAIN_WINDOW_MIN_WIDTH
    assert window.minimumHeight() == layout_config.MAIN_WINDOW_MIN_HEIGHT


def test_content_fits_inside_the_configured_minimum(window):
    hint = window.minimumSizeHint()
    assert hint.width() <= layout_config.MAIN_WINDOW_MIN_WIDTH
    assert hint.height() <= layout_config.MAIN_WINDOW_MIN_HEIGHT


def test_window_shrinks_to_the_minimum(window):
    window.resize(
        layout_config.MAIN_WINDOW_MIN_WIDTH,
        layout_config.MAIN_WINDOW_MIN_HEIGHT,
    )
    QApplication.processEvents()
    assert window.width() == layout_config.MAIN_WINDOW_MIN_WIDTH
    assert window.height() == layout_config.MAIN_WINDOW_MIN_HEIGHT
    assert window.control_bar.isVisible()


def test_graphs_and_player_bar_can_shrink(window):
    assert window.spectrogram.minimumHeight() <= 48
    assert window.visualizer.minimumHeight() <= 48
    assert window.control_bar.minimumHeight() <= 100


def test_graphs_grow_back_when_there_is_room(window):
    window.resize(1250, 950)
    QApplication.processEvents()
    spectrogram, visualizer = window.spectrogram, window.visualizer
    assert spectrogram.minimumHeight() < spectrogram.height() <= 120
    assert visualizer.minimumHeight() < visualizer.height() <= 100


def test_initial_size_never_exceeds_the_screen():
    width, height = initial_window_size()
    screen = QApplication.primaryScreen().availableGeometry()
    assert width <= max(
        layout_config.MAIN_WINDOW_MIN_WIDTH, screen.width() - 40
    )
    assert height <= max(
        layout_config.MAIN_WINDOW_MIN_HEIGHT, screen.height() - 40
    )
    assert width >= layout_config.MAIN_WINDOW_MIN_WIDTH
    assert height >= layout_config.MAIN_WINDOW_MIN_HEIGHT


def test_default_window_is_smaller_than_before():
    assert layout_config.MAIN_WINDOW_WIDTH <= 1000
    assert layout_config.MAIN_WINDOW_HEIGHT <= 700


_INPUT_TYPES = (
    QLabel,
    QPushButton,
    QCheckBox,
    QRadioButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
)


def _squeezed_widgets(win):
    found = []
    for scroll in win.findChildren(QScrollArea):
        if scroll.isVisible() and scroll.horizontalScrollBar().isVisible():
            found.append(f"horizontal scroll in {scroll.parent().objectName()}")
    for widget in win.findChildren(QWidget):
        if not widget.isVisible() or not isinstance(widget, _INPUT_TYPES):
            continue
        if isinstance(widget, QLabel) and widget.wordWrap():
            continue
        hint = widget.minimumSizeHint()
        if (
            widget.height() < hint.height() - 1
            or widget.width() < hint.width() - 1
        ):
            found.append(
                f"{type(widget).__name__} {getattr(widget, 'text', lambda: '')()!r} {widget.width()}x{widget.height()} "
                f"needs {hint.width()}x{hint.height()}"
            )
    return found


@pytest.mark.parametrize("point_size", [None, 12])
def test_no_page_is_squeezed_or_scrolls_sideways_at_the_minimum(
    window, point_size
):
    assert FONTS_AVAILABLE, (
        "Qt has no fonts, so every glyph is a square box and text widths "
        "mean nothing; install a TTF font or point QT_QPA_FONTDIR at one"
    )
    app = QApplication.instance()
    original = app.font()
    if point_size:
        font = QFont(original)
        font.setPointSize(point_size)
        app.setFont(font)
    try:
        window.resize(
            layout_config.MAIN_WINDOW_MIN_WIDTH,
            layout_config.MAIN_WINDOW_MIN_HEIGHT,
        )
        problems = []
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            pages = [window.tabs.tabText(index)]
            if window.tabs.widget(index) is window.stacked_content.parent():
                pages = [
                    f"EFFECTS/{window.list_categories.item(row).text()}"
                    for row in range(window.list_categories.count())
                ]
            for position, label in enumerate(pages):
                if len(pages) > 1:
                    window.list_categories.setCurrentRow(position)
                QApplication.processEvents()
                problems += [f"{label}: {p}" for p in _squeezed_widgets(window)]
        assert not problems
    finally:
        app.setFont(original)
        QApplication.processEvents()
