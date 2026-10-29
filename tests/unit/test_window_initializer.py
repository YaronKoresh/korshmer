import ast
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow

from korshmer.core.window_initializer import (
    build_main_stylesheet,
    init_window_chrome,
    initial_window_size,
)
from korshmer.ui.styling import (
    MAT_BG,
    MAT_BORDER,
    MAT_HOVER,
    MAT_PRIMARY,
    MAT_TEXT,
)

MAIN_PY = Path(__file__).parents[2] / "src" / "korshmer" / "__main__.py"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


class _RecordingWindow(QMainWindow):
    def __init__(self):
        self.filter_seen_state = None
        super().__init__()

    def eventFilter(self, obj, event):
        if self.filter_seen_state is None:
            self.filter_seen_state = (
                self.resize_direction,
                self.moving_window,
                self.drag_pos,
            )
        return False


class TestBuildMainStylesheet:
    def test_uses_material_colors(self):
        css = build_main_stylesheet()
        for color in (MAT_BG, MAT_BORDER, MAT_HOVER, MAT_PRIMARY, MAT_TEXT):
            assert color in css

    def test_braces_are_single_and_balanced(self):
        css = build_main_stylesheet()
        assert "{{" not in css
        assert "}}" not in css
        assert css.count("{") == css.count("}")
        assert "QMessageBox QPushButton:hover {" in css


class TestInitWindowChrome:
    def test_configures_frameless_window(self, qapp):
        window = _RecordingWindow()
        init_window_chrome(window)
        try:
            assert window.windowTitle() == "KORSHMER"
            assert (window.width(), window.height()) == initial_window_size()
            assert window.layoutDirection() == Qt.LeftToRight
            assert window.windowFlags() & Qt.FramelessWindowHint
            assert window.testAttribute(Qt.WA_TranslucentBackground)
            assert window.testAttribute(Qt.WA_Hover)
            assert window.styleSheet() == build_main_stylesheet()
        finally:
            qapp.removeEventFilter(window)
            window.deleteLater()

    def test_state_attributes_exist_before_event_filter(self, qapp):
        window = _RecordingWindow()
        init_window_chrome(window)
        try:
            assert window.metal_texture is None
            assert window.resize_margin == 10
            assert window.resize_direction is None
            assert window.drag_pos is None
            assert window.start_geometry is None
            assert window.moving_window is False
            window.setStyleSheet(window.styleSheet())
            qapp.processEvents()
            assert window.filter_seen_state == (None, False, None)
        finally:
            qapp.removeEventFilter(window)
            window.deleteLater()


def test_main_module_no_longer_holds_stylesheet():
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    main_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    init = next(
        node
        for node in main_window.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = [
        node.func.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "setStyleSheet" not in calls
    assert "QMessageBox QPushButton:hover" not in MAIN_PY.read_text(
        encoding="utf-8"
    )
