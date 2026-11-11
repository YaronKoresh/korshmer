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


class TestWindowFrame:
    def _window(self, qapp):
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        window = QMainWindow()
        init_window_chrome(window)
        central = QWidget()
        QVBoxLayout(central)
        window.setCentralWidget(central)
        window.resize(200, 120)
        return window

    def test_edge_is_solid_and_inside_is_translucent(self, qapp):
        from PySide6.QtGui import QImage, QPainter

        from korshmer.core.window_initializer import paint_window_frame
        from korshmer.ui.styling import WINDOW_EDGE, WINDOW_FILL

        window = self._window(qapp)
        image = QImage(window.size(), QImage.Format_ARGB32)
        image.fill(0)
        with QPainter(image) as painter:
            paint_window_frame(window, painter)
        edge = image.pixelColor(100, 0)
        middle = image.pixelColor(100, 60)
        corner = image.pixelColor(0, 0)
        assert edge.alpha() == 255
        assert (edge.red(), edge.green(), edge.blue()) == (
            WINDOW_EDGE.red(),
            WINDOW_EDGE.green(),
            WINDOW_EDGE.blue(),
        )
        assert middle.alpha() == WINDOW_FILL.alpha() < 255
        assert corner.alpha() == 0

    def test_content_is_inset_only_while_framed(self, qapp):
        from korshmer.core.window_initializer import (
            frame_margin,
            update_frame_margin,
        )
        from korshmer.ui.styling import WINDOW_EDGE_WIDTH

        window = self._window(qapp)
        update_frame_margin(window)
        assert frame_margin(window) == WINDOW_EDGE_WIDTH
        margins = window.centralWidget().layout().contentsMargins()
        assert margins.left() == WINDOW_EDGE_WIDTH

    def test_containers_are_transparent_but_inputs_stay_opaque(self):
        css = build_main_stylesheet()
        first_rule = css.split("}")[0]
        assert "transparent" in first_rule
        assert "QLineEdit" in css and "QMenu" in css


class TestDragFromMaximized:
    def _window(self, qapp):
        window = QMainWindow()
        init_window_chrome(window)
        window.resize(600, 400)
        window.show()
        qapp.processEvents()
        return window

    def test_dragging_a_maximized_window_restores_it(self, qapp):
        from PySide6.QtCore import QPoint

        from korshmer.core.window_initializer import drag_window_to

        window = self._window(qapp)
        window.showMaximized()
        qapp.processEvents()
        full_width = window.width()
        window.moving_window = True
        window.drag_pos = QPoint(full_width // 2, 10)
        press = window.drag_pos + window.frameGeometry().topLeft()

        drag_window_to(window, press + QPoint(1, 1))
        assert window.isMaximized()

        share = window.drag_pos.x() / window.width()
        normal_width = window.normalGeometry().width()
        drag_window_to(window, press + QPoint(80, 60))
        qapp.processEvents()
        assert not window.isMaximized()
        assert window.drag_pos.x() == round(share * normal_width)
        assert window.drag_pos.y() == 10
        window.close()

    def test_dragging_a_normal_window_just_moves_it(self, qapp):
        from PySide6.QtCore import QPoint

        from korshmer.core.window_initializer import drag_window_to

        window = self._window(qapp)
        window.drag_pos = QPoint(50, 10)
        drag_window_to(window, QPoint(300, 200))
        assert window.pos() == QPoint(250, 190)
        drag_window_to(window, QPoint(301, 200))
        assert window.pos() == QPoint(251, 190)
        window.drag_pos = None
        drag_window_to(window, QPoint(0, 0))
        assert window.pos() == QPoint(251, 190)
        window.close()
