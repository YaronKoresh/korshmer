from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication

from korshmer.ui.layout_config import (
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_MIN_HEIGHT,
    MAIN_WINDOW_MIN_WIDTH,
    MAIN_WINDOW_WIDTH,
)
from korshmer.ui.styling import (
    MAT_BG,
    MAT_BORDER,
    MAT_HOVER,
    MAT_PRIMARY,
    MAT_TEXT,
    WINDOW_EDGE,
    WINDOW_EDGE_WIDTH,
    WINDOW_FILL,
    WINDOW_RADIUS,
)

OPAQUE_WIDGETS = (
    "QMenu, QMenuBar::item:selected, QComboBox QAbstractItemView, "
    "QDialog, QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, "
    "QDoubleSpinBox, QTableView, QTreeView, QHeaderView::section"
)


def build_main_stylesheet():
    return f"""
            * {{
                background: transparent;
                background-color: transparent;
            }}

            {OPAQUE_WIDGETS} {{
                background: {MAT_BG};
                background-color: {MAT_BG};
            }}

            QMessageBox {{
                background: {MAT_BG};
                background-color: {MAT_BG};
                border: 1px solid {MAT_BORDER};
            }}
            QMessageBox QLabel {{
                color: {MAT_TEXT};
                background: none;
                background-color: transparent;
                font-weight: normal;
            }}
            QMessageBox QPushButton {{
                background-color: {MAT_BG};
                color: {MAT_TEXT};
                border: 1px solid {MAT_BORDER};
                padding: 5px 15px;
                border-radius: 4px;
            }}
            QMessageBox QPushButton:hover {{
                background: {MAT_HOVER};
                background-color: {MAT_HOVER};
            }}

            QDialog, QHBoxLayout, QVBoxLayout,
            QTabWidget, QTabWidget::pane {{
                outline: none;
                color: {MAT_TEXT};
                background: {MAT_BG};
                background-color: {MAT_BG};
                padding: 0px;
                margin: 2px;
                border-radius: 4px;
                border: 1px solid {MAT_BORDER};
                font-family: 'Tahoma';
                font-weight: bold;
                font-size: 12px;
            }}

            QMainWindow {{
                background: transparent;
                background-color: transparent;
                border: none;
                border-color: transparent;
            }}

            QListWidget {{
                outline: none;
                color: {MAT_TEXT};
                background: {MAT_BG};
                background-color: {MAT_BG};
                border: none;
                padding: 4px;
            }}

            QListWidget::item {{
                padding: 4px;
                margin: 2px;
                border-radius: 4px;
                border: none;
            }}

            QListWidget::item:selected {{
                background-color: {MAT_HOVER};
                color: {MAT_PRIMARY};
                border: 1px solid {MAT_PRIMARY};
            }}

            QListWidget::item:hover {{
                background-color: {MAT_HOVER};
            }}

            QFormLayout , QFrame,
            QWidget QLabel,
            QWidget QPushButton,
            QWidget QRadioButton,
            QWidget QCheckBox,
            QFormLayout QLabel,
            QFormLayout QPushButton,
            QFormLayout QRadioButton,
            QFormLayout QCheckBox,
            QFrame QLabel,
            QFrame QPushButton,
            QFrame QRadioButton,
            QFrame QCheckBox {{
                background: transparent;
                background-color: transparent;
            }}

            QRadioButton, QLabel, QPushButton, QSlider, QTabBar {{
                outline: none;
                color: {MAT_TEXT};
                background: {MAT_BG};
                background-color: {MAT_BG};
                padding: 0px;
                margin: 0px;
                border-radius: 4px;
                border: none;
                font-family: 'Tahoma';
                font-size: 12px;
            }}

            QPushButton:hover {{
                background-color: {MAT_HOVER};
            }}

            QScrollBar:horizontal {{
                border: none;
                background: {MAT_BG};
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: {MAT_HOVER};
                min-width: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            QScrollBar:vertical {{
                border: none;
                background: {MAT_BG};
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {MAT_HOVER};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            QSlider::groove:horizontal {{
                height: 4px;
                background: {MAT_HOVER};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {MAT_PRIMARY};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {MAT_PRIMARY};
                border: 1px solid {MAT_PRIMARY};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {MAT_HOVER};
            }}

            QToolTip {{
                color: {MAT_TEXT};
                background-color: {MAT_BG};
                border: 1px solid {MAT_BORDER};
                padding: 5px;
            }}

            QComboBox {{
                min-width: 80px;
                background: {MAT_BG};
                padding: 2px 4px;
                border-radius: 4px;
            }}

            QCheckBox {{
                spacing: 8px;
                font-size: 11px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 1px solid #5F6368;
                border-radius: 3px;
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {MAT_PRIMARY};
                border-color: {MAT_PRIMARY};
            }}

            QSplitter::handle {{
                background-color: {MAT_BORDER};
            }}
        """


def initial_window_size():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT
    area = screen.availableGeometry()
    return (
        max(MAIN_WINDOW_MIN_WIDTH, min(MAIN_WINDOW_WIDTH, area.width() - 40)),
        max(
            MAIN_WINDOW_MIN_HEIGHT, min(MAIN_WINDOW_HEIGHT, area.height() - 40)
        ),
    )


def _framed(window):
    return not (window.isMaximized() or window.isFullScreen())


def frame_margin(window):
    return WINDOW_EDGE_WIDTH if _framed(window) else 0


def paint_window_frame(window, painter):
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    rect = QRectF(window.rect())
    if not _framed(window):
        painter.fillRect(rect, WINDOW_FILL)
        return
    half = WINDOW_EDGE_WIDTH / 2
    path = QPainterPath()
    path.addRoundedRect(
        rect.adjusted(half, half, -half, -half), WINDOW_RADIUS, WINDOW_RADIUS
    )
    painter.fillRect(rect, Qt.transparent)
    painter.fillPath(path, WINDOW_FILL)
    painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
    painter.setPen(QPen(WINDOW_EDGE, WINDOW_EDGE_WIDTH))
    painter.drawPath(path)


def update_frame_margin(window):
    central = window.centralWidget()
    layout = central.layout() if central is not None else None
    if layout is not None:
        m = frame_margin(window)
        layout.setContentsMargins(m, m, m, m)


def drag_window_to(window, global_pos):
    grab = window.drag_pos
    if grab is None:
        return
    if not _framed(window):
        press = grab + window.frameGeometry().topLeft()
        if (global_pos - press).manhattanLength() < (
            QApplication.startDragDistance()
        ):
            return
        share = grab.x() / max(1, window.width())

        normal = window.normalGeometry()
        width = normal.width() if normal.width() > 0 else window.width()
        window.showNormal()
        grab = QPoint(int(round(share * width)), grab.y())
        window.drag_pos = grab
    window.move(global_pos - grab)


def init_window_chrome(window):
    window.setWindowTitle("KORSHMER")
    window.resize(*initial_window_size())
    window.setLayoutDirection(Qt.LeftToRight)
    window.setWindowFlags(Qt.FramelessWindowHint)
    window.setAttribute(Qt.WA_TranslucentBackground)
    window.metal_texture = None
    window.resize_margin = 10
    window.resize_direction = None
    window.drag_pos = None
    window.start_geometry = None
    window.moving_window = False
    window.setAttribute(Qt.WA_Hover)
    QApplication.instance().installEventFilter(window)

    window.setStyleSheet(build_main_stylesheet())
