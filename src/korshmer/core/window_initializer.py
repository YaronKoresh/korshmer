from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
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
)


def build_main_stylesheet():
    return f"""
            * {{
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
