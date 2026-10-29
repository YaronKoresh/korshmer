import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from korshmer.ui.managers.accessibility import (
    AccessibilityManager,
    apply_accessible_names,
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

CONTROL_TYPES = (QAbstractButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox)


def test_enable_screen_reader_updates_settings():
    manager = AccessibilityManager()
    assert manager.get_settings()["screen_reader"] is False
    manager.enable_screen_reader()
    assert manager.get_settings()["screen_reader"] is True


def test_names_come_from_text_tooltip_then_object_name():
    root = QWidget()
    with_text = QPushButton("&Play", root)
    with_tip = QPushButton("", root)
    with_tip.setToolTip("Undo (Ctrl+Z)\nsecond line")
    with_object = QSlider(root)
    with_object.setObjectName("volume")
    bare = QComboBox(root)
    assert apply_accessible_names(root) == 4
    assert with_text.accessibleName() == "Play"
    assert with_tip.accessibleName() == "Undo (Ctrl+Z)"
    assert with_object.accessibleName() == "volume"
    assert bare.accessibleName() == "Selection list"


def test_existing_accessible_name_is_kept():
    root = QWidget()
    button = QPushButton("Text", root)
    button.setAccessibleName("Custom")
    apply_accessible_names(root)
    assert button.accessibleName() == "Custom"


def test_main_window_controls_have_accessible_names(monkeypatch):
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    window = MainWindow()
    try:
        unnamed = [
            type(w).__name__
            for w in window.findChildren(QWidget)
            if isinstance(w, CONTROL_TYPES) and not w.accessibleName().strip()
        ]
        assert unnamed == []
    finally:
        window.close()
