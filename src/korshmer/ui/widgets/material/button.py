from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QSizePolicy

from korshmer.ui.styling import (
    MAT_BG,
    MAT_ERROR,
    MAT_HOVER,
    MAT_PRIMARY,
    MAT_SUCCESS,
    MAT_TEXT,
)

DISABLED_BG = "#AEB8C6"
DISABLED_TEXT = "#2E3744"


class MaterialButton(QPushButton):
    def __init__(self, text, role="default", parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setCursor(Qt.PointingHandCursor)
        self.set_role(role)

    def set_role(self, role):
        self.role = role
        if role == "primary":
            self.bg_normal = MAT_PRIMARY
            self.bg_hover = "#1557B0"
            self.text_col = "white"
        elif role == "success":
            self.bg_normal = MAT_SUCCESS
            self.bg_hover = "#2D8E47"
            self.text_col = "white"
        elif role == "danger" or role == "error":
            self.bg_normal = MAT_ERROR
            self.bg_hover = "#B71C1C"
            self.text_col = "white"
        elif role == "tonal":
            self.bg_normal = "#E8F0FE"
            self.bg_hover = "#D2E3FC"
            self.text_col = MAT_PRIMARY
        else:
            self.bg_normal = MAT_BG
            self.bg_hover = MAT_HOVER
            self.text_col = MAT_TEXT

        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.bg_normal};
                color: {self.text_col};
            }}

            QPushButton:hover {{
                background-color: {self.bg_hover};
            }}

            QPushButton:pressed {{
                padding-top: 2px;
            }}

            QPushButton:disabled {{
                background-color: {DISABLED_BG};
                color: {DISABLED_TEXT};
            }}
        """
        )


class MaterialToggleButton(QPushButton):
    def __init__(self, text, color=MAT_PRIMARY, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedSize(32, 32)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setCursor(Qt.PointingHandCursor)
        self.active_color = color
        self.update_style()
        self.toggled.connect(self.update_style)

    def update_style(self):
        if self.isChecked():
            bg = "#E8F0FE" if self.active_color == MAT_PRIMARY else "#FCE8E6"
            fg = self.active_color
        else:
            bg = MAT_BG
            fg = MAT_TEXT
        self.setStyleSheet(
            f"""
            QPushButton {{ background-color: {bg}; color: {fg}; }}
        """
        )


class HelpButton(QPushButton):
    def __init__(self, tooltip_text, parent=None):
        super().__init__("?", parent)
        self.setFixedSize(18, 18)
        self.setToolTip(tooltip_text)
        self.setCursor(Qt.WhatsThisCursor)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.setGraphicsEffect(None)
        self.raise_()
        self.setStyleSheet(
            """
            HelpButton {
                background-color: #1976D2;
                color: white;
                border-radius: 9px;
                font-weight: bold;
                font-size: 12px;
            }
            HelpButton:hover {
                background-color: #1565C0;
            }
            """
        )

    def paintEvent(self, event):

        super().paintEvent(event)
