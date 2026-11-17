from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from korshmer.ui.widgets.material.adaptive import FlowLayout
from korshmer.ui.widgets.material.card import MaterialCard

CHOICE_STYLE = """
    QPushButton {
        background-color: white;
        color: #3C4043;
        border: 1px solid #DADCE0;
        border-radius: 13px;
        padding: 4px 12px;
        font-size: 12px;
    }
    QPushButton:hover { background-color: #F1F3F4; }
    QPushButton:checked {
        background-color: #E8F0FE;
        color: #1A73E8;
        border: 1px solid #1A73E8;
        font-weight: bold;
    }
"""
HINT_STYLE = "color: #5F6368; font-size: 11px;"
SECTION_TITLE_STYLE = (
    "color: #5F6368; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
)
CHOICE_HEIGHT = 26
CHOICE_FONT_PX = 12
CHOICE_SIDE_PADDING = 28


class ChoiceBar(QWidget):
    def __init__(self, combo, hints=None, parent=None):
        super().__init__(parent)
        self.combo = combo
        self.hints = dict(hints or {})
        self.buttons = []
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QWidget()
        flow = FlowLayout(row, spacing=6)
        for index in range(combo.count()):
            button = QPushButton(combo.itemText(index))
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(CHOICE_HEIGHT)
            button.setStyleSheet(CHOICE_STYLE)
            button.setMinimumWidth(self._bold_width(button))
            button.setFocusPolicy(Qt.TabFocus)
            self._group.addButton(button, index)
            flow.addWidget(button)
            self.buttons.append(button)
        layout.addWidget(row)
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(HINT_STYLE)
        self.hint_label.setVisible(bool(self.hints))
        layout.addWidget(self.hint_label)
        self._group.idClicked.connect(combo.setCurrentIndex)
        combo.currentIndexChanged.connect(self._follow)
        self._follow(combo.currentIndex())

    def _bold_width(self, button):
        font = QFont(button.font())
        font.setPixelSize(CHOICE_FONT_PX)
        font.setBold(True)
        text_width = QFontMetrics(font).horizontalAdvance(button.text())
        return text_width + CHOICE_SIDE_PADDING

    def labels(self):
        return [button.text() for button in self.buttons]

    def checked_label(self):
        button = self._group.checkedButton()
        return button.text() if button is not None else ""

    def _follow(self, index):
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)
            self.hint_label.setText(
                self.hints.get(self.buttons[index].text(), "")
            )


class SectionCard(MaterialCard):
    def __init__(self, title, hint="", parent=None):
        super().__init__(parent)
        self.title = str(title)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(8)
        self.title_label = QLabel(self.title.upper())
        self.title_label.setStyleSheet(SECTION_TITLE_STYLE)
        outer.addWidget(self.title_label)
        self.hint_label = QLabel(hint)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(HINT_STYLE)
        self.hint_label.setVisible(bool(hint))
        outer.addWidget(self.hint_label)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        outer.addLayout(self.body)
        outer.addStretch()
