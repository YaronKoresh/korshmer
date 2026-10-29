from PySide6.QtCore import QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect

CARD_MIN_HEIGHT = 60


class MaterialCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)

        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QSize(hint.width(), max(hint.height(), CARD_MIN_HEIGHT))
