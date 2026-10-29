from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

try:
    from korshmer.ui.styling import MAT_TEXT
except ImportError:
    MAT_TEXT = "#111111BB"


class MarqueeLabel(QWidget):
    def __init__(self, text="", parent=None, speed=1, color=MAT_TEXT):
        super().__init__(parent)
        self._text = text
        self._speed = speed
        self._color = color
        self._offset = 0
        self._direction = -1
        self._wait_counter = 30
        self.setMinimumWidth(32)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.update_scroll)

    def setText(self, text):
        self._text = text
        self._offset = 0
        self._wait_counter = 30
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font())
        painter.setPen(QColor(self._color))

        fm = QFontMetrics(self.font())
        text_width = fm.horizontalAdvance(self._text)
        rect = self.rect()

        y_pos = (rect.height() - fm.height()) // 2 + fm.ascent()

        if text_width > rect.width():
            if not self.timer.isActive():
                self.timer.start()
            painter.drawText(self._offset, y_pos, self._text)
        else:
            if self.timer.isActive():
                self.timer.stop()
            self._offset = 0
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, self._text)

    def update_scroll(self):
        fm = QFontMetrics(self.font())
        text_width = fm.horizontalAdvance(self._text)
        widget_width = self.width()

        if self._wait_counter > 0:
            self._wait_counter -= 1
            return

        self._offset += self._direction * self._speed

        if self._direction == -1:
            if self._offset <= (widget_width - text_width - 20):
                self._direction = 1
                self._wait_counter = 40
        else:
            if self._offset >= 0:
                self._direction = -1
                self._wait_counter = 40
        self.update()
