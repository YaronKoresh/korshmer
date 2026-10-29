from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRectF,
    Qt,
)
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

try:
    from korshmer.ui.styling import MAT_BG, MAT_BORDER, MAT_PRIMARY
except ImportError:
    MAT_BORDER = "#222"
    MAT_PRIMARY = "#1A73E8"
    MAT_BG = "#FFFFFFBB"


class SteppedProgressBar(QWidget):
    def __init__(self, steps=5, parent=None):
        super().__init__(parent)
        self.total_steps = steps
        self._progress = 0.0
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.anim = QPropertyAnimation(self, b"progress_val")
        self.anim.setEasingCurve(QEasingCurve.OutQuad)
        self.anim.setDuration(400)

    def get_progress(self):
        return self._progress

    def set_progress(self, val):
        self._progress = val
        self.update()

    progress_val = Property(float, get_progress, set_progress)

    def setValue(self, value):
        self.anim.stop()
        self.anim.setStartValue(self._progress)
        self.anim.setEndValue(value)
        self.anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        padding = 15
        available_w = w - (padding * 2)
        step_gap = available_w / (self.total_steps - 1)
        cy = h / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(MAT_BORDER))
        bg_rect = QRectF(padding, cy - 2, available_w, 4)
        painter.drawRoundedRect(bg_rect, 2, 2)
        if self._progress > 0:
            fill_ratio = self._progress / 100.0
            fill_width = available_w * fill_ratio
            grad = QLinearGradient(padding, cy, padding + fill_width, cy)
            grad.setColorAt(0, QColor(MAT_PRIMARY))
            grad.setColorAt(1, QColor("#4285F4"))
            painter.setBrush(grad)
            fill_rect = QRectF(padding, cy - 2, fill_width, 4)
            painter.drawRoundedRect(fill_rect, 2, 2)
        for i in range(self.total_steps):
            cx = padding + (i * step_gap)
            step_threshold = (i / (self.total_steps - 1)) * 100
            is_active = self._progress >= step_threshold
            if is_active:
                radius = 6
                painter.setBrush(QColor(MAT_PRIMARY))
                painter.setPen(QPen(QColor(MAT_BG), 2))
            else:
                radius = 4
                painter.setBrush(QColor("#BDC1C6"))
                painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(cx, cy), radius, radius)
