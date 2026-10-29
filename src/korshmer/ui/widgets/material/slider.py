from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

try:
    from korshmer.ui.styling import MAT_PRIMARY, MAT_TEXT
except ImportError:
    MAT_TEXT = "#111111BB"
    MAT_PRIMARY = "#1A73E8"

try:
    from korshmer.ui.widgets.material.button import HelpButton
except ImportError:
    from PySide6.QtWidgets import QPushButton

    class HelpButton(QPushButton):
        def __init__(self, tooltip):
            super().__init__("?")
            self.setToolTip(tooltip)
            self.setFixedSize(16, 16)


class MaterialSlider(QWidget):
    def __init__(
        self, title, tooltip, min_val=0, max_val=100, default=100, postfix="%"
    ):
        super().__init__()
        self.postfix = postfix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.lbl_name = QLabel(title)
        self.lbl_name.setStyleSheet(f"color: {MAT_TEXT};")
        self.lbl_val = QLabel(f"{default}{self.postfix}")
        self.lbl_val.setStyleSheet(f"color: {MAT_PRIMARY};")
        header.addWidget(self.lbl_name)
        header.addWidget(HelpButton(tooltip), alignment=Qt.AlignLeft)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setLayoutDirection(Qt.LeftToRight)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)
        self.slider.valueChanged.connect(self.on_change)
        layout.addLayout(header)
        layout.addWidget(self.slider)

    def on_change(self, val):
        self.lbl_val.setText(f"{val}{self.postfix}")

    def value(self):
        return self.slider.value()

    def setValue(self, val):
        self.slider.setValue(val)


class ClickableSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._is_dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = self.pixelPosToRangeValue(event.position().toPoint())
            self.setValue(val)

            self._is_dragging = True

            self.sliderPressed.emit()

            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() & Qt.LeftButton:
            val = self.pixelPosToRangeValue(event.position().toPoint())
            self.setValue(val)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_dragging:
            val = self.pixelPosToRangeValue(event.position().toPoint())
            self.setValue(val)
            self._is_dragging = False

            self.sliderReleased.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        gr = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        sr = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
        )
        if self.orientation() == Qt.Horizontal:
            sliderLength = sr.width()
            sliderMin = gr.x()
            sliderMax = gr.right() - sliderLength + 1
            pos = pos.x()
        else:
            sliderLength = sr.height()
            sliderMin = gr.y()
            sliderMax = gr.bottom() - sliderLength + 1
            pos = pos.y()
        return QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            pos - sliderMin,
            sliderMax - sliderMin,
            opt.upsideDown,
        )
