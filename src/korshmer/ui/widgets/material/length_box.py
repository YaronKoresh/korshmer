from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from korshmer.ui.widgets.material.adaptive import FlowLayout

BEATS_PER_BAR = 4
QUICK_BARS = (8, 16, 32, 48, 64, 96, 128)
BARS_HELP = (
    "How many bars to create. Use a quick button, the arrows, or type a "
    "number. The time below shows how long that is at the current BPM."
)


def clock_text(seconds):
    seconds = int(round(float(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


class BarsPicker(QWidget):
    def __init__(self, minimum, maximum, value, step, tempo=120.0, parent=None):
        super().__init__(parent)
        self.tempo = float(tempo)
        self.box = QSpinBox()
        self.box.setRange(int(minimum), int(maximum))
        self.box.setSingleStep(int(step))
        self.box.setValue(int(value))
        self.box.setSuffix(" bars")
        self.box.setToolTip(BARS_HELP)
        self.box.setAccessibleName("Number of bars to create")
        self.time = QLabel("")
        self.quick = {}
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.box)
        row = FlowLayout(spacing=4)
        for bars in QUICK_BARS:
            if not minimum <= bars <= maximum:
                continue
            button = QPushButton(str(bars))
            button.setCheckable(True)
            button.setToolTip(f"Create {bars} bars")
            button.setAccessibleName(f"{bars} bars")
            button.clicked.connect(
                lambda _=False, bars=bars: self.box.setValue(bars)
            )
            self.quick[bars] = button
            row.addWidget(button)
        column.addLayout(row)
        column.addWidget(self.time)
        self.box.valueChanged.connect(self._refresh)
        self._refresh()

    def set_tempo(self, tempo):
        if tempo and float(tempo) > 0:
            self.tempo = float(tempo)
            self._refresh()

    def seconds(self):
        return self.box.value() * BEATS_PER_BAR * 60.0 / self.tempo

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.box.setEnabled(enabled)

    def _refresh(self, *_):
        for bars, button in self.quick.items():
            button.setChecked(bars == self.box.value())
        self.time.setText(
            f"About {clock_text(self.seconds())} at {self.tempo:g} BPM"
        )
