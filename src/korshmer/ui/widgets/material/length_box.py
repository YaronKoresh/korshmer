import re

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from korshmer.ui.widgets.material.adaptive import FlowLayout

STEP_SECONDS = 15
QUICK_LENGTHS = (30, 60, 90, 120, 180, 240, 300)
BEATS_PER_BAR = 4
CLOCK = re.compile(r"^\s*(\d{1,3})\s*:\s*([0-5]?\d)\s*$")
PARTIAL_CLOCK = re.compile(r"^\s*\d{0,3}\s*:?\s*\d{0,2}\s*$")
PLAIN = re.compile(r"^\s*(\d+(?:[.,]\d*)?)\s*(s|sec|m|min)?\s*$", re.IGNORECASE)
LENGTH_HELP = (
    "Type minutes and seconds such as 2:30, or just minutes such as 2.5, "
    "or seconds such as 90s. The arrows change it by 15 seconds."
)


def clock_text(minutes):
    seconds = int(round(float(minutes) * 60.0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def minutes_from_text(text):
    clock = CLOCK.match(text)
    if clock:
        return int(clock.group(1)) + int(clock.group(2)) / 60.0
    plain = PLAIN.match(text)
    if not plain:
        return None
    amount = float(plain.group(1).replace(",", "."))
    unit = (plain.group(2) or "m").lower()
    return amount / 60.0 if unit.startswith("s") else amount


class LengthBox(QDoubleSpinBox):
    def __init__(self, minimum, maximum, value, parent=None):
        super().__init__(parent)
        self.setDecimals(4)
        self.setRange(float(minimum), float(maximum))
        self.setSingleStep(STEP_SECONDS / 60.0)
        self.setValue(float(value))
        self.setToolTip(LENGTH_HELP)
        self.setAccessibleName("Length in minutes and seconds")

    def textFromValue(self, value):
        return clock_text(value)

    def valueFromText(self, text):
        minutes = minutes_from_text(text)
        return self.value() if minutes is None else minutes

    def validate(self, text, position):
        if minutes_from_text(text) is not None:
            return QValidator.Acceptable, text, position
        if PARTIAL_CLOCK.match(text) or PLAIN.match(text + "0"):
            return QValidator.Intermediate, text, position
        return QValidator.Invalid, text, position


class LengthPicker(QWidget):
    def __init__(self, minimum, maximum, value, tempo=120.0, parent=None):
        super().__init__(parent)
        self.tempo = float(tempo)
        self.box = LengthBox(minimum, maximum, value)
        self.bars = QLabel("")
        self.quick = {}
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.box)
        row = FlowLayout(spacing=4)
        for seconds in QUICK_LENGTHS:
            minutes = seconds / 60.0
            if not minimum <= minutes <= maximum:
                continue
            button = QPushButton(clock_text(minutes))
            button.setCheckable(True)
            button.setToolTip(f"Set the length to {clock_text(minutes)}")
            button.setAccessibleName(f"Length {clock_text(minutes)}")
            button.clicked.connect(
                lambda _=False, minutes=minutes: self.box.setValue(minutes)
            )
            self.quick[seconds] = button
            row.addWidget(button)
        column.addLayout(row)
        column.addWidget(self.bars)
        self.box.valueChanged.connect(self._refresh)
        self._refresh()

    def set_tempo(self, tempo):
        if tempo and float(tempo) > 0:
            self.tempo = float(tempo)
            self._refresh()

    def bar_count(self):
        return int(round(self.box.value() * self.tempo / BEATS_PER_BAR))

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.box.setEnabled(enabled)

    def _refresh(self, *_):
        seconds = int(round(self.box.value() * 60.0))
        for wanted, button in self.quick.items():
            button.setChecked(wanted == seconds)
        self.bars.setText(
            f"About {self.bar_count()} bars at {self.tempo:g} BPM"
        )
