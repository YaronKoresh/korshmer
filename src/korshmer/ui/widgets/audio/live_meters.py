from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

from korshmer.audio_processing.live_levels import (
    SILENCE_DB,
    Levels,
    measure_levels,
)

REFRESH_MS = 120
PLAYING = 1
METER_FLOOR_DB = -60.0
READOUTS = (
    ("momentary_lufs", "LUFS-M", "Loudness of the last 0.4 seconds"),
    ("short_term_lufs", "LUFS-S", "Loudness of the last 3 seconds"),
    ("rms_db", "RMS", "Average level of the last 0.4 seconds"),
    ("peak_db", "Peak", "Highest sample of the last 0.4 seconds"),
    ("crest_db", "Crest", "Peak minus RMS: how punchy the sound is"),
    (
        "correlation",
        "Stereo",
        "+1 is mono, 0 is wide, below 0 has phase trouble",
    ),
)
VALUE_STYLE = (
    "font-family: 'Consolas', monospace; font-size: 11px; "
    "font-weight: bold; color: #202124;"
)
READOUT_COLUMNS = 3
READOUT_GAP = 12
PAIR_GAP = 3
NAME_STYLE = "font-size: 10px; color: #5F6368;"
BAR_STYLE = """
    QProgressBar { border: 1px solid #DADCE0; border-radius: 3px;
        background: #F1F3F4; }
    QProgressBar::chunk { background: %s; border-radius: 2px; }
"""
BAR_OK = "#34A853"
BAR_HOT = "#EA4335"


def format_reading(name, value):
    if name == "correlation":
        return f"{value:+.2f}"
    if value <= SILENCE_DB:
        return "-inf"
    return f"{value:.1f}"


class LiveMeters(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None
        self.levels = Levels()
        self.values = {}
        self.titles = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet(BAR_STYLE % BAR_OK)
        layout.addWidget(self.bar, 0, 0, 1, READOUT_COLUMNS)
        layout.setHorizontalSpacing(READOUT_GAP)
        for index, (name, label, tip) in enumerate(READOUTS):
            cell = QWidget()
            cell.setToolTip(tip)
            pair = QHBoxLayout(cell)
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(PAIR_GAP)
            title = QLabel(label)
            title.setStyleSheet(NAME_STYLE)
            value = QLabel()
            value.setStyleSheet(VALUE_STYLE)
            value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            pair.addWidget(title)
            pair.addWidget(value)
            pair.addStretch()
            layout.addWidget(
                cell, 1 + index // READOUT_COLUMNS, index % READOUT_COLUMNS
            )
            self.values[name] = value
            self.titles[name] = title
        self.show_levels(self.levels)
        self.timer = QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self.refresh)

    def set_engine(self, engine):
        self.engine = engine
        self.timer.start()

    def refresh(self):
        engine = self.engine
        if engine is None or engine.playbackState() != PLAYING:
            return
        self.show_levels(measure_levels(engine.data, engine.fs, engine.idx))

    def show_levels(self, levels):
        self.levels = levels
        for name, label in self.values.items():
            label.setText(format_reading(name, getattr(levels, name)))
        span = -METER_FLOOR_DB
        fill = (max(METER_FLOOR_DB, levels.peak_db) - METER_FLOOR_DB) / span
        self.bar.setValue(int(round(100 * min(1.0, fill))))
        self.bar.setStyleSheet(
            BAR_STYLE % (BAR_HOT if levels.clipping else BAR_OK)
        )
