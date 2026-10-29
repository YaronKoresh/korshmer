from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from korshmer.ui.widgets.material.button import MaterialButton
from korshmer.ui.widgets.material.slider import MaterialSlider


class RangeEQDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Range EQ Options")
        self.resize(350, 400)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Frequency Range (Hz):"))
        h_freq = QHBoxLayout()

        h_freq.addWidget(QLabel("Freq A:"))
        self.spin_a = QSpinBox()
        self.spin_a.setRange(1, 22000)
        self.spin_a.setValue(200)
        self.spin_a.setSuffix(" Hz")
        h_freq.addWidget(self.spin_a)

        h_freq.addWidget(QLabel("Freq B:"))
        self.spin_b = QSpinBox()
        self.spin_b.setRange(1, 22000)
        self.spin_b.setValue(800)
        self.spin_b.setSuffix(" Hz")
        h_freq.addWidget(self.spin_b)

        layout.addLayout(h_freq)

        self.slider_gain = MaterialSlider(
            "Gain (dB)",
            "Total Gain applied to range",
            min_val=-12,
            max_val=12,
            default=0,
            postfix="dB",
        )
        layout.addWidget(self.slider_gain)

        self.slider_peak = MaterialSlider(
            "Peak Position %",
            "Where the strongest EQ applies",
            min_val=1,
            max_val=99,
            default=50,
        )
        layout.addWidget(self.slider_peak)

        h_curve = QHBoxLayout()
        self.spin_curve_l = MaterialSlider(
            "Left Curve Shape",
            """<b>Left Slope:</b><br>
Controls the rise from the starting frequency to the peak.<br>
• High value: sharp and immediate rise.<br>
• Low value: soft and gradual rise.""",
            min_val=-10.0,
            max_val=10.0,
            default=0.0,
            postfix="",
        )
        h_curve.addWidget(self.spin_curve_l)

        self.spin_curve_r = MaterialSlider(
            "Right Curve Shape",
            """<b>Right Slope:</b><br>
Controls the slope from the peak to the end frequency.<br>
• High value: sharp slope (fast cut).<br>
• Low value: moderate and soft slope.""",
            min_val=-10.0,
            max_val=10.0,
            default=0.0,
            postfix="",
        )
        h_curve.addWidget(self.spin_curve_r)

        layout.addLayout(h_curve)
        layout.addStretch()

        btn_box = QHBoxLayout()
        btn_ok = MaterialButton("Process", role="primary")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = MaterialButton("Cancel", role="tonal")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)

    def get_params(self):
        return {
            "freq_start": float(self.spin_a.value()),
            "freq_end": float(self.spin_b.value()),
            "gain": float(self.slider_gain.value()),
            "peak_pos": float(self.slider_peak.value()),
            "curve_l": float(self.spin_curve_l.value()),
            "curve_r": float(self.spin_curve_r.value()),
        }
