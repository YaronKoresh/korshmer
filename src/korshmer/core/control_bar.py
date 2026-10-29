from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from korshmer.core.audio_engine import AudioEngine
from korshmer.ui.widgets.marquee import MarqueeLabel
from korshmer.ui.widgets.material.button import MaterialButton
from korshmer.ui.widgets.material.progress import SteppedProgressBar
from korshmer.ui.widgets.material.slider import ClickableSlider

try:
    from korshmer.ui.styling import MAT_ERROR, MAT_WARNING
except ImportError:
    MAT_WARNING = "#FBBC04"
    MAT_ERROR = "#EA4335"


class ControlBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setMaximumHeight(140)

        self.player = AudioEngine()
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        self.progress_bar = SteppedProgressBar(steps=5)
        self.progress_bar.setVisible(False)

        self.btn_play = MaterialButton("▶\ufe0e", "primary")
        self.btn_stop = MaterialButton("■\ufe0e", "error")
        self.btn_panic = QPushButton("⚠\ufe0e")
        self.btn_ab = MaterialButton("A/B", "tonal")

        for b in [self.btn_play, self.btn_stop, self.btn_panic, self.btn_ab]:
            b.setFixedSize(36, 36)

        self.btn_ab.setStyleSheet("font-size: 11px;")

        self.btn_panic.setStyleSheet(
            f"""
            QPushButton {{
                background: {MAT_WARNING};
            }}
            QPushButton:hover {{
                background: {MAT_ERROR};
            }}
        """
        )

        self.btn_ab.setCheckable(True)

        self.slider_seek = ClickableSlider(Qt.Horizontal)
        self.slider_seek.setLayoutDirection(Qt.LeftToRight)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_sys = QLabel("Sys: Init...")
        self.lbl_sys.setMinimumWidth(
            self.lbl_sys.fontMetrics().horizontalAdvance(
                "CPU: 100.0% | RAM: 100.0%"
            )
        )

        self.btn_convert = MaterialButton("Start Processing", role="primary")
        self.btn_convert.setFixedWidth(140)
        self.btn_convert.setFixedHeight(40)

        self.lbl_status = MarqueeLabel("Ready", speed=1)

        controls_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.btn_panic)
        row1.addWidget(self.btn_ab)
        row1.addWidget(self.btn_stop)
        row1.addWidget(self.btn_play)
        controls_layout.addLayout(row1)
        controls_layout.addWidget(self.lbl_sys)
        controls_layout.setAlignment(self.lbl_sys, Qt.AlignCenter)

        seek_layout = QVBoxLayout()
        seek_layout.addWidget(self.lbl_status)

        h_seek = QHBoxLayout()
        h_seek.addWidget(self.slider_seek)
        h_seek.addWidget(self.lbl_time)
        h_seek.addWidget(self.btn_convert)
        seek_layout.addLayout(h_seek)

        seek_layout.addWidget(self.progress_bar)

        layout.addLayout(controls_layout)
        layout.addLayout(seek_layout)

        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_audio)

        self.slider_is_pressed = False
        self.slider_seek.sliderPressed.connect(self.on_slider_pressed)
        self.slider_seek.sliderReleased.connect(self.on_slider_released)
        self.slider_seek.valueChanged.connect(self.on_slider_moved)

        self.sys_timer = QTimer(self)
        self.sys_timer.timeout.connect(self.update_sys_stats)
        self.sys_timer.start(500)

    def update_sys_stats(self):
        try:
            import psutil

            if not hasattr(self, "_proc"):
                self._proc = psutil.Process()
                self._proc.cpu_percent()

            cpu = self._proc.cpu_percent() / psutil.cpu_count()
            ram = psutil.virtual_memory().percent
            self.lbl_sys.setText(f"CPU: {cpu:.1f}% | RAM: {ram:.1f}%")
        except ImportError:
            self.lbl_sys.setText("Monitor: N/A")
        except Exception:
            self.lbl_sys.setText("Monitor: Error")

    def load_file(self, path):
        self.stop_audio()
        if not path:
            return
        if self.player.load(path):
            self.slider_seek.setRange(0, self.player.duration())
            self.lbl_time.setText(
                f"00:00 / {self.fmt_time(self.player.duration())}"
            )

    def toggle_play(self):
        if self.player.data is None:
            return
        if self.player.playbackState() == 1:
            self.player.pause()
        else:
            self.player.play()

    def stop_audio(self):
        self.player.stop()

    def on_position_changed(self, position):
        if not self.slider_is_pressed:
            self.slider_seek.setValue(position)
            self.update_time_label(position, self.player.duration())

    def on_playback_state_changed(self, state):
        if state == 1:
            self.btn_play.setText("❚❚\ufe0e")
        else:
            self.btn_play.setText("▶\ufe0e")

    def on_slider_pressed(self):
        self.slider_is_pressed = True

    def on_slider_released(self):
        self.slider_is_pressed = False
        target_pos = self.slider_seek.value()
        self.player.setPosition(target_pos)
        self.update_time_label(target_pos, self.player.duration())

    def on_slider_moved(self, value):
        self.update_time_label(value, self.player.duration())

    def update_time_label(self, current_ms, total_ms):
        self.lbl_time.setText(
            f"{self.fmt_time(current_ms)} / {self.fmt_time(total_ms)}"
        )

    def fmt_time(self, ms):
        seconds = (ms // 1000) % 60
        minutes = ms // 60000
        return f"{minutes:02}:{seconds:02}"
