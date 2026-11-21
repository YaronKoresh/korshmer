from PySide6.QtCore import Qt, QTimer, Signal
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
from korshmer.utils.audio_host import apply_audio_preferences

try:
    from korshmer.ui.styling import MAT_ERROR, MAT_WARNING
except ImportError:
    MAT_WARNING = "#FBBC04"
    MAT_ERROR = "#EA4335"


TIME_CLOCK = "clock"
TIME_MILLIS = "millis"
TIME_TIMECODE = "timecode"
TIME_BARS = "bars"
TIME_DISPLAY_NAMES = {
    TIME_CLOCK: "minutes:seconds",
    TIME_MILLIS: "minutes:seconds.milliseconds",
    TIME_TIMECODE: "hours:minutes:seconds",
    TIME_BARS: "bar.beat.tick",
}
TIME_DISPLAY_MODES = tuple(TIME_DISPLAY_NAMES)
HOUR_MS = 3600 * 1000
TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 120.0
DEFAULT_BEATS_PER_BAR = 4
TIME_LABEL_SLACK = 6


def format_clock(ms, with_hours, with_millis):
    seconds, millis = divmod(max(0, int(ms)), 1000)
    minutes, seconds = divmod(seconds, 60)
    if with_hours:
        hours, minutes = divmod(minutes, 60)
        text = f"{hours}:{minutes:02}:{seconds:02}"
    else:
        text = f"{minutes:02}:{seconds:02}"
    return f"{text}.{millis:03}" if with_millis else text


def format_bars(beats, beats_per_bar):
    ticks = int(round(max(0.0, float(beats)) * TICKS_PER_BEAT))
    whole_beats, tick = divmod(ticks, TICKS_PER_BEAT)
    bar, beat = divmod(whole_beats, max(1, int(beats_per_bar)))
    return f"{bar + 1}.{beat + 1}.{tick:03d}"


class TimeLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class ControlBar(QFrame):
    time_display_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setMaximumHeight(140)

        self.player = AudioEngine()
        self.audio_notice = apply_audio_preferences(self.player)
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

        self._time_mode = TIME_CLOCK
        self._tempo = DEFAULT_TEMPO
        self._beats_per_bar = DEFAULT_BEATS_PER_BAR
        self._beat_mapper = None
        self._shown_ms = 0
        self._total_ms = 0
        self._reserved_for = None
        self.lbl_time = TimeLabel()
        self.lbl_time.setCursor(Qt.PointingHandCursor)
        self.lbl_time.clicked.connect(self.cycle_time_display)
        self._describe_time_mode()
        self._render_time()
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
            self.update_time_label(0, self.player.duration())

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
        self._shown_ms = current_ms
        self._total_ms = total_ms
        self._render_time()

    def _beats_at(self, ms):
        if self._beat_mapper is not None:
            beats = self._beat_mapper(ms / 1000.0)
            if beats is not None:
                return beats
        return ms * self._tempo / 60000.0

    def fmt_time(self, ms):
        if self._time_mode == TIME_BARS:
            return format_bars(self._beats_at(ms), self._beats_per_bar)
        return format_clock(
            ms,
            self._time_mode == TIME_TIMECODE
            or max(self._shown_ms, self._total_ms) >= HOUR_MS,
            self._time_mode == TIME_MILLIS,
        )

    def _reserve_time_width(self, total_text):
        shape = (self._time_mode, len(total_text))
        if shape == self._reserved_for:
            return
        self._reserved_for = shape
        widest = "".join("0" if c.isdigit() else c for c in total_text)
        metrics = self.lbl_time.fontMetrics()
        self.lbl_time.setMinimumWidth(
            metrics.horizontalAdvance(f"{widest} / {widest}") + TIME_LABEL_SLACK
        )

    def _render_time(self):
        total_text = self.fmt_time(self._total_ms)
        self._reserve_time_width(total_text)
        text = f"{self.fmt_time(self._shown_ms)} / {total_text}"
        if text != self.lbl_time.text():
            self.lbl_time.setText(text)

    def _describe_time_mode(self):
        self.lbl_time.setToolTip(
            f"Time shown as {TIME_DISPLAY_NAMES[self._time_mode]}. "
            "Click to change the format."
        )

    def time_display_mode(self):
        return self._time_mode

    def set_time_display_mode(self, mode):
        if mode not in TIME_DISPLAY_MODES or mode == self._time_mode:
            return
        self._time_mode = mode
        self._describe_time_mode()
        self._render_time()
        self.time_display_changed.emit(mode)

    def cycle_time_display(self):
        index = TIME_DISPLAY_MODES.index(self._time_mode)
        self.set_time_display_mode(
            TIME_DISPLAY_MODES[(index + 1) % len(TIME_DISPLAY_MODES)]
        )

    def set_tempo(self, bpm):
        if bpm > 0 and bpm != self._tempo:
            self._tempo = float(bpm)
            self._render_time()

    def set_beats_per_bar(self, beats):
        if beats >= 1 and beats != self._beats_per_bar:
            self._beats_per_bar = int(beats)
            self._render_time()

    def set_beat_mapper(self, mapper):
        self._beat_mapper = mapper
        self._render_time()

    def go_to_start(self):
        self.player.setPosition(0)
        self.slider_seek.setValue(0)
        self.update_time_label(0, self.player.duration())
