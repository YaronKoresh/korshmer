import traceback

from PySide6.QtCore import QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from korshmer.ui.viewmodels.visualizer_vm import VisualizerViewModel
from korshmer.visualization.spectrum import calculate_waveform_features


class VisualizerDataWorker(QThread):
    analysis_finished = Signal(object, float)

    def __init__(self, audio_path, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path

    def run(self):
        try:
            data, duration = calculate_waveform_features(self.audio_path)
            self.analysis_finished.emit(data, duration)
        except Exception:
            traceback.print_exc()


class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = VisualizerViewModel()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

        self.worker = None

        self.setMinimumSize(120, 36)
        self.setMaximumHeight(100)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_StyledBackground, True)

    def sizeHint(self):
        return QSize(200, 100)

    @property
    def bars_count(self):
        return self.vm.bars_count

    @property
    def values(self):
        return self.vm.values

    @property
    def rms_data(self):
        return self.vm.rms_data

    @property
    def duration_sec(self):
        return self.vm.duration_sec

    @property
    def current_time_sec(self):
        return self.vm.current_time_sec

    @property
    def is_playing(self):
        return self.vm.is_playing

    @property
    def current_energy(self):
        return self.vm.current_energy

    @property
    def flash_state(self):
        return self.vm.flash_state

    def analyze_audio(self, path):
        if self.worker is not None:
            try:
                self.worker.analysis_finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self.worker = None

        self.vm.clear_analysis()
        self.worker = VisualizerDataWorker(path, self)
        self.worker.analysis_finished.connect(self.on_analysis_done)
        self.worker.start()

    def on_analysis_done(self, data, duration):
        self.vm.set_analysis(data, duration)

    def set_playing(self, playing):
        self.vm.is_playing = playing

    def set_position(self, ms):
        self.vm.set_position(ms)

    def set_volume_level(self, volume):
        self.vm.volume_scale = volume

    def animate(self):
        self.vm.tick()
        self.update()

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()

            alpha = int(self.vm.flash_alpha() * 255)
            painter.fillRect(self.rect(), QColor(50, 0, 100, alpha))

            bar_w = w / self.vm.bars_count
            gap = 1
            draw_w = max(1, bar_w - gap)

            r, g, b = self.vm.bar_color_rgb()
            bar_color = QColor(r, g, b)

            for i, val in enumerate(self.vm.values):
                bar_h = val * h * 0.95
                bar_h = max(3, bar_h)

                y = (h - bar_h) / 2
                x = i * bar_w + (gap / 2)

                grad = QLinearGradient(x, y, x, y + bar_h)
                grad.setColorAt(0, bar_color)
                grad.setColorAt(1, bar_color.lighter(130))

                painter.setBrush(grad)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(QRectF(x, y, draw_w, bar_h), 2, 2)
