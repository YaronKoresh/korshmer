import librosa
import soundfile as sf
from PySide6.QtCore import QRectF, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from korshmer.ui.utils.throttle import EventThrottle
from korshmer.ui.viewmodels.spectrogram_vm import SpectrogramViewModel
from korshmer.ui.widgets.material import MAT_BG, MAT_PRIMARY, MAT_TEXT
from korshmer.visualization.spectrum import create_spectrogram_image


class GenerateSpectrogramThread(QThread):
    finished_signal = Signal(QImage)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        img = create_spectrogram_image(self.path)
        self.finished_signal.emit(img)


class SpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = SpectrogramViewModel()
        self.setMinimumHeight(40)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.spectrogram_image = None
        self.current_thread = None
        self.hover_y = -1
        self.hover_x = -1
        self.hover_text = ""
        self._hover_throttle = EventThrottle(self._update_hover, parent=self)

    def sizeHint(self):
        return QSize(200, 120)

    @property
    def loading(self):
        return self.vm.loading

    @property
    def duration(self):
        return self.vm.duration

    def set_audio(self, audio_path):
        if not self.vm.set_loading(audio_path):
            self.spectrogram_image = None
            self.update()
            return

        try:
            self.vm.duration = sf.info(audio_path).duration
        except Exception:
            self.vm.duration = 0

        if self.current_thread is not None:
            try:
                self.current_thread.finished_signal.disconnect()
            except Exception:
                pass
            self.current_thread.finished.connect(
                self.current_thread.deleteLater
            )
            self.current_thread = None

        self.spectrogram_image = None
        self.hover_text = ""
        self.update()

        self.current_thread = GenerateSpectrogramThread(audio_path, parent=self)
        self.current_thread.finished_signal.connect(self.on_spectrogram_ready)
        self.current_thread.start()

    @Slot(QImage)
    def on_spectrogram_ready(self, img):
        self.spectrogram_image = img
        self.vm.on_ready(img is not None and not img.isNull())
        self.update()

    def mouseMoveEvent(self, event):
        if not self.spectrogram_image:
            return
        pos = event.position()
        self._hover_throttle.submit(pos.x(), pos.y())

    def _update_hover(self, x, y):
        if not self.spectrogram_image:
            return
        self.hover_x = x
        self.hover_y = y

        info = self.vm.hover_info(
            self.hover_x, self.hover_y, self.width(), self.height()
        )
        if info:
            note_name = (
                librosa.hz_to_note(info["freq"]) if info["freq"] > 0 else ""
            )
            self.hover_text = (
                f"{info['time_str']} | {note_name} ({info['freq_int']}Hz)"
            )
        else:
            self.hover_text = ""
        self.update()

    def leaveEvent(self, event):
        self._hover_throttle.cancel()
        self.hover_y = -1
        self.hover_x = -1
        self.hover_text = ""
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.testAttribute(Qt.WA_StyledBackground):
            painter.fillRect(rect, QColor(MAT_BG))
        if self.vm.loading:
            painter.setPen(QColor(MAT_TEXT))
            painter.drawText(rect, Qt.AlignCenter, "Processing Audio...")
            return
        if self.spectrogram_image:
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), 12, 12)
            painter.setClipPath(path)
            painter.drawImage(rect, self.spectrogram_image)
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.DashLine))
            painter.setFont(QFont("Tahoma", 8))
            octaves = [130.81, 261.63, 523.25, 1046.50, 2093.00, 4186.01]
            labels = ["C3", "C4", "C5", "C6", "C7", "C8"]
            for freq, label in zip(octaves, labels, strict=False):
                y_pos = self.vm.hz_to_y(freq, rect.height())
                if 0 <= y_pos <= rect.height():
                    painter.drawLine(0, int(y_pos), rect.width(), int(y_pos))
                    painter.drawText(5, int(y_pos) - 2, label)

            if self.hover_y >= 0 and self.hover_text:
                painter.setPen(QPen(QColor(MAT_PRIMARY), 1))
                painter.drawLine(
                    0, int(self.hover_y), rect.width(), int(self.hover_y)
                )
                if self.hover_x >= 0:
                    painter.drawLine(
                        int(self.hover_x), 0, int(self.hover_x), rect.height()
                    )

                text_rect = painter.fontMetrics().boundingRect(self.hover_text)
                txt_x = rect.width() - text_rect.width() - 10
                txt_y = int(self.hover_y) - 15
                if txt_y < 5:
                    txt_y = int(self.hover_y) + 15

                text_rect.moveTo(txt_x, txt_y)
                text_rect.adjust(-4, -2, 4, 2)
                painter.setBrush(QColor(0, 0, 0, 180))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(text_rect, 4, 4)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.AlignCenter, self.hover_text)
        else:
            painter.setPen(QColor(0, 0, 0, 20))
            for i in range(0, rect.width(), 20):
                painter.drawLine(i, 0, i, rect.height())
            painter.setPen(QColor(MAT_TEXT))
            painter.setFont(QFont("Tahoma", 10))
            painter.drawText(rect, Qt.AlignCenter, "No Preview")

    def hz_to_y(self, freq, height):
        return self.vm.hz_to_y(freq, height)

    def y_to_hz(self, y, height):
        return self.vm.y_to_hz(y, height)
