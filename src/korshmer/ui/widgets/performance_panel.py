from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

PERFORMANCE_BUTTON_MIN_HEIGHT = 64


class PerformancePanel(QWidget):
    play_pause_clicked = Signal()
    stop_clicked = Signal()
    tempo_up_clicked = Signal()
    tempo_down_clicked = Signal()
    previous_scene_clicked = Signal()
    next_scene_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.btn_previous_scene = self._make_button("⏮︎ Scene")
        self.btn_play_pause = self._make_button("▶︎ / ❚❚︎")
        self.btn_stop = self._make_button("■︎ Stop")
        self.btn_tempo_down = self._make_button("Tempo −")
        self.btn_tempo_up = self._make_button("Tempo +")
        self.btn_next_scene = self._make_button("Scene ⏭︎")

        self.btn_previous_scene.clicked.connect(self.previous_scene_clicked)
        self.btn_play_pause.clicked.connect(self.play_pause_clicked)
        self.btn_stop.clicked.connect(self.stop_clicked)
        self.btn_tempo_down.clicked.connect(self.tempo_down_clicked)
        self.btn_tempo_up.clicked.connect(self.tempo_up_clicked)
        self.btn_next_scene.clicked.connect(self.next_scene_clicked)

        for button in (
            self.btn_previous_scene,
            self.btn_play_pause,
            self.btn_stop,
            self.btn_tempo_down,
            self.btn_tempo_up,
            self.btn_next_scene,
        ):
            layout.addWidget(button, 1)

    def buttons(self):
        return self.findChildren(QPushButton)

    @staticmethod
    def _make_button(text):
        button = QPushButton(text)
        button.setMinimumHeight(PERFORMANCE_BUTTON_MIN_HEIGHT)
        button.setStyleSheet("font-size: 20px; font-weight: bold;")
        return button
