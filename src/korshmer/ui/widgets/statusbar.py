from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

MESSAGE_SECONDS = 6


class EnhancedStatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)

        self.action_label = QLabel("")
        self.layout.addWidget(self.action_label)
        self.layout.addStretch()

        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.setInterval(MESSAGE_SECONDS * 1000)
        self._clear_timer.timeout.connect(self.clear_action)

        self.setStyleSheet(
            """
            QLabel {
                color: #b0b0b0;
                font-size: 10px;
                padding: 0 5px;
            }
        """
        )

    def _host(self):
        host = self.parentWidget()
        return host if isinstance(host, QStatusBar) else None

    def update_action(self, action):
        self.action_label.setText(action)
        host = self._host()
        if host is not None:
            host.setVisible(bool(action))
        if action:
            self._clear_timer.start()
        else:
            self._clear_timer.stop()

    def clear_action(self):
        self.update_action("")
