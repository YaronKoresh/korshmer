from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QLabel, QScrollArea, QVBoxLayout

from korshmer.ui.widgets.material.button import MaterialButton


class StatsDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instrument Statistics")
        self.resize(400, 500)
        self.setMaximumHeight(600)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QLabel(text)
        content.setWordWrap(True)
        content.setTextFormat(Qt.RichText)
        content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        btn = MaterialButton("Close", role="primary")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
