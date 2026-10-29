from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QSizePolicy, QVBoxLayout

from korshmer.ui.widgets.material.button import MaterialButton


class ImagePreviewDialog(QDialog):
    def __init__(self, image_path, title="Image Preview", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1250, 700)

        layout = QVBoxLayout(self)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        pixmap = QImage(image_path)
        if not pixmap.isNull():
            px = QPixmap.fromImage(pixmap)
            self.lbl_image.setPixmap(
                px.scaled(
                    1250, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            self.lbl_image.setScaledContents(True)
        else:
            self.lbl_image.setText("Failed to load image.")

        layout.addWidget(self.lbl_image)

        btn_close = MaterialButton("Close", role="tonal")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
