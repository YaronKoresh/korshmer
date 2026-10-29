import sys

try:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except ImportError:
    QApplication = None
    QFont = None
    _QT_AVAILABLE = False

from korshmer.core.di_container import DIContainer as Container


class Application:
    def __init__(self, container: Container):
        self.container = container
        self._app = None

    def bootstrap(self):
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        font = QFont("Tahoma", 12)
        self._app.setFont(font)

        return self._app

    def run(self, window):
        if self._app is None:
            self.bootstrap()

        window.show()
        sys.exit(self._app.exec())

    def shutdown(self):
        self.container.clear()
