import sys

try:
    from PySide6.QtGui import QFont as _QFont
    from PySide6.QtWidgets import QApplication as _QApplication, QWidget

    QApplication: type[_QApplication] | None = _QApplication
    QFont: type[_QFont] | None = _QFont
    _QT_AVAILABLE = True
except ImportError:
    QApplication = None
    QFont = None
    _QT_AVAILABLE = False

from korshmer.core.di_container import DIContainer as Container


class Application:
    def __init__(self, container: Container) -> None:
        self.container = container
        self._app: _QApplication | None = None

    def bootstrap(self) -> "_QApplication":
        if QApplication is None or QFont is None:
            raise ImportError(
                "The 'PySide6' package is required to start the application."
            )
        existing = QApplication.instance()
        app = (
            existing
            if isinstance(existing, QApplication)
            else QApplication(sys.argv)
        )

        font = QFont("Tahoma", 12)
        app.setFont(font)

        self._app = app
        return app

    def run(self, window: "QWidget") -> None:
        app = self._app if self._app is not None else self.bootstrap()

        window.show()
        sys.exit(app.exec())

    def shutdown(self) -> None:
        self.container.clear()
