import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from korshmer.ui.widgets.audio import spectrogram

if not QApplication.instance():
    app = QApplication(sys.argv)

DELETE_WHILE_DRAWING = textwrap.dedent(
    """
    import sys
    import threading
    import time

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from korshmer.ui.widgets.audio import spectrogram

    app = QApplication([])
    release = threading.Event()

    def slow(path):
        release.wait(5)
        return QImage()

    spectrogram.create_spectrogram_image = slow
    widget = spectrogram.SpectrogramWidget()
    widget.set_audio(sys.argv[1])
    time.sleep(0.2)
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    release.set()
    spectrogram.wait_for_spectrograms()
    print("closed")
    """
)


def _clip(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050)
    return str(path)


def test_deleting_the_widget_while_its_spectrogram_is_drawn_does_not_abort(
    tmp_path,
):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    done = subprocess.run(
        [sys.executable, "-c", DELETE_WHILE_DRAWING, _clip(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "closed" in done.stdout
    assert "Destroyed while thread" not in done.stderr


def test_the_drawing_thread_belongs_to_the_application_not_the_widget(
    tmp_path, monkeypatch
):
    release = threading.Event()

    def slow(path):
        release.wait(5)
        return QImage()

    monkeypatch.setattr(spectrogram, "create_spectrogram_image", slow)
    widget = spectrogram.SpectrogramWidget()
    widget.set_audio(_clip(tmp_path))
    thread = widget.current_thread
    assert thread.parent() is QCoreApplication.instance()
    assert thread.isRunning()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    release.set()
    spectrogram.wait_for_spectrograms()
    assert not thread.isRunning()
