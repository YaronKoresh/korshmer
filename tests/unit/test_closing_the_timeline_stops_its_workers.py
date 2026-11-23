import sys
import threading

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from korshmer.ui.widgets.arranger import timeline as timeline_module
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline


def _wav(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(str(path), np.zeros(4410, np.float32), 44100)
    return str(path)


def _slow_waveforms(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow(path, points=512):
        started.set()
        release.wait(2.0)
        return [0.0] * points, 0.1

    monkeypatch.setattr(timeline_module, "compute_clip_waveform", slow)
    return started, release


def test_closing_the_timeline_waits_for_a_waveform_still_loading(
    tmp_path, monkeypatch
):
    app = QApplication.instance() or QApplication(sys.argv)
    started, release = _slow_waveforms(monkeypatch)
    arranger = ArrangerTimeline()
    arranger.show()
    arranger.add_audio_clip(_wav(tmp_path), start_sec=0.0, name="Clip")
    assert started.wait(2.0)
    workers = list(arranger._waveform_workers)
    assert any(worker.isRunning() for worker in workers)

    threading.Timer(0.2, release.set).start()
    arranger.close()

    assert not any(worker.isRunning() for worker in workers)
    assert all(worker.cancelled for worker in workers)
    arranger.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_stop_workers_asks_running_clip_edits_to_stop(tmp_path, monkeypatch):
    QApplication.instance() or QApplication(sys.argv)
    arranger = ArrangerTimeline()
    asked = []

    class Edit:
        def requestInterruption(self):
            asked.append("stop")

        def wait(self, timeout_ms):
            asked.append("wait")
            return True

    arranger._clip_edit_workers = [Edit()]
    arranger.stop_workers()

    assert asked == ["stop", "wait"]
    arranger._clip_edit_workers = []
