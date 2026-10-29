import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

if not QApplication.instance():
    QApplication(sys.argv)

try:
    from korshmer.__main__ import MainWindow
except Exception as import_error:
    MainWindow = None
    MAIN_IMPORT_ERROR = str(import_error)
else:
    MAIN_IMPORT_ERROR = ""


pytestmark = pytest.mark.integration


class _Signal:
    def connect(self, _slot):
        pass


class _FakeWorker:
    created = []

    def __init__(self, paths, params, process_track_func=None):
        self.params = params
        self.progress_update = _Signal()
        self.process_finished = _Signal()
        _FakeWorker.created.append(self)

    def start(self):
        pass

    def isRunning(self):
        return False


@pytest.fixture
def window(monkeypatch):
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    win = MainWindow()
    yield win
    win.close()
    QApplication.processEvents()


def _convert(window, monkeypatch):
    _FakeWorker.created.clear()
    monkeypatch.setattr("korshmer.__main__.WorkerThread", _FakeWorker)
    window.start_conversion_logic(["song.wav"])
    return _FakeWorker.created[0].params


def test_the_conversion_defaults_skip_non_music_and_use_best(
    window, monkeypatch
):
    params = _convert(window, monkeypatch)

    assert params["music_options"].enabled is True
    assert params["separation_options"].quality == "best"
    assert [
        window.combo_separation.itemText(i)
        for i in range(window.combo_separation.count())
    ] == ["Fast", "Balanced", "Best"]


def test_the_conversion_follows_the_music_and_separation_widgets(
    window, monkeypatch
):
    window.chk_music_only.setChecked(False)
    window.combo_separation.setCurrentText("Balanced")

    params = _convert(window, monkeypatch)

    assert params["music_options"].enabled is False
    assert params["separation_options"].quality == "balanced"


def test_stem_separation_uses_the_separation_choice(
    window, monkeypatch, tmp_path
):
    song = tmp_path / "song.wav"
    song.write_bytes(b"")
    seen = []

    class FakeThread:
        def __init__(self, path, float32=False, options=None):
            seen.append(options)
            for name in (
                "progress_update",
                "session_ready",
                "separation_failed",
                "finished",
            ):
                setattr(self, name, _Signal())

        def isRunning(self):
            return False

        def start(self):
            pass

    monkeypatch.setattr(
        "korshmer.workers.stem_separation.StemSeparationThread", FakeThread
    )
    monkeypatch.setattr(
        window, "_get_selected_file", lambda **_kwargs: str(song)
    )
    window.combo_separation.setCurrentText("Fast")

    window.run_stem_separation()

    assert seen[0].quality == "fast"
