import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("korshmer_test")


@pytest.fixture
def sample_midi_file(test_data_dir):

    return test_data_dir / "sample.mid"


@pytest.fixture
def sample_audio_file(test_data_dir):

    return test_data_dir / "sample.wav"


@pytest.fixture(autouse=True)
def reset_env():

    original_env = os.environ.copy()

    yield

    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    yield app


_RECOVERY_FILE = {}


@pytest.fixture(autouse=True, scope="session")
def app_recovery_file_stays_out_of_the_real_data_folder(tmp_path_factory):
    from korshmer.constants import BASE_DIR
    from korshmer.project_management.project_manager import ProjectManager

    original = ProjectManager._recovery_file
    _RECOVERY_FILE["path"] = tmp_path_factory.mktemp("recovery") / "first.gz"

    def recovery_file(self):
        if Path(self.base_dir).resolve() == Path(BASE_DIR).resolve():
            return _RECOVERY_FILE["path"]
        return original(self)

    patch = pytest.MonkeyPatch()
    patch.setattr(ProjectManager, "_recovery_file", recovery_file)
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def each_test_starts_without_a_recovery_file(tmp_path_factory):
    _RECOVERY_FILE["path"] = tmp_path_factory.mktemp("recovery") / "test.gz"


@pytest.fixture(autouse=True)
def _no_song_left_open():
    from korshmer.project_management.song_folder import set_open_song_folder

    set_open_song_folder(None)
    yield
    set_open_song_folder(None)


@pytest.fixture(autouse=True)
def clips_are_not_split_by_a_real_vocal_model(monkeypatch):
    from korshmer.audio_processing import clip_stems

    monkeypatch.setattr(clip_stems, "vocals_of", lambda mix, **kwargs: None)


def _qt_application():
    widgets = sys.modules.get("PySide6.QtWidgets")
    return None if widgets is None else widgets.QApplication.instance()


def _delete_widgets(app, widgets):
    from PySide6.QtCore import QCoreApplication, QEvent

    for widget in widgets:
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            continue
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture(autouse=True)
def qt_windows_opened_by_a_test_are_deleted_after_it():
    app = _qt_application()
    before = set() if app is None else set(app.topLevelWidgets())
    yield
    app = _qt_application()
    if app is None:
        return
    _delete_widgets(app, [w for w in app.topLevelWidgets() if w not in before])


def pytest_sessionfinish(session, exitstatus):
    import gc

    app = _qt_application()
    if app is None:
        return
    _delete_widgets(app, app.topLevelWidgets())
    gc.collect()
    _delete_widgets(app, app.topLevelWidgets())
