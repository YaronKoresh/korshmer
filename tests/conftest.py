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
