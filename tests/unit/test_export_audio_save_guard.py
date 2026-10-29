from unittest import mock

import pytest

from korshmer.mixins.actions import core as core_module
from korshmer.mixins.actions.core import CoreActionsMixin


class _Harness(CoreActionsMixin):
    def __init__(self):
        self.spin_bpm = mock.Mock(value=mock.Mock(return_value=120))

    def get_selected_midi(self):
        return "/some/song.mid"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    saved = tmp_path / "saved_data"
    saved.mkdir()
    sibling = tmp_path / "saved_data_other"
    sibling.mkdir()
    monkeypatch.setattr(core_module, "DATA_DIR", saved)
    monkeypatch.setattr(core_module, "SF3_PATH", str(tmp_path / "sf.sf3"))
    monkeypatch.setattr(core_module.os.path, "exists", lambda p: True)
    return saved, sibling


def _run_export(chosen_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    render = mock.Mock()
    warning = mock.Mock()
    with (
        mock.patch.object(
            QFileDialog,
            "getSaveFileName",
            return_value=(str(chosen_path), "WAV Audio (*.wav)"),
        ),
        mock.patch.object(QMessageBox, "warning", warning),
        mock.patch.object(QMessageBox, "information"),
        mock.patch.object(QMessageBox, "critical"),
        mock.patch.object(core_module.subprocess, "call"),
        mock.patch("korshmer.audio_processors.render_midi_to_wav", render),
        mock.patch.object(core_module, "publish"),
    ):
        _Harness().export_audio_action()
    return render, warning


def test_rejects_sibling_folder_sharing_saved_data_prefix(data_dir):
    _, sibling = data_dir
    render, warning = _run_export(sibling / "out.wav")

    render.assert_not_called()
    warning.assert_called_once()
    assert warning.call_args.args[1] == "Safety Error"


def test_allows_file_inside_saved_data(data_dir):
    saved, _ = data_dir
    target = saved / "out.wav"
    render, warning = _run_export(target)

    warning.assert_not_called()
    render.assert_called_once()
    assert render.call_args.args[1] == str(target)
