from pathlib import Path
from unittest import mock

from PySide6.QtWidgets import QFileDialog

from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)


class _Harness(ProjectManagementActionsMixin):
    def __init__(self, projects_dir):
        self.project_manager = mock.Mock()
        self.project_manager.current_project_data = {"name": "Song"}
        self.project_manager.projects_dir = projects_dir


def test_save_dialog_filter_uses_uppercase_name(tmp_path):
    harness = _Harness(tmp_path)
    with mock.patch.object(
        QFileDialog, "getSaveFileName", return_value=("", "")
    ) as dialog:
        harness.save_project_as_action()

    assert dialog.call_args.args[3] == "KORSHMER Project (*.kproj)"


def test_load_dialog_filter_uses_uppercase_name(tmp_path):
    harness = _Harness(Path(tmp_path))
    with mock.patch.object(
        QFileDialog, "getOpenFileName", return_value=("", "")
    ) as dialog:
        harness.load_project_action()

    assert dialog.call_args.args[3] == "KORSHMER Project (*.kproj *.json)"
