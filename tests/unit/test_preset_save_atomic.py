from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from korshmer.project_management.export_manager import ExportManager
from korshmer.services.preset_service import PresetService
from korshmer.ui.managers.keyboard_shortcuts import KeyboardShortcutManager


def test_export_manager_save_presets_uses_replace():
    source = inspect.getsource(ExportManager._save_presets)
    assert "os.replace" in source, (
        "_save_presets should use os.replace for atomic write"
    )
    assert ".tmp" in source


def test_preset_service_save_all_uses_replace():
    source = inspect.getsource(PresetService.save_all)
    assert "os.replace" in source, (
        "save_all should use os.replace for atomic write"
    )
    assert ".tmp" in source


def test_keyboard_shortcuts_save_preset_uses_replace():
    source = inspect.getsource(KeyboardShortcutManager.save_preset)
    assert "os.replace" in source, (
        "save_preset should use os.replace for atomic write"
    )
    assert ".tmp" in source


def test_keyboard_shortcuts_save_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        mgr = KeyboardShortcutManager(data_dir=td)
        mgr.shortcuts = {"play": "Space", "stop": "Escape"}
        assert mgr.save_preset("test")

        path = Path(td) / "shortcuts_test.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["play"] == "Space"

        tmp_path = Path(td) / ".shortcuts_test.json.tmp"
        assert not tmp_path.exists(), "Temp file should be cleaned up"


def test_preset_service_save_all_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "presets.json"
        presets = {"my_preset": {"octave": "Original"}}
        with patch("korshmer.services.preset_service.PRESETS_FILE", target):
            PresetService.save_all(presets)
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["my_preset"]["octave"] == "Original"
