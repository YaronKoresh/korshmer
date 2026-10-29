from __future__ import annotations

import inspect

from korshmer.project_management.asset_manager import AssetManager
from korshmer.ui.theming.theme_manager import ThemeManager


def test_asset_manager_save_library_uses_replace():
    source = inspect.getsource(AssetManager._save_library)
    assert "os.replace" in source, (
        "_save_library should use os.replace for atomic write"
    )
    assert ".tmp" in source


def test_asset_manager_save_favorites_uses_replace():
    source = inspect.getsource(AssetManager._save_favorites)
    assert "os.replace" in source, (
        "_save_favorites should use os.replace for atomic write"
    )
    assert ".tmp" in source


def test_theme_manager_save_custom_theme_uses_replace():
    source = inspect.getsource(ThemeManager.save_custom_theme)
    assert "os.replace" in source, (
        "save_custom_theme should use os.replace for atomic write"
    )
    assert ".tmp" in source
