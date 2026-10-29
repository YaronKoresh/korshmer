from unittest.mock import MagicMock


def test_register_ui_managers_passes_data_dir_to_keyboard_shortcut_manager(
    tmp_path,
):
    from korshmer.app.components import ComponentRegistry
    from korshmer.core.di_container import DIContainer

    container = DIContainer()
    registry = ComponentRegistry(container)
    window = MagicMock()

    registry.register_ui_managers(window, tmp_path)

    from korshmer.ui.managers import KeyboardShortcutManager

    mgr = window.keyboard_shortcut_manager
    assert isinstance(mgr, KeyboardShortcutManager)
    assert mgr.data_dir is not None
    assert mgr.data_dir == tmp_path


def test_keyboard_shortcut_manager_save_load_preset_with_data_dir(tmp_path):
    from korshmer.ui.managers import KeyboardShortcutManager

    mgr = KeyboardShortcutManager(data_dir=tmp_path)
    mgr.shortcuts["custom_action"] = "Ctrl+K"
    assert mgr.save_preset("test") is True

    mgr2 = KeyboardShortcutManager(data_dir=tmp_path)
    assert mgr2.load_preset("test") is True
    assert mgr2.shortcuts.get("custom_action") == "Ctrl+K"
