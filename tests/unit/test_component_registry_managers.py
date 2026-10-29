from unittest.mock import MagicMock


def test_register_project_managers_passes_base_dir(tmp_path):
    from korshmer.app.components import ComponentRegistry
    from korshmer.core.di_container import DIContainer

    container = DIContainer()
    registry = ComponentRegistry(container)
    window = MagicMock()

    registry.register_project_managers(window, tmp_path)

    assert hasattr(window, "export_manager")
    assert hasattr(window, "import_manager")
    assert hasattr(window, "collaboration_manager")
    assert hasattr(window, "analytics_manager")

    from korshmer.project_management import (
        AnalyticsManager,
        CollaborationManager,
        ExportManager,
        ImportManager,
    )

    assert isinstance(window.export_manager, ExportManager)
    assert isinstance(window.import_manager, ImportManager)
    assert isinstance(window.collaboration_manager, CollaborationManager)
    assert isinstance(window.analytics_manager, AnalyticsManager)


def test_register_ui_managers_passes_kwargs(tmp_path):
    from korshmer.app.components import ComponentRegistry
    from korshmer.core.di_container import DIContainer

    container = DIContainer()
    registry = ComponentRegistry(container)
    window = MagicMock()

    registry.register_ui_managers(window, tmp_path)

    assert hasattr(window, "tutorial_manager")
    from korshmer.ui.managers import TutorialManager

    assert isinstance(window.tutorial_manager, TutorialManager)
