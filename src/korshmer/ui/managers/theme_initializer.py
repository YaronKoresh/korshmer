from korshmer.ui.managers.theme import ThemeManager


def init_theme_manager(data_dir):
    theme_manager = ThemeManager(data_dir)
    theme_manager.load_custom_themes()
    return theme_manager
