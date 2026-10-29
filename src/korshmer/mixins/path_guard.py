from PySide6.QtWidgets import QMessageBox

from korshmer.utils.filesystem import is_safe_path

UNSAFE_PATH_MESSAGE = (
    "The selected path contains control characters and cannot be used."
)


def reject_unsafe_path(parent, path) -> bool:
    if is_safe_path(path):
        return False
    QMessageBox.warning(parent, "Unsafe Path", UNSAFE_PATH_MESSAGE)
    return True
