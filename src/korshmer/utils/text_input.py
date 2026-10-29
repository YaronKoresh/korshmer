from __future__ import annotations

MAX_TEXT_INPUT_LENGTH: int = 4096


def is_within_max_length(text, max_length: int = MAX_TEXT_INPUT_LENGTH) -> bool:
    return text is None or len(text) <= max_length


def limit_line_edit(line_edit, max_length: int = MAX_TEXT_INPUT_LENGTH):
    line_edit.setMaxLength(max_length)
    return line_edit


def get_limited_text(
    parent,
    title: str,
    label: str,
    text: str = "",
    max_length: int = MAX_TEXT_INPUT_LENGTH,
) -> tuple[str, bool]:
    from PySide6.QtWidgets import QDialog, QInputDialog, QLineEdit, QMessageBox

    dialog = QInputDialog(parent)
    dialog.setInputMode(QInputDialog.TextInput)
    dialog.setWindowTitle(title)
    dialog.setLabelText(label)
    dialog.setTextValue(text)
    line_edit = dialog.findChild(QLineEdit)
    if line_edit is not None:
        limit_line_edit(line_edit, max_length)

    ok = dialog.exec() == QDialog.Accepted
    value = dialog.textValue()
    dialog.deleteLater()

    if ok and not is_within_max_length(value, max_length):
        QMessageBox.warning(
            parent,
            title,
            f"The text is too long ({len(value)} characters). "
            f"The maximum is {max_length} characters.",
        )
        return "", False
    return value, ok
