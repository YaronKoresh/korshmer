from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from korshmer.ui.managers.keyboard_shortcuts import find_clashes
from korshmer.ui.theming.color_utils import MAT_ERROR
from korshmer.ui.widgets.material.button import MaterialButton


class ShortcutsDialog(QDialog):
    def __init__(self, names, current, defaults, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customise Shortcuts")
        self.resize(420, 560)
        self.names = dict(names)
        self.defaults = dict(defaults)
        self.edits = {}
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        form = QFormLayout(content)
        for action, name in self.names.items():
            edit = QKeySequenceEdit(QKeySequence(current.get(action, "")))
            edit.setObjectName(action)
            edit.setMaximumSequenceLength(1)
            edit.setClearButtonEnabled(True)
            self.edits[action] = edit
            form.addRow(name, edit)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.lbl_problem = QLabel()
        self.lbl_problem.setWordWrap(True)
        self.lbl_problem.setStyleSheet(f"color: {MAT_ERROR};")
        layout.addWidget(self.lbl_problem)
        buttons = QHBoxLayout()
        self.btn_reset = MaterialButton("Reset to defaults", role="tonal")
        self.btn_reset.clicked.connect(self.reset)
        self.btn_cancel = MaterialButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save = MaterialButton("Save", role="primary")
        self.btn_save.clicked.connect(self.accept)
        buttons.addWidget(self.btn_reset)
        buttons.addStretch()
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_save)
        layout.addLayout(buttons)

    def chosen(self):
        return {
            action: edit.keySequence().toString(QKeySequence.PortableText)
            for action, edit in self.edits.items()
        }

    def reset(self):
        for action, edit in self.edits.items():
            edit.setKeySequence(QKeySequence(self.defaults.get(action, "")))
        self.lbl_problem.clear()

    def accept(self):
        clashes = find_clashes(self.chosen())
        if clashes:
            first, second, keys = clashes[0]
            self.lbl_problem.setText(
                f"{keys} is used by both '{self.names[first]}' and "
                f"'{self.names[second]}'. Give one of them another key."
            )
            return
        super().accept()
