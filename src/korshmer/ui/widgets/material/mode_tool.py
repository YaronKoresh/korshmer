from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from korshmer.ui.widgets.material.button import HelpButton, MaterialButton
from korshmer.ui.widgets.material.card import MaterialCard


@dataclass
class ToolMode:
    label: str
    action: object
    widgets: tuple = field(default_factory=tuple)
    help: str = ""


class ModeToolCard(MaterialCard):
    def __init__(
        self,
        title,
        modes,
        help_text="",
        button_text="Apply",
        role="primary",
        parent=None,
    ):
        super().__init__(parent)
        if not modes:
            raise ValueError("A ModeToolCard needs at least one mode")
        labels = [m.label for m in modes]
        if len(set(labels)) != len(labels):
            raise ValueError("Mode labels must be unique")
        self.modes = list(modes)
        self._base_help = help_text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.title_label = QLabel(f"<b>{title}</b>")
        header.addWidget(self.title_label)
        self.help_button = HelpButton(help_text)
        header.addWidget(self.help_button, alignment=Qt.AlignLeft)
        header.addStretch()
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(labels)
        self.combo_mode.setVisible(len(modes) > 1)
        header.addWidget(self.combo_mode, 1)
        layout.addLayout(header)

        self._params = QWidget()
        self._params_layout = QHBoxLayout(self._params)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(8)
        seen = set()
        for mode in self.modes:
            for widget in mode.widgets:
                if id(widget) not in seen:
                    seen.add(id(widget))
                    self._params_layout.addWidget(widget)
        layout.addWidget(self._params)

        self.button = MaterialButton(button_text, role)
        self.button.clicked.connect(self.apply)
        layout.addWidget(self.button)

        self.combo_mode.currentIndexChanged.connect(self._sync)
        self._sync()

    def mode_labels(self):
        return [m.label for m in self.modes]

    def current_mode(self):
        return self.modes[max(0, self.combo_mode.currentIndex())]

    def set_mode(self, label):
        self.combo_mode.setCurrentText(label)

    def visible_widgets(self):
        mode = self.current_mode()
        return [w for w in mode.widgets if not w.isHidden()]

    def _sync(self, _index=None):
        mode = self.current_mode()
        active = {id(w) for w in mode.widgets}
        for m in self.modes:
            for widget in m.widgets:
                widget.setVisible(id(widget) in active)
        self._params.setVisible(bool(mode.widgets))
        tip = mode.help or self._base_help
        self.help_button.setToolTip(tip)
        self.button.setToolTip(tip)

    def apply(self):
        return self.current_mode().action()
