from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from korshmer.constants import HELP_TEXTS
from korshmer.midi_generation import composer
from korshmer.midi_generation.engine import SONG_STYLES
from korshmer.midi_generation.korshmer_schema import (
    DEFAULT_MINUTES,
    MAX_MINUTES,
    MIN_MINUTES,
)
from korshmer.ui.widgets.material.adaptive import FlowLayout
from korshmer.ui.widgets.material.button import HelpButton, MaterialButton
from korshmer.ui.widgets.material.card import MaterialCard


class GeneratorPageMixin:
    def _create_midi_generator_panel(self):
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.addWidget(QLabel("MIDI Generator"))
        header.addWidget(
            HelpButton(HELP_TEXTS["midi_generator"]), alignment=Qt.AlignLeft
        )
        header.addStretch()
        layout.addLayout(header)
        layout.addWidget(self._ui_card_generator_settings())
        layout.addWidget(self._ui_card_generator_roles())

        hint = QLabel(
            "Key and tempo come from the THEORY tab. Auto detects them "
            "from what is already in the editor."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; font-weight: normal;")
        layout.addWidget(hint)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return panel

    def _ui_card_generator_settings(self):
        card = MaterialCard()
        grid = QGridLayout(card)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.combo_gen_mode = QComboBox()
        self.combo_gen_mode.addItems(list(composer.MODES))
        self.spin_gen_minutes = QDoubleSpinBox()
        self.spin_gen_minutes.setRange(MIN_MINUTES, MAX_MINUTES)
        self.spin_gen_minutes.setDecimals(1)
        self.spin_gen_minutes.setSingleStep(0.5)
        self.spin_gen_minutes.setValue(DEFAULT_MINUTES)
        self.spin_gen_minutes.setSuffix(" min")
        self.combo_gen_style = QComboBox()
        self.combo_gen_style.addItems(list(SONG_STYLES))
        self.combo_gen_mood = QComboBox()
        self.combo_gen_mood.addItems(list(composer.MOODS))
        self.spin_gen_seed = QSpinBox()
        self.spin_gen_seed.setRange(0, 99999)
        self.spin_gen_seed.setSpecialValueText("Random")
        self.combo_gen_mode.currentTextChanged.connect(
            self._update_generator_mode_fields
        )
        self.chk_gen_replace = QCheckBox("Replace editor content")
        self.chk_gen_replace.setChecked(True)

        grid.setColumnStretch(1, 1)
        rows = (
            (
                self.create_label_help("Mode", HELP_TEXTS["generator_mode"]),
                self.combo_gen_mode,
            ),
            (
                self.create_label_help(
                    "Length", HELP_TEXTS["generator_minutes"]
                ),
                self.spin_gen_minutes,
            ),
            (QLabel("Style"), self.combo_gen_style),
            (QLabel("Mood"), self.combo_gen_mood),
            (
                self.create_label_help("Seed", HELP_TEXTS["seed"]),
                self.spin_gen_seed,
            ),
        )
        for row, (label, field) in enumerate(rows):
            grid.addWidget(label, row, 0)
            grid.addWidget(field, row, 1)
        grid.addWidget(self.chk_gen_replace, len(rows), 0, 1, 2)
        self._update_generator_mode_fields(self.combo_gen_mode.currentText())
        return card

    def _update_generator_mode_fields(self, mode):
        layering = mode == "Layer over editor"
        self.spin_gen_minutes.setEnabled(not layering)
        self.chk_gen_replace.setEnabled(mode == "New song")

    def _ui_card_generator_roles(self):
        card = MaterialCard()
        grid = QGridLayout(card)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        roles_row = FlowLayout(spacing=8)
        roles_row.addWidget(
            self.create_label_help("Roles", HELP_TEXTS["generator_roles"])
        )
        self.gen_role_boxes = {}
        for key in composer.ROLE_KEYS:
            box = QCheckBox(composer.ROLE_LABELS[key])
            box.setChecked(key in composer.DEFAULT_ROLES)
            self.gen_role_boxes[key] = box
            roles_row.addWidget(box)
        grid.addLayout(roles_row, 0, 0, 1, 2)

        btn_generate = MaterialButton("Generate → Editor", role="primary")
        btn_generate.setFixedHeight(38)
        btn_generate.clicked.connect(self.run_midi_generator)
        btn_play = MaterialButton("▶︎ Play Editor", role="success")
        btn_play.clicked.connect(
            lambda: (
                self.arranger.sync_and_play()
                if hasattr(self, "arranger")
                else None
            )
        )
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.addWidget(btn_generate, 1, 0)
        grid.addWidget(btn_play, 1, 1)
        return card
