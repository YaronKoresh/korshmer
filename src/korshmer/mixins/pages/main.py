from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from korshmer.constants import GRID_RESOLUTIONS, HELP_TEXTS, NOTE_NAMES, SCALES
from korshmer.ui.theming.color_utils import (
    BEAT_INDICATOR_OFF,
    BPM_SPINBOX_NORMAL,
    MAT_ERROR,
)
from korshmer.ui.widgets.material.button import (
    HelpButton,
    MaterialButton,
    MaterialToggleButton,
)
from korshmer.ui.widgets.material.card import MaterialCard
from korshmer.ui.widgets.material.slider import MaterialSlider
from korshmer.utils.audio_host import (
    HOST_ASIO,
    HOST_COREAUDIO,
    HOST_DEFAULT,
    HOST_JACK,
    available_hosts,
    get_host_preference,
    set_host_preference,
)
from korshmer.utils.tempo import BPM_DECIMALS, MAX_BPM, MIN_BPM


class MainPagesMixin:
    def _create_page_music(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        layout.addWidget(QLabel("Theory & Rhythm"))
        layout.addWidget(self._ui_music_theory_grid())
        layout.addSpacing(5)
        layout.addWidget(QLabel("Groove & Dynamics"))
        layout.addWidget(self._ui_music_groove_grid())
        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return page

    def _ui_music_theory_grid(self):
        card = MaterialCard()
        l = QFormLayout(card)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(8)
        self.combo_root = QComboBox()
        self.combo_root.addItems(NOTE_NAMES)
        self.combo_root.insertItem(0, "Auto")
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(list(SCALES.keys()))
        self.combo_scale.insertItem(0, "Auto (Detect)")
        self.combo_octave = QComboBox()
        self.combo_octave.addItems(["Original", "2", "3", "4", "5", "6", "7"])

        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(float(MIN_BPM), float(MAX_BPM))
        self.spin_bpm.setDecimals(BPM_DECIMALS)
        self.spin_bpm.setSingleStep(1.0)
        self.spin_bpm.setValue(120.0)
        self.spin_bpm.setSuffix(" BPM")
        self.spin_bpm.setStyleSheet(BPM_SPINBOX_NORMAL)
        self.spin_bpm.valueChanged.connect(self.update_editor_tempo)

        l.addRow(
            self.create_label_help("Tempo", HELP_TEXTS["bpm"]), self.spin_bpm
        )
        self.btn_tap_tempo = MaterialButton("Tap")
        self.btn_tap_tempo.setToolTip("Tap repeatedly to set the tempo (T)")
        self.btn_tap_tempo.clicked.connect(self.on_tap_tempo)
        self.beat_indicator = QLabel()
        self.beat_indicator.setFixedSize(16, 16)
        self.beat_indicator.setStyleSheet(BEAT_INDICATOR_OFF)
        tap_row = QHBoxLayout()
        tap_row.addWidget(self.btn_tap_tempo)
        tap_row.addWidget(self.beat_indicator)
        l.addRow(self.create_label_help("Tap", HELP_TEXTS["bpm"]), tap_row)
        self.chk_link = QCheckBox("Ableton Link")
        self.chk_link.setToolTip(HELP_TEXTS["link"])
        l.addRow(
            self.create_label_help("Sync", HELP_TEXTS["link"]), self.chk_link
        )
        self.chk_osc = QCheckBox("OSC remote control")
        self.chk_osc.setToolTip(HELP_TEXTS["osc"])
        l.addRow(
            self.create_label_help("Remote", HELP_TEXTS["osc"]), self.chk_osc
        )
        self.chk_obs = QCheckBox("OBS recording")
        self.chk_obs.setToolTip(HELP_TEXTS["obs"])
        l.addRow(self.create_label_help("OBS", HELP_TEXTS["obs"]), self.chk_obs)
        l.addRow(
            self.create_label_help("Root", HELP_TEXTS["root"]), self.combo_root
        )
        l.addRow(
            self.create_label_help("Scale", HELP_TEXTS["scale"]),
            self.combo_scale,
        )
        l.addRow(
            self.create_label_help("Octave", HELP_TEXTS["octave"]),
            self.combo_octave,
        )

        return card

    def _ui_music_groove_grid(self):
        card = MaterialCard()
        l = QVBoxLayout(card)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(8)
        h_top = QHBoxLayout()
        self.combo_grid = QComboBox()
        self.combo_grid.addItems(list(GRID_RESOLUTIONS.keys()))
        self.combo_grid.setCurrentText("1/16")
        self.combo_velocity = QComboBox()
        self.combo_velocity.addItems(["Original", "Fixed", "Compressed"])
        self.combo_velocity.setCurrentText("Fixed")
        h_top.addWidget(self.create_label_help("Grid:", HELP_TEXTS["grid"]))
        h_top.addWidget(self.combo_grid, 1)
        h_top.addWidget(self.create_label_help("Vel:", HELP_TEXTS["velocity"]))
        h_top.addWidget(self.combo_velocity, 1)
        l.addLayout(h_top)
        self.slider_quantize = MaterialSlider(
            "Quantize", HELP_TEXTS["quantize"], default=100
        )
        self.slider_swing = MaterialSlider(
            "Swing", HELP_TEXTS["swing"], max_val=75, default=0
        )
        self.slider_vel_val = MaterialSlider(
            "Val/Comp", "Velocity Value/Compression", default=80
        )
        l.addWidget(self.slider_quantize)
        l.addWidget(self.slider_swing)
        l.addWidget(self.slider_vel_val)
        return card

    def _create_page_stems(self):
        page = MaterialCard()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        l = QVBoxLayout(content)
        l.setContentsMargins(15, 15, 15, 15)

        h = QHBoxLayout()
        h.addWidget(QLabel("Instrument Separation (Stems)"))
        h.addWidget(HelpButton(HELP_TEXTS["stems"]), alignment=Qt.AlignLeft)
        h.addStretch()
        l.addLayout(h)
        l.addSpacing(15)

        g = QGridLayout()
        g.setSpacing(15)

        self.btn_drums = MaterialToggleButton("Drums")
        self.btn_drums.setChecked(True)

        self.btn_bass = MaterialToggleButton("Bass")
        self.btn_bass.setChecked(True)

        self.btn_other = MaterialToggleButton("Other / Acc")
        self.btn_other.setChecked(True)

        self.btn_harmonic = MaterialToggleButton("Harmonic")
        self.btn_harmonic.setChecked(True)

        self.btn_vocals = MaterialToggleButton("Vocals", color=MAT_ERROR)
        self.btn_vocals.setChecked(False)

        stems_buttons = [
            self.btn_drums,
            self.btn_bass,
            self.btn_other,
            self.btn_harmonic,
            self.btn_vocals,
        ]

        for btn in stems_buttons:
            btn.setMinimumSize(0, 0)
            btn.setMaximumSize(16777215, 16777215)

            btn.setFixedHeight(50)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFont(QFont("Tahoma", 11, QFont.Bold))

        g.addWidget(self.btn_drums, 0, 0)
        g.addWidget(self.btn_bass, 0, 1)

        g.addWidget(self.btn_harmonic, 1, 0, 1, 2)

        g.addWidget(self.btn_other, 2, 0)
        g.addWidget(self.btn_vocals, 2, 1)

        l.addLayout(g)
        l.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return page

    def _create_audio_interface_card(self):
        card = MaterialCard()
        row = QHBoxLayout(card)
        self.combo_audio_host = QComboBox()
        labels = {
            HOST_DEFAULT: "Default",
            HOST_ASIO: "ASIO (low latency)",
            HOST_COREAUDIO: "CoreAudio",
            HOST_JACK: "JACK",
        }
        for host in available_hosts():
            self.combo_audio_host.addItem(labels[host], host)
        index = self.combo_audio_host.findData(get_host_preference())
        self.combo_audio_host.setCurrentIndex(max(index, 0))
        self.combo_audio_host.currentIndexChanged.connect(
            lambda _i: set_host_preference(self.combo_audio_host.currentData())
        )
        row.addWidget(QLabel("Audio interface:"))
        row.addWidget(self.combo_audio_host, 1)
        return card

    def _create_page_export(self):
        page = MaterialCard()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        l = QVBoxLayout(content)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(12)
        l.addWidget(QLabel("Export Settings"))
        card = MaterialCard()
        lc = QVBoxLayout(card)
        box = QHBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(list(self.presets.keys()))
        self.combo_presets.currentTextChanged.connect(self.apply_preset)
        btn_save = MaterialButton("Save", role="tonal")
        btn_save.clicked.connect(self.save_preset_dialog)
        btn_save.setFixedSize(70, 30)
        box.addWidget(QLabel("Preset:"))
        box.addWidget(self.combo_presets, 1)
        box.addWidget(btn_save)
        lc.addLayout(box)
        self.chk_cc = QCheckBox("Generate CC")
        self.chk_gm = QCheckBox("GM Mapping")
        self.chk_wav32 = QCheckBox("32-bit WAV")
        self.chk_music_only = QCheckBox("Skip non-music parts")
        self.chk_music_only.setChecked(False)
        self.combo_separation = QComboBox()
        self.combo_separation.addItems(["Fast", "Balanced", "Best"])
        self.combo_separation.setCurrentText("Best")
        lc.addWidget(
            self.create_checkbox_with_help(self.chk_cc, HELP_TEXTS["cc"])
        )
        lc.addWidget(
            self.create_checkbox_with_help(self.chk_gm, HELP_TEXTS["gm"])
        )
        lc.addWidget(
            self.create_checkbox_with_help(self.chk_wav32, HELP_TEXTS["wav32"])
        )
        lc.addWidget(
            self.create_checkbox_with_help(
                self.chk_music_only, HELP_TEXTS["music_only"]
            )
        )
        lc.addWidget(
            self.create_label_help("Separation", HELP_TEXTS["separation"])
        )
        lc.addWidget(self.combo_separation)
        rad_lay = QHBoxLayout()
        self.rad_merged = QRadioButton("Merged")
        self.rad_separate = QRadioButton("Separate")
        self.rad_separate.setChecked(True)
        rad_lay.addWidget(self.rad_merged)
        rad_lay.addWidget(self.rad_separate)
        rad_lay.addStretch()
        lc.addLayout(rad_lay)
        l.addWidget(card)
        if len(available_hosts()) > 1:
            l.addWidget(self._create_audio_interface_card())
        btn_json = MaterialButton("Export JSON", role="tonal")
        btn_json.clicked.connect(self.run_json_export)
        l.addWidget(btn_json)
        btn_reset = MaterialButton("Reset Defaults", role="danger")
        btn_reset.clicked.connect(self.apply_preset_default)
        l.addWidget(btn_reset)
        l.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return page
