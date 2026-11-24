from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from korshmer.event_bus import Event, publish
from korshmer.services.preset_service import DEFAULT_SEPARATION
from korshmer.ui.theming.color_utils import (
    BEAT_INDICATOR_OFF,
    MAT_ERROR,
)
from korshmer.ui.widgets.material.adaptive import FlowLayout
from korshmer.ui.widgets.material.button import (
    HelpButton,
    MaterialButton,
    MaterialToggleButton,
)
from korshmer.ui.widgets.material.card import MaterialCard
from korshmer.ui.widgets.material.choice import (
    HINT_STYLE,
    ChoiceBar,
    SectionCard,
)
from korshmer.ui.widgets.material.slider import MaterialSlider
from korshmer.utils import audio_host
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

TEMPO_HINT = "Speed of the song. It is filled in when a song is converted."
KEY_HINT = "Leave on Auto to use the key detected from the song."
TIMING_HINT = "How tightly converted notes are pulled onto the beat."
DYNAMICS_HINT = "How loud each converted note is played."
SYNC_HINT = "Connect to other music software. Leave off if you are unsure."
PRESET_HINT = "A preset remembers every setting on the Theory and Export tabs."
MIDI_FILES_HINT = "What Start Processing writes for the converted notes."
STEMS_HINT = "How the song is split into vocals, drums, bass and the rest."
AUDIO_HINT = (
    "Where the sound is played. A smaller buffer reacts faster, a larger "
    "one avoids crackles. Changes apply the next time playback starts."
)
SYSTEM_DEFAULT_DEVICE = "System default"
AUTOMATIC_BUFFER = "Automatic"
NOT_CONNECTED = " (not connected)"
LATENCY_UNKNOWN = "Output latency: measured while a song plays"
VELOCITY_HINTS = {
    "Original": "Keep the loudness heard in the song.",
    "Fixed": "Every note gets the same loudness.",
    "Compressed": "Loud and quiet notes are brought closer together.",
}
VELOCITY_AMOUNT_NAMES = {
    "Original": "Amount (not used)",
    "Fixed": "Loudness of every note",
    "Compressed": "Compression strength",
}
EXPORT_MODE_HINTS = {
    "Separate": "One MIDI file for each instrument, ready to edit one by one.",
    "Merged": "One MIDI file holding all the instruments together.",
}
SEPARATION_HINTS = {
    "Fast": "Quickest. Good for a first listen.",
    "Balanced": "Cleaner vocals and instruments. About three times slower.",
    "Best": "Cleanest result. The slowest choice.",
}
SCALE_BOX_LETTERS = 12
LABEL_SLACK = 8
TEMPO_BOX_STYLE = """
    QDoubleSpinBox#tempoBox {
        font-size: 24px;
        font-weight: bold;
        color: #202124;
        background-color: white;
        border: 1px solid #DADCE0;
        border-radius: 6px;
        padding: 2px 8px;
    }
    QDoubleSpinBox#tempoBox:focus { border: 1px solid #1A73E8; }
"""


class DeviceComboBox(QComboBox):
    def __init__(self, refill, parent=None):
        super().__init__(parent)
        self.refill = refill

    def showPopup(self):
        self.refill()
        super().showPopup()


class MainPagesMixin:
    def _scroll_page(self, page, sections):
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(15, 15, 15, 15)
        grid.setSpacing(14)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        row = 0
        for line in sections:
            if len(line) == 1:
                grid.addWidget(line[0], row, 0, 1, 2)
            else:
                grid.addWidget(line[0], row, 0)
                grid.addWidget(line[1], row, 1)
            row += 1
        grid.setRowStretch(row, 1)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return page

    def _labelled(self, text, tooltip, widget):
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.create_label_help(text, tooltip))
        column.addWidget(widget)
        return box

    def _create_page_music(self):
        return self._scroll_page(
            QWidget(),
            [
                (self._ui_tempo_section(), self._ui_key_section()),
                (self._ui_timing_section(), self._ui_dynamics_section()),
                (self._ui_sync_section(),),
            ],
        )

    def _ui_tempo_section(self):
        card = SectionCard("Tempo", TEMPO_HINT)
        card.setStyleSheet(TEMPO_BOX_STYLE)
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setObjectName("tempoBox")
        self.spin_bpm.setRange(float(MIN_BPM), float(MAX_BPM))
        self.spin_bpm.setDecimals(BPM_DECIMALS)
        self.spin_bpm.setSingleStep(1.0)
        self.spin_bpm.setValue(120.0)
        self.spin_bpm.setSuffix(" BPM")
        self.spin_bpm.setAlignment(Qt.AlignCenter)
        self.spin_bpm.setMinimumHeight(48)
        self.spin_bpm.setToolTip(HELP_TEXTS["bpm"])
        self.btn_tap_tempo = MaterialButton("Tap tempo", role="tonal")
        self.btn_tap_tempo.setToolTip("Tap repeatedly to set the tempo (T)")
        self.btn_tap_tempo.setMinimumHeight(48)
        self.btn_tap_tempo.clicked.connect(self.on_tap_tempo)
        self.beat_indicator = QLabel()
        self.beat_indicator.setFixedSize(16, 16)
        self.beat_indicator.setStyleSheet(BEAT_INDICATOR_OFF)
        self.beat_indicator.setToolTip("Flashes on every beat")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.spin_bpm, 2)
        row.addWidget(self.btn_tap_tempo, 1)
        row.addWidget(self.beat_indicator)
        card.body.addLayout(row)
        return card

    def _ui_key_section(self):
        card = SectionCard("Key", KEY_HINT)
        self.combo_root = QComboBox()
        self.combo_root.addItems(NOTE_NAMES)
        self.combo_root.insertItem(0, "Auto")
        self.combo_root.setCurrentIndex(0)
        self.combo_root.hide()
        self.root_bar = ChoiceBar(self.combo_root)
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(list(SCALES.keys()))
        self.combo_scale.insertItem(0, "Auto (Detect)")
        self.combo_scale.setCurrentIndex(0)
        self.combo_scale.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo_scale.setMinimumContentsLength(SCALE_BOX_LETTERS)
        self.combo_octave = QComboBox()
        self.combo_octave.addItems(["Original", "2", "3", "4", "5", "6", "7"])
        self.combo_octave.hide()
        self.octave_bar = ChoiceBar(self.combo_octave)
        card.body.addWidget(
            self._labelled("Root note", HELP_TEXTS["root"], self.root_bar)
        )
        card.body.addWidget(
            self._labelled("Scale", HELP_TEXTS["scale"], self.combo_scale)
        )
        card.body.addWidget(
            self._labelled("Octave", HELP_TEXTS["octave"], self.octave_bar)
        )
        return card

    def _ui_timing_section(self):
        card = SectionCard("Timing", TIMING_HINT)
        self.combo_grid = QComboBox()
        self.combo_grid.addItems(list(GRID_RESOLUTIONS.keys()))
        self.combo_grid.setCurrentText("1/16")
        self.combo_grid.hide()
        self.grid_bar = ChoiceBar(self.combo_grid)
        self.slider_quantize = MaterialSlider(
            "Quantize", HELP_TEXTS["quantize"], default=100
        )
        self.slider_swing = MaterialSlider(
            "Swing", HELP_TEXTS["swing"], max_val=75, default=0
        )
        card.body.addWidget(
            self._labelled("Grid", HELP_TEXTS["grid"], self.grid_bar)
        )
        card.body.addWidget(self.slider_quantize)
        card.body.addWidget(self.slider_swing)
        return card

    def _ui_dynamics_section(self):
        card = SectionCard("Dynamics", DYNAMICS_HINT)
        self.combo_velocity = QComboBox()
        self.combo_velocity.addItems(list(VELOCITY_HINTS))
        self.combo_velocity.setCurrentText("Original")
        self.combo_velocity.hide()
        self.velocity_bar = ChoiceBar(self.combo_velocity, VELOCITY_HINTS)
        self.slider_vel_val = MaterialSlider(
            VELOCITY_AMOUNT_NAMES["Original"],
            "Velocity Value/Compression",
            default=80,
        )
        metrics = self.slider_vel_val.lbl_name.fontMetrics()
        self.slider_vel_val.lbl_name.setMinimumWidth(
            max(
                metrics.horizontalAdvance(name)
                for name in VELOCITY_AMOUNT_NAMES.values()
            )
            + LABEL_SLACK
        )
        self.combo_velocity.currentTextChanged.connect(
            self._sync_velocity_amount
        )
        card.body.addWidget(
            self._labelled(
                "Note velocity", HELP_TEXTS["velocity"], self.velocity_bar
            )
        )
        card.body.addWidget(self.slider_vel_val)
        self._sync_velocity_amount(self.combo_velocity.currentText())
        return card

    def _sync_velocity_amount(self, mode):
        self.slider_vel_val.lbl_name.setText(
            VELOCITY_AMOUNT_NAMES.get(mode, VELOCITY_AMOUNT_NAMES["Original"])
        )
        self.slider_vel_val.setEnabled(mode != "Original")

    def _ui_sync_section(self):
        card = SectionCard("Sync and remote control", SYNC_HINT)
        self.chk_link = QCheckBox("Ableton Link")
        self.chk_link.setToolTip(HELP_TEXTS["link"])
        self.chk_osc = QCheckBox("OSC remote control")
        self.chk_osc.setToolTip(HELP_TEXTS["osc"])
        self.chk_obs = QCheckBox("OBS recording")
        self.chk_obs.setToolTip(HELP_TEXTS["obs"])
        row = QHBoxLayout()
        row.setSpacing(24)
        for box, key in (
            (self.chk_link, "link"),
            (self.chk_osc, "osc"),
            (self.chk_obs, "obs"),
        ):
            row.addWidget(self.create_checkbox_with_help(box, HELP_TEXTS[key]))
        row.addStretch()
        card.body.addLayout(row)
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
        card = SectionCard("Audio", AUDIO_HINT)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        card.body.addLayout(grid)
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
        self.combo_audio_device = DeviceComboBox(self._fill_audio_devices)
        self.combo_audio_device.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo_audio_device.setMinimumContentsLength(SCALE_BOX_LETTERS)
        self._show_audio_devices([])
        self.combo_audio_device.activated.connect(self._on_audio_device_chosen)
        self.combo_audio_device.currentIndexChanged.connect(
            self._on_audio_device_chosen
        )
        self.combo_audio_buffer = QComboBox()
        self.combo_audio_buffer.addItem(AUTOMATIC_BUFFER, None)
        for frames in audio_host.BUFFER_SIZE_CHOICES:
            self.combo_audio_buffer.addItem(f"{frames} samples", frames)
        saved_buffer = audio_host.get_buffer_size_preference()
        self.combo_audio_buffer.setCurrentIndex(
            max(self.combo_audio_buffer.findData(saved_buffer), 0)
        )
        self.combo_audio_buffer.currentIndexChanged.connect(
            self._on_audio_buffer_chosen
        )
        for row, (text, box) in enumerate(
            (
                ("Audio interface", self.combo_audio_host),
                ("Output device", self.combo_audio_device),
                ("Buffer size", self.combo_audio_buffer),
            )
        ):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(box, row, 1)
        self.lbl_audio_latency = QLabel(LATENCY_UNKNOWN)
        self.lbl_audio_latency.setStyleSheet(HINT_STYLE)
        card.body.addWidget(self.lbl_audio_latency)
        return card

    def _show_audio_devices(self, names):
        combo = self.combo_audio_device
        saved = audio_host.get_output_device_preference()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(SYSTEM_DEFAULT_DEVICE, None)
        for name in names:
            combo.addItem(name, name)
        if saved is not None and saved not in names:
            combo.addItem(saved + (NOT_CONNECTED if names else ""), saved)
        combo.setCurrentIndex(max(combo.findData(saved), 0))
        combo.blockSignals(False)

    def _fill_audio_devices(self):
        self._show_audio_devices(
            [name for _index, name in audio_host.list_output_devices()]
        )

    def _apply_audio_preferences(self):
        control_bar = getattr(self, "control_bar", None)
        if control_bar is None:
            return
        notice = audio_host.apply_audio_preferences(control_bar.player)
        if notice:
            publish(Event.STATUS_UPDATE, text=notice)

    def _on_audio_device_chosen(self, _index):
        chosen = self.combo_audio_device.currentData()
        if chosen == audio_host.get_output_device_preference():
            return
        audio_host.set_output_device_preference(chosen)
        self._apply_audio_preferences()

    def _on_audio_buffer_chosen(self, _index):
        audio_host.set_buffer_size_preference(
            self.combo_audio_buffer.currentData()
        )
        self._apply_audio_preferences()

    def refresh_audio_latency(self):
        player = self.control_bar.player
        latency = getattr(player, "output_latency_ms", 0.0)
        buffer_size = getattr(player, "active_buffer_size", None)
        if not latency or latency <= 0:
            return
        text = f"Output latency: {latency:.1f} ms"
        if buffer_size:
            text += f" (buffer {buffer_size} samples)"
        self.lbl_audio_latency.setText(text)

    def _create_page_export(self):
        sections = [
            (self._ui_preset_section(),),
            (self._ui_midi_files_section(), self._ui_stem_quality_section()),
            (self._create_audio_interface_card(),),
        ]
        return self._scroll_page(MaterialCard(), sections)

    def _ui_preset_section(self):
        card = SectionCard("Preset", PRESET_HINT)
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(list(self.presets.keys()))
        self.combo_presets.currentTextChanged.connect(self.apply_preset)
        btn_save = MaterialButton("Save preset", role="tonal")
        btn_save.clicked.connect(self.save_preset_dialog)
        btn_reset = MaterialButton("Reset to defaults")
        btn_reset.setToolTip("Put every setting back to its starting value")
        btn_reset.clicked.connect(self.apply_preset_default)
        btn_json = MaterialButton("Export settings as JSON")
        btn_json.clicked.connect(self.run_json_export)
        for button in (btn_save, btn_reset, btn_json):
            button.setFixedHeight(30)
            button.setMinimumWidth(button.sizeHint().width() + 24)
        card.body.addWidget(self.combo_presets)
        row = FlowLayout(spacing=8)
        row.addWidget(btn_save)
        row.addWidget(btn_reset)
        row.addWidget(btn_json)
        card.body.addLayout(row)
        return card

    def _option_tile(self, radio, text):
        tile = QWidget()
        column = QVBoxLayout(tile)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(radio)
        note = QLabel(text)
        note.setWordWrap(True)
        note.setStyleSheet(HINT_STYLE)
        note.setContentsMargins(22, 0, 0, 0)
        column.addWidget(note)
        return tile

    def _ui_midi_files_section(self):
        card = SectionCard("MIDI files", MIDI_FILES_HINT)
        self.rad_merged = QRadioButton("Merged")
        self.rad_separate = QRadioButton("Separate")
        self.rad_separate.setChecked(True)
        card.body.addWidget(
            self._option_tile(self.rad_separate, EXPORT_MODE_HINTS["Separate"])
        )
        card.body.addWidget(
            self._option_tile(self.rad_merged, EXPORT_MODE_HINTS["Merged"])
        )
        self.chk_cc = QCheckBox("Generate CC")
        self.chk_gm = QCheckBox("GM Mapping")
        card.body.addWidget(
            self.create_checkbox_with_help(self.chk_cc, HELP_TEXTS["cc"])
        )
        card.body.addWidget(
            self.create_checkbox_with_help(self.chk_gm, HELP_TEXTS["gm"])
        )
        return card

    def _ui_stem_quality_section(self):
        card = SectionCard("Stems", STEMS_HINT)
        self.combo_separation = QComboBox()
        self.combo_separation.addItems(list(SEPARATION_HINTS))
        self.combo_separation.setCurrentText(DEFAULT_SEPARATION)
        self.combo_separation.hide()
        self.separation_bar = ChoiceBar(self.combo_separation, SEPARATION_HINTS)
        card.body.addWidget(
            self._labelled(
                "Separation quality",
                HELP_TEXTS["separation"],
                self.separation_bar,
            )
        )
        self.chk_music_only = QCheckBox("Skip non-music parts")
        self.chk_music_only.setChecked(True)
        self.chk_wav32 = QCheckBox("32-bit WAV")
        card.body.addWidget(
            self.create_checkbox_with_help(
                self.chk_music_only, HELP_TEXTS["music_only"]
            )
        )
        card.body.addWidget(
            self.create_checkbox_with_help(self.chk_wav32, HELP_TEXTS["wav32"])
        )
        return card
