import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from korshmer.core.control_bar import ControlBar
from korshmer.event_bus import Event, publish
from korshmer.ui.layout_config import (
    LEFT_PANEL_HEADER_HEIGHT,
    LEFT_PANEL_MIN_WIDTH,
    MAIN_WINDOW_MIN_HEIGHT,
    MAIN_WINDOW_MIN_WIDTH,
    RIGHT_PANEL_MARGINS,
    RIGHT_PANEL_MIN_WIDTH,
)
from korshmer.ui.theming.color_utils import MAT_BORDER, MAT_PRIMARY, MAT_TEXT
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline
from korshmer.ui.widgets.audio.file_drop_list import (
    FileDropList,
    OutputFileList,
)
from korshmer.ui.widgets.audio.spectrogram import SpectrogramWidget
from korshmer.ui.widgets.audio.visualizer import AudioVisualizer
from korshmer.ui.widgets.marquee import MarqueeLabel
from korshmer.ui.widgets.material.button import HelpButton, MaterialButton
from korshmer.ui.widgets.selectable_item import SelectableItem
from korshmer.ui.widgets.titlebar import KorshmerTitleBar
from korshmer.utils.filesystem import is_midi_path

from .pages.generator import GeneratorPageMixin
from .pages.main import MainPagesMixin
from .pages.tools import ToolsPageMixin


class UiMixin(MainPagesMixin, GeneratorPageMixin, ToolsPageMixin):
    def setup_ui(self):
        self.setLayoutDirection(Qt.LeftToRight)

        self._last_selected_list = "input"
        self._init_main_window()
        self._setup_left_panel()
        self._setup_right_panel()
        self._setup_control_bar()
        self._final_connections()
        self.title_bar.installEventFilter(self)
        self.centralWidget().installEventFilter(self)
        self.setAcceptDrops(True)

    def _init_main_window(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.setMinimumSize(MAIN_WINDOW_MIN_WIDTH, MAIN_WINDOW_MIN_HEIGHT)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.title_bar = KorshmerTitleBar(self)
        self.main_layout.addWidget(self.title_bar)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.main_layout.addWidget(self.splitter, 1)

    def _setup_left_panel(self):
        left_panel = QWidget()
        left_panel.setMinimumWidth(LEFT_PANEL_MIN_WIDTH)
        l = QVBoxLayout(left_panel)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(LEFT_PANEL_HEADER_HEIGHT)
        header.setStyleSheet(f"border-bottom: 1px solid {MAT_BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(15, 0, 15, 0)

        logo = MarqueeLabel("♫ KORSHMER")
        logo.setStyleSheet(f"font-size: 18px; color: {MAT_PRIMARY};")
        hl.addWidget(logo)
        hl.addStretch()

        list_container = QWidget()
        lcl = QVBoxLayout(list_container)
        lcl.setContentsMargins(12, 8, 12, 8)
        lcl.setSpacing(6)

        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        self.btn_add_file = MaterialButton("✚ File", role="primary")
        self.btn_add_folder = MaterialButton("📂\ufe0e Folder", role="primary")
        self.btn_clear_list = MaterialButton("✕ Clear", role="danger")

        btn_box.addWidget(self.btn_add_file)
        btn_box.addWidget(self.btn_add_folder)
        btn_box.addWidget(self.btn_clear_list)

        lcl.addLayout(btn_box)

        self.btn_refresh = MaterialButton("⟳ Refresh", role="tonal")
        self.btn_mix_preview = MaterialButton("▶\ufe0e Mix", role="success")

        lcl.addWidget(self.btn_refresh)
        lcl.addWidget(self.btn_mix_preview)

        self.lbl_input = QLabel("Added Input Files")
        self.lbl_input.setStyleSheet("text-transform: uppercase;")

        self.input_list = FileDropList()
        self.input_list.setFocusPolicy(Qt.NoFocus)

        self.lbl_output = QLabel("Generated Output Files")
        self.lbl_output.setStyleSheet("text-transform: uppercase;")
        self.output_list = OutputFileList()
        self.output_list.setFocusPolicy(Qt.NoFocus)
        self.output_list.setAcceptDrops(False)
        self.output_list.setToolTip(
            "Double-click to preview. Drag a file onto the editor to add it."
        )

        lcl.addWidget(self.lbl_input)
        lcl.addWidget(self.input_list, 1)
        lcl.addWidget(self.lbl_output)
        lcl.addWidget(self.output_list, 1)

        l.addWidget(header)
        l.addWidget(list_container)

        viz_container = QWidget()
        vl = QVBoxLayout(viz_container)
        vl.setContentsMargins(12, 0, 12, 8)
        vl.setSpacing(4)

        self.spectrogram = SpectrogramWidget()
        self.visualizer = AudioVisualizer()

        vl.addWidget(QLabel("Spectrogram"))
        vl.addWidget(self.spectrogram)
        vl.addWidget(QLabel("Waveform"))
        vl.addWidget(self.visualizer)

        l.addWidget(viz_container)
        self.splitter.addWidget(left_panel)

    def _setup_right_panel(self):
        right_panel = QWidget()
        right_panel.setMinimumWidth(RIGHT_PANEL_MIN_WIDTH)
        l = QVBoxLayout(right_panel)
        l.setContentsMargins(*RIGHT_PANEL_MARGINS)
        l.setSpacing(15)

        top_bar = QHBoxLayout()
        self.file_label = MarqueeLabel(
            "No file selected", speed=1, color=MAT_TEXT
        )
        top_bar.addWidget(self.file_label, 1)

        self.tabs = QTabWidget()

        self.arranger = ArrangerTimeline()
        self.arranger.recording_saved.connect(self.add_file)
        self.tabs.addTab(self.arranger, "EDITOR")
        self.tabs.addTab(self._create_page_music(), "THEORY")
        self.tabs.addTab(self._create_page_tools(), "EFFECTS")
        self.tabs.addTab(self._create_page_stems(), "STEMS")
        self.tabs.addTab(self._create_page_export(), "EXPORT")

        l.addLayout(top_bar)
        l.addWidget(self.tabs)
        self.splitter.addWidget(right_panel)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(1, 2)

    def _setup_control_bar(self):
        self.control_bar = ControlBar()
        self.main_layout.addWidget(self.control_bar)

        self.arranger.looper = self.control_bar.player.looper

    def _clear_output_list_selection(self, item):
        self.output_list.clearSelection()
        self.output_list.setCurrentItem(None)
        self._last_selected_list = "input"

    def _clear_input_list_selection(self, item):
        self.input_list.clearSelection()
        self.input_list.setCurrentItem(None)
        self._last_selected_list = "output"

    def _final_connections(self):
        self.input_list.itemClicked.connect(self.on_input_file_select)
        self.input_list.itemPressed.connect(self._clear_output_list_selection)
        self.output_list.itemPressed.connect(self._clear_input_list_selection)

        self.input_list.itemDoubleClicked.connect(
            self.on_input_file_double_click
        )
        self.output_list.itemDoubleClicked.connect(
            self.on_output_file_double_click
        )
        self.control_bar.btn_convert.clicked.connect(self.start_processing)

        self.input_list.files_dropped.connect(self.on_files_dropped_list)

        self.btn_add_file.clicked.connect(self.browse_files)
        self.btn_add_folder.clicked.connect(self.run_batch_folder_action)
        self.btn_refresh.clicked.connect(self.refresh_lists)
        self.btn_clear_list.clicked.connect(self.clear_files)

        self.control_bar.player.positionChanged.connect(
            self.visualizer.set_position
        )
        self.control_bar.player.positionChanged.connect(
            self.arranger.piano_roll.update_playhead
        )
        self.control_bar.player.playbackStateChanged.connect(
            lambda s: self.visualizer.set_playing(s == 1)
        )
        self.arranger.piano_roll.midi_dropped.connect(
            self.add_midi_lanes_to_editor
        )
        self.control_bar.btn_panic.clicked.connect(self.panic_action)
        self.control_bar.btn_ab.toggled.connect(self.toggle_ab_compare)

        self.control_bar.player.volumeChanged.connect(
            self.visualizer.set_volume_level
        )
        self.control_bar.player.playbackStateChanged.connect(
            self.on_playback_state_changed
        )

        self.visualizer.set_volume_level(self.control_bar.player.volume())

        self.btn_mix_preview.clicked.connect(self.play_full_mix_action)
        self.arranger.editor_play_request.connect(
            self.handle_editor_play_request
        )

        self.input_list.itemClicked.connect(
            lambda item: self.file_label.setText(item.text())
        )
        self.input_list.itemDoubleClicked.connect(
            lambda item: self.file_label.setText(item.text())
        )

        self.combo_root.currentTextChanged.connect(
            lambda: self.update_piano_scale_view()
        )
        self.combo_scale.currentTextChanged.connect(
            lambda: self.update_piano_scale_view()
        )

        self.spin_bpm.valueChanged.connect(self.update_editor_tempo)
        self.arranger.tempo_changed.connect(self.spin_bpm.setValue)
        if hasattr(self, "chk_link"):
            self.chk_link.toggled.connect(self.set_link_enabled)
        if hasattr(self, "chk_osc"):
            self.chk_osc.toggled.connect(self.set_osc_enabled)
        if hasattr(self, "chk_obs"):
            self.chk_obs.toggled.connect(self.set_obs_enabled)

        roll = self.arranger.piano_roll
        if hasattr(roll, "audio_dropped"):
            roll.audio_dropped.connect(
                lambda p: publish(
                    Event.STATUS_UPDATE,
                    text=f"Added clip to editor: {os.path.basename(p)}",
                )
            )
        if hasattr(self.arranger, "tracks_changed"):
            self.arranger.tracks_changed.connect(
                lambda: publish(Event.EDITOR_TRACKS_CHANGED)
            )
        if hasattr(self.arranger, "export_mix_requested"):
            self.arranger.export_mix_requested.connect(
                self.export_editor_mix_action
            )
        if hasattr(self.arranger, "export_clip_requested"):
            self.arranger.export_clip_requested.connect(
                self.export_music_clip_action
            )
        if hasattr(self.arranger, "midi_exported"):
            self.arranger.midi_exported.connect(
                lambda p: publish(
                    Event.OUTPUT_VIEW_REFRESH, output_dir=os.path.dirname(p)
                )
            )

        self.update_piano_scale_view()

    def create_checkbox_with_help(self, checkbox, tooltip):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(4)
        l.addWidget(checkbox)
        l.addWidget(HelpButton(tooltip))
        l.addStretch()
        return w

    def create_label_help(self, text, tooltip):

        c = QWidget()
        c.setAttribute(Qt.WA_OpaquePaintEvent, False)
        l = QHBoxLayout(c)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(4)
        lbl = QLabel(text)
        help_btn = HelpButton(tooltip)
        l.addWidget(lbl)
        l.addWidget(help_btn)
        l.addStretch()

        c.setMinimumHeight(22)
        return c

    def _create_list_item_widget(
        self, file_name, file_path, list_widget, item_ref, is_output=False
    ):
        widget = SelectableItem(list_widget, item_ref)
        widget.setFixedHeight(40)
        l = QHBoxLayout(widget)
        l.setContentsMargins(0, 0, 0, 0)
        l.setAlignment(Qt.AlignVCenter)

        if os.path.isdir(file_path):
            icon = "📂\ufe0e"
        elif is_output:
            icon = "🎹\ufe0e"
        else:
            icon = "♪"

        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet(
            "font-size: 16px; border: none; background: transparent;"
        )
        lbl_icon.setFixedWidth(24)
        lbl_icon.setAlignment(Qt.AlignCenter)

        lbl_name = MarqueeLabel(file_name, speed=1, color="#202124")
        lbl_name.setFixedHeight(28)

        l.addWidget(lbl_icon)
        l.addWidget(lbl_name, 1)

        def setup_btn(btn):
            btn.setFixedWidth(28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton { font-size: 14px; border: none; background: transparent; margin: 0px; padding: 0px; border-radius: 4px; color: #5F6368; }
                QPushButton:hover { color: #1A73E8; background: #F1F3F4; }
            """
            )

        if is_output:
            lower_path = file_path.lower()
            if is_midi_path(lower_path):
                b_edit = QPushButton("✎")
                b_edit.setToolTip("Edit in Piano Roll (replaces lanes)")
                setup_btn(b_edit)
                b_edit.clicked.connect(
                    lambda: self.handle_midi_selection(
                        file_path, load_editor=True
                    )
                )
                l.addWidget(b_edit)

                b_merge = QPushButton("⊕")
                b_merge.setToolTip("Add as new lanes in the editor")
                setup_btn(b_merge)
                b_merge.clicked.connect(
                    lambda: self.add_midi_lanes_to_editor(file_path)
                )
                l.addWidget(b_merge)
            elif lower_path.endswith(
                (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
            ):
                b_clip = QPushButton("🎤\ufe0e")
                b_clip.setToolTip("Send to editor as audio clip")
                setup_btn(b_clip)
                b_clip.clicked.connect(
                    lambda: self.send_audio_to_editor(file_path)
                )
                l.addWidget(b_clip)
            elif os.path.isdir(file_path) and os.path.exists(
                os.path.join(file_path, "stems_session.json")
            ):
                b_sess = QPushButton("⤵")
                b_sess.setToolTip(
                    "Load stems into editor (vocals clip + MIDI lanes)"
                )
                setup_btn(b_sess)
                b_sess.clicked.connect(
                    lambda: self.load_output_folder_session(file_path)
                )
                l.addWidget(b_sess)

            b_exp = QPushButton("💾\ufe0e")
            b_exp.setToolTip("Export / Save As")
            setup_btn(b_exp)
            b_exp.clicked.connect(lambda: self.export_file_action(file_path))
            l.addWidget(b_exp)

            b_loc = QPushButton("📂\ufe0e")
            b_loc.setToolTip("Open Location")
            setup_btn(b_loc)
            b_loc.clicked.connect(
                lambda: self.open_file_location_action(file_path)
            )
            l.addWidget(b_loc)

        b_info = QPushButton("ℹ")
        b_info.setToolTip("Info")
        setup_btn(b_info)
        b_info.clicked.connect(lambda: self.on_info_click(file_path))
        l.addWidget(b_info)

        b_del = QPushButton("✕")
        b_del.setToolTip("Delete")
        setup_btn(b_del)
        b_del.setStyleSheet(
            """
            QPushButton { font-size: 14px; border: none; background: transparent; margin: 0px; padding: 0px; border-radius: 4px; color: #5F6368; }
            QPushButton:hover { color: #D93025; background: #FCE8E6; }
        """
        )
        b_del.clicked.connect(
            lambda: self.on_delete_click(
                file_path, list_widget, item_ref, is_output
            )
        )
        l.addWidget(b_del)

        return widget
