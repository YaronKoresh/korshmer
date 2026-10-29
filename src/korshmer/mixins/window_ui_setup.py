from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QStatusBar

from korshmer.ui.widgets.statusbar import EnhancedStatusBar


class MainWindowUiSetupMixin:
    def setup_window_ui_extras(self):
        self._setup_status_bar()
        self._setup_midi_hotplug()
        self._setup_keyboard_shortcuts()
        self._setup_context_menus()
        self._setup_tooltips()

    def _setup_status_bar(self):
        self.enhanced_status_bar = EnhancedStatusBar(self)

        real_status_bar = QStatusBar()
        real_status_bar.setContentsMargins(0, 0, 0, 0)
        real_status_bar.addWidget(self.enhanced_status_bar, 1)
        self.setStatusBar(real_status_bar)
        real_status_bar.setVisible(False)

    def _setup_midi_hotplug(self):
        from korshmer.midi.hotplug import MidiHotplugMonitor

        self.midi_hotplug = MidiHotplugMonitor(parent=self)
        self.midi_hotplug.device_connected.connect(
            lambda name, controller: self.enhanced_status_bar.update_action(
                f"MIDI device connected: {name} ({controller})"
            )
        )
        self.midi_hotplug.device_disconnected.connect(
            lambda name: self.enhanced_status_bar.update_action(
                f"MIDI device disconnected: {name}"
            )
        )
        self.midi_hotplug.start()

    def _setup_keyboard_shortcuts(self):
        from PySide6.QtGui import QAction, QKeySequence

        shortcuts = self.keyboard_shortcuts.get_all()

        play_pause_action = QAction(self)
        play_pause_action.setShortcut(
            QKeySequence(shortcuts.get("play_pause", "Space"))
        )
        play_pause_action.triggered.connect(lambda: self._handle_play_pause())
        self.addAction(play_pause_action)

        tap_tempo_action = QAction(self)
        tap_tempo_action.setShortcut(
            QKeySequence(shortcuts.get("tap_tempo", "T"))
        )
        tap_tempo_action.triggered.connect(lambda: self.on_tap_tempo())
        self.addAction(tap_tempo_action)

        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence(shortcuts.get("undo", "Ctrl+Z")))
        undo_action.triggered.connect(lambda: self._handle_undo())
        self.addAction(undo_action)

        redo_action = QAction(self)
        redo_action.setShortcut(QKeySequence(shortcuts.get("redo", "Ctrl+Y")))
        redo_action.triggered.connect(lambda: self._handle_redo())
        self.addAction(redo_action)

        redo_alt_action = QAction(self)
        redo_alt_action.setShortcut(
            QKeySequence(shortcuts.get("redo_alt", "Ctrl+Shift+Z"))
        )
        redo_alt_action.triggered.connect(lambda: self._handle_redo())
        self.addAction(redo_alt_action)

        save_action = QAction(self)
        save_action.setShortcut(QKeySequence(shortcuts.get("save", "Ctrl+S")))
        save_action.triggered.connect(lambda: self._handle_save())
        self.addAction(save_action)

        select_all_action = QAction(self)
        select_all_action.setShortcut(
            QKeySequence(shortcuts.get("select_all", "Ctrl+A"))
        )
        select_all_action.triggered.connect(lambda: self._handle_select_all())
        self.addAction(select_all_action)

        help_action = QAction(self)
        help_action.setShortcut(QKeySequence(shortcuts.get("help", "F1")))
        help_action.triggered.connect(lambda: self._show_help())
        self.addAction(help_action)

        fullscreen_action = QAction(self)
        fullscreen_action.setShortcut(
            QKeySequence(shortcuts.get("fullscreen", "F11"))
        )
        fullscreen_action.triggered.connect(lambda: self._toggle_fullscreen())
        self.addAction(fullscreen_action)

        performance_action = QAction(self)
        performance_action.setShortcut(
            QKeySequence(shortcuts.get("performance_mode", "F10"))
        )
        performance_action.triggered.connect(
            lambda: self.toggle_performance_mode()
        )
        self.addAction(performance_action)

        editor_bindings = [
            ("duplicate", "Ctrl+D", self.on_piano_roll_duplicate),
            ("quantize", "Q", self.on_piano_roll_quantize),
            ("humanize", "U", self.on_piano_roll_humanize),
            ("transpose_up", "Ctrl+Up", self.on_piano_roll_transpose_up),
            ("transpose_down", "Ctrl+Down", self.on_piano_roll_transpose_down),
            ("octave_up", "Ctrl+Shift+Up", self.on_piano_roll_octave_up),
            ("octave_down", "Ctrl+Shift+Down", self.on_piano_roll_octave_down),
            ("zoom_in", "Ctrl+=", lambda: self.editor_zoom(1.25)),
            ("zoom_out", "Ctrl+-", lambda: self.editor_zoom(0.8)),
            ("toggle_snap", "N", self.editor_toggle_snap),
            ("build_song", "Ctrl+B", self.run_midi_generator),
        ]
        for key, default, handler in editor_bindings:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcuts.get(key, default)))
            action.triggered.connect(
                lambda _=False, h=handler: self._run_editor_shortcut(h)
            )
            self.addAction(action)

    def _run_editor_shortcut(self, handler):
        if not self._editor_tab_active():
            return
        focus = QApplication.focusWidget()
        if focus is not None and focus.inherits("QLineEdit"):
            return
        handler()

    def _setup_context_menus(self):
        if hasattr(self, "arranger") and hasattr(self.arranger, "piano_roll"):
            self.arranger.piano_roll.setContextMenuPolicy(Qt.CustomContextMenu)
            self.arranger.piano_roll.customContextMenuRequested.connect(
                self._show_piano_roll_context_menu
            )

    def _setup_tooltips(self):
        if hasattr(self, "control_bar"):
            if hasattr(self.control_bar, "btn_play"):
                self.control_bar.btn_play.setToolTip(
                    self.tooltip_manager.get("play_pause")
                )
            if hasattr(self.control_bar, "btn_stop"):
                self.control_bar.btn_stop.setToolTip(
                    self.tooltip_manager.get("stop")
                )

        if hasattr(self, "spin_bpm"):
            self.spin_bpm.setToolTip(self.tooltip_manager.get("bpm"))
