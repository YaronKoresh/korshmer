from typing import NamedTuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QKeySequenceEdit,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QStatusBar,
    QTextEdit,
)

from korshmer.ui.managers.keyboard_shortcuts import DEFAULT_SHORTCUTS
from korshmer.ui.widgets.dialogs.shortcuts import ShortcutsDialog
from korshmer.ui.widgets.statusbar import EnhancedStatusBar


class ShortcutBinding(NamedTuple):
    group: str
    action: str
    description: str
    handler: str
    editor_only: bool = False
    while_typing: bool = False


PLAYBACK = "Playback"
EDITING = "Editing (EDITOR tab)"
TOOLS = "Tools (EDITOR tab)"
VIEW = "View"
FILE = "File"
HELP = "Help"
EDITOR_KEYS_GROUP = "Inside the piano roll"
TEXT_INPUTS = (
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QAbstractSpinBox,
    QKeySequenceEdit,
)
KEY_NAVIGATED_WIDGETS = (QAbstractItemView, QComboBox, QAbstractSlider)
NAVIGATION_KEYS = frozenset(
    (
        Qt.Key.Key_Home,
        Qt.Key.Key_End,
        Qt.Key.Key_PageUp,
        Qt.Key.Key_PageDown,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
    )
)

SHORTCUT_BINDINGS = (
    ShortcutBinding(
        PLAYBACK, "play_pause", "Play / Pause", "_handle_play_pause"
    ),
    ShortcutBinding(
        PLAYBACK,
        "stop",
        "Stop (clears a note selection first)",
        "_handle_escape",
    ),
    ShortcutBinding(PLAYBACK, "home", "Go to start", "_handle_go_to_start"),
    ShortcutBinding(PLAYBACK, "tap_tempo", "Tap tempo", "on_tap_tempo"),
    ShortcutBinding(EDITING, "undo", "Undo", "_handle_undo"),
    ShortcutBinding(EDITING, "redo", "Redo", "_handle_redo"),
    ShortcutBinding(EDITING, "redo_alt", "Redo", "_handle_redo"),
    ShortcutBinding(EDITING, "select_all", "Select all", "_handle_select_all"),
    ShortcutBinding(
        EDITING,
        "duplicate",
        "Duplicate selection",
        "on_piano_roll_duplicate",
        True,
    ),
    ShortcutBinding(
        EDITING,
        "transpose_up",
        "Transpose up 1 semitone",
        "on_piano_roll_transpose_up",
        True,
    ),
    ShortcutBinding(
        EDITING,
        "transpose_down",
        "Transpose down 1 semitone",
        "on_piano_roll_transpose_down",
        True,
    ),
    ShortcutBinding(
        EDITING,
        "octave_up",
        "Transpose up an octave",
        "on_piano_roll_octave_up",
        True,
    ),
    ShortcutBinding(
        EDITING,
        "octave_down",
        "Transpose down an octave",
        "on_piano_roll_octave_down",
        True,
    ),
    ShortcutBinding(
        TOOLS, "quantize", "Quantize selection", "on_piano_roll_quantize", True
    ),
    ShortcutBinding(
        TOOLS, "humanize", "Humanize selection", "on_piano_roll_humanize", True
    ),
    ShortcutBinding(
        TOOLS, "toggle_snap", "Toggle snap", "editor_toggle_snap", True
    ),
    ShortcutBinding(
        TOOLS,
        "build_song",
        "Generate MIDI (MIDI Generator settings)",
        "run_midi_generator",
        True,
    ),
    ShortcutBinding(VIEW, "zoom_in", "Zoom in", "_editor_zoom_in", True),
    ShortcutBinding(VIEW, "zoom_out", "Zoom out", "_editor_zoom_out", True),
    ShortcutBinding(
        VIEW, "fullscreen", "Fullscreen", "_toggle_fullscreen", False, True
    ),
    ShortcutBinding(
        VIEW,
        "performance_mode",
        "Performance mode",
        "toggle_performance_mode",
        False,
        True,
    ),
    ShortcutBinding(
        VIEW, "find_tool", "Find a tool", "focus_tool_finder", False, True
    ),
    ShortcutBinding(FILE, "new", "New song", "new_project_action", False, True),
    ShortcutBinding(
        FILE, "open", "Open song", "load_project_action", False, True
    ),
    ShortcutBinding(FILE, "save", "Save song", "_handle_save", False, True),
    ShortcutBinding(
        FILE, "export", "Export the editor mix", "export_editor_mix_action"
    ),
    ShortcutBinding(HELP, "help", "Show this help", "_show_help", False, True),
)

EDITOR_KEYS = (
    ("Delete / Backspace", "Delete selection"),
    ("Ctrl+C", "Copy"),
    ("Ctrl+V", "Paste"),
    ("Up / Down", "Transpose 1 semitone"),
    ("Shift+Up / Shift+Down", "Transpose an octave"),
    ("Left / Right", "Nudge by grid step"),
)


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
        for action in getattr(self, "shortcut_actions", {}).values():
            self.removeAction(action)
            action.deleteLater()
        self.shortcut_actions = {}
        self._bindings_by_keys = {}
        for binding in SHORTCUT_BINDINGS:
            keys = QKeySequence(self.keyboard_shortcuts.get(binding.action))
            if keys.isEmpty():
                continue
            action = QAction(self)
            action.setShortcut(keys)
            action.triggered.connect(
                lambda _=False, b=binding: self._run_shortcut(b)
            )
            self.addAction(action)
            self.shortcut_actions[binding.action] = action
            self._bindings_by_keys[keys.toString()] = binding

    def _run_shortcut(self, binding):
        handler = getattr(self, binding.handler)
        if binding.editor_only:
            self._run_editor_shortcut(handler)
        else:
            handler()

    def _run_editor_shortcut(self, handler):
        if not self._editor_tab_active():
            return
        focus = QApplication.focusWidget()
        if focus is not None and focus.inherits("QLineEdit"):
            return
        handler()

    def _an_owner_keeps_key(self, widget, event):
        owner = widget
        while owner is not None and owner is not self:
            keeps_key = getattr(owner, "keeps_key", None)
            if callable(keeps_key) and keeps_key(event):
                return True
            owner = owner.parentWidget()
        return False

    def focus_widget_keeps_key(self, widget, event):
        pressed = QKeySequence(event.keyCombination()).toString()
        binding = self._bindings_by_keys.get(pressed)
        if binding is None:
            return False
        if isinstance(widget, TEXT_INPUTS):
            return not binding.while_typing
        if isinstance(widget, QComboBox) and widget.isEditable():
            return not binding.while_typing
        if self._an_owner_keeps_key(widget, event):
            return True
        if event.key() in NAVIGATION_KEYS:
            return isinstance(widget, KEY_NAVIGATED_WIDGETS)
        plain = not event.modifiers() & ~(Qt.ShiftModifier | Qt.KeypadModifier)
        text = event.text()
        if plain and text.strip() and text.isprintable():
            return isinstance(widget, (QAbstractItemView, QComboBox))
        return False

    def shortcut_help_lines(self):
        lines = []
        group = None
        for binding in SHORTCUT_BINDINGS:
            action = self.shortcut_actions.get(binding.action)
            if action is None:
                continue
            if binding.group != group:
                group = binding.group
                lines.extend(["", f"{group}:"])
            keys = action.shortcut().toString(QKeySequence.NativeText)
            lines.append(f"  {keys} - {binding.description}")
        lines.extend(["", f"{EDITOR_KEYS_GROUP}:"])
        lines.extend(f"  {keys} - {text}" for keys, text in EDITOR_KEYS)
        return lines

    def customise_shortcuts_action(self):
        names = {b.action: b.description for b in SHORTCUT_BINDINGS}
        names["redo_alt"] = "Redo (second key)"
        dialog = ShortcutsDialog(
            names, self.keyboard_shortcuts.get_all(), DEFAULT_SHORTCUTS, self
        )
        if dialog.exec() != QDialog.Accepted:
            return
        clashes = self.keyboard_shortcuts.customise(dialog.chosen())
        if clashes:
            first, second, keys = clashes[0]
            QMessageBox.warning(
                self,
                "Customise Shortcuts",
                f"{keys} is already used by {first} and {second}. "
                "Nothing was changed.",
            )
            return
        self._setup_keyboard_shortcuts()
        self.enhanced_status_bar.update_action("Keyboard shortcuts saved")

    def _editor_zoom_in(self):
        self.editor_zoom(1.25)

    def _editor_zoom_out(self):
        self.editor_zoom(0.8)

    def _handle_escape(self):
        if self._editor_tab_active():
            scene = self.arranger.piano_roll.scene
            if scene.selectedItems():
                scene.clearSelection()
                return
        self.control_bar.stop_audio()

    def _handle_go_to_start(self):
        self.control_bar.go_to_start()

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
