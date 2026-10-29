import logging

from PySide6.QtWidgets import QMenu, QMessageBox


class ContextMenuManager:
    def __init__(self, parent=None):
        self.recent_actions = []
        self.max_recent = 10

    def record_action(self, action_name):
        if action_name in self.recent_actions:
            self.recent_actions.remove(action_name)
        self.recent_actions.insert(0, action_name)
        if len(self.recent_actions) > self.max_recent:
            self.recent_actions.pop()

    def get_recent_actions(self):
        return list(self.recent_actions)

    def _handle_action(self, category, action_name, parent):
        self.record_action(f"{category}:{action_name}")

        handler_base = action_name.strip().lower().replace(" ", "_")
        handler_name = f"on_{category}_{handler_base}"
        if parent is not None and hasattr(parent, handler_name):
            handler = getattr(parent, handler_name)
            if callable(handler):
                try:
                    handler()
                except Exception as e:
                    logging.exception(
                        f"Error handling context menu action '{action_name}' ({handler_name})"
                    )
                    QMessageBox.critical(
                        parent,
                        "Action Error",
                        f"'{action_name}' failed:\n{e}",
                    )

    def _handle_piano_roll_action(self, action_name, parent):
        self._handle_action("piano_roll", action_name, parent)

    def _handle_waveform_action(self, action_name, parent):
        self._handle_action("waveform", action_name, parent)

    def _handle_track_action(self, action_name, parent):
        self._handle_action("track", action_name, parent)

    def _handle_score_editor_action(self, action_name, parent):
        self._handle_action("score_editor", action_name, parent)

    def create_piano_roll_menu(self, parent):
        menu = QMenu(parent)

        select_all_action = menu.addAction("Select All")
        select_all_action.triggered.connect(
            lambda _=False, p=parent, name="Select All": (
                self._handle_piano_roll_action(name, p)
            )
        )

        menu.addSeparator()

        quantize_action = menu.addAction("Quantize")
        quantize_action.triggered.connect(
            lambda _=False, p=parent, name="Quantize": (
                self._handle_piano_roll_action(name, p)
            )
        )

        humanize_action = menu.addAction("Humanize")
        humanize_action.triggered.connect(
            lambda _=False, p=parent, name="Humanize": (
                self._handle_piano_roll_action(name, p)
            )
        )

        transpose_up_action = menu.addAction("Transpose Up")
        transpose_up_action.triggered.connect(
            lambda _=False, p=parent, name="Transpose Up": (
                self._handle_piano_roll_action(name, p)
            )
        )

        transpose_down_action = menu.addAction("Transpose Down")
        transpose_down_action.triggered.connect(
            lambda _=False, p=parent, name="Transpose Down": (
                self._handle_piano_roll_action(name, p)
            )
        )

        menu.addSeparator()

        duplicate_action = menu.addAction("Duplicate")
        duplicate_action.triggered.connect(
            lambda _=False, p=parent, name="Duplicate": (
                self._handle_piano_roll_action(name, p)
            )
        )

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(
            lambda _=False, p=parent, name="Delete": (
                self._handle_piano_roll_action(name, p)
            )
        )

        menu.addSeparator()

        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(
            lambda _=False, p=parent, name="Copy": (
                self._handle_piano_roll_action(name, p)
            )
        )

        paste_action = menu.addAction("Paste")
        paste_action.triggered.connect(
            lambda _=False, p=parent, name="Paste": (
                self._handle_piano_roll_action(name, p)
            )
        )

        menu.addSeparator()

        octave_up_action = menu.addAction("Octave Up")
        octave_up_action.triggered.connect(
            lambda _=False, p=parent, name="Octave Up": (
                self._handle_piano_roll_action(name, p)
            )
        )

        octave_down_action = menu.addAction("Octave Down")
        octave_down_action.triggered.connect(
            lambda _=False, p=parent, name="Octave Down": (
                self._handle_piano_roll_action(name, p)
            )
        )

        menu.addSeparator()

        add_clip_action = menu.addAction("Add Audio Clip")
        add_clip_action.triggered.connect(
            lambda _=False, p=parent, name="Add Audio Clip": (
                self._handle_piano_roll_action(name, p)
            )
        )

        return menu

    def create_waveform_menu(self, parent):
        menu = QMenu(parent)

        normalize_action = menu.addAction("Normalize")
        normalize_action.triggered.connect(
            lambda _=False, p=parent, name="Normalize": (
                self._handle_waveform_action(name, p)
            )
        )

        reverse_action = menu.addAction("Reverse")
        reverse_action.triggered.connect(
            lambda _=False, p=parent, name="Reverse": (
                self._handle_waveform_action(name, p)
            )
        )

        fade_in_action = menu.addAction("Fade In")
        fade_in_action.triggered.connect(
            lambda _=False, p=parent, name="Fade In": (
                self._handle_waveform_action(name, p)
            )
        )

        fade_out_action = menu.addAction("Fade Out")
        fade_out_action.triggered.connect(
            lambda _=False, p=parent, name="Fade Out": (
                self._handle_waveform_action(name, p)
            )
        )

        menu.addSeparator()

        cut_action = menu.addAction("Cut")
        cut_action.triggered.connect(
            lambda _=False, p=parent, name="Cut": self._handle_waveform_action(
                name, p
            )
        )

        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(
            lambda _=False, p=parent, name="Copy": self._handle_waveform_action(
                name, p
            )
        )

        paste_action = menu.addAction("Paste")
        paste_action.triggered.connect(
            lambda _=False, p=parent, name="Paste": (
                self._handle_waveform_action(name, p)
            )
        )

        menu.addSeparator()

        export_selection_action = menu.addAction("Export Selection")
        export_selection_action.triggered.connect(
            lambda _=False, p=parent, name="Export Selection": (
                self._handle_waveform_action(name, p)
            )
        )

        return menu

    def create_track_menu(self, parent):
        menu = QMenu(parent)

        duplicate_track_action = menu.addAction("Duplicate Track")
        duplicate_track_action.triggered.connect(
            lambda _=False, p=parent, name="Duplicate Track": (
                self._handle_track_action(name, p)
            )
        )

        delete_track_action = menu.addAction("Delete Track")
        delete_track_action.triggered.connect(
            lambda _=False, p=parent, name="Delete Track": (
                self._handle_track_action(name, p)
            )
        )

        menu.addSeparator()

        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(
            lambda _=False, p=parent, name="Rename": self._handle_track_action(
                name, p
            )
        )

        change_color_action = menu.addAction("Change Color")
        change_color_action.triggered.connect(
            lambda _=False, p=parent, name="Change Color": (
                self._handle_track_action(name, p)
            )
        )

        menu.addSeparator()

        mute_action = menu.addAction("Mute")
        mute_action.triggered.connect(
            lambda _=False, p=parent, name="Mute": self._handle_track_action(
                name, p
            )
        )

        solo_action = menu.addAction("Solo")
        solo_action.triggered.connect(
            lambda _=False, p=parent, name="Solo": self._handle_track_action(
                name, p
            )
        )

        menu.addSeparator()

        export_track_action = menu.addAction("Export Track")
        export_track_action.triggered.connect(
            lambda _=False, p=parent, name="Export Track": (
                self._handle_track_action(name, p)
            )
        )

        return menu

    def create_score_editor_menu(self, parent):
        menu = QMenu(parent)

        transpose_up_action = menu.addAction("Transpose Up")
        transpose_up_action.triggered.connect(
            lambda _=False, p=parent, name="Transpose Up": (
                self._handle_score_editor_action(name, p)
            )
        )

        transpose_down_action = menu.addAction("Transpose Down")
        transpose_down_action.triggered.connect(
            lambda _=False, p=parent, name="Transpose Down": (
                self._handle_score_editor_action(name, p)
            )
        )

        menu.addSeparator()

        add_articulation_action = menu.addAction("Add Articulation")
        add_articulation_action.triggered.connect(
            lambda _=False, p=parent, name="Add Articulation": (
                self._handle_score_editor_action(name, p)
            )
        )

        change_dynamic_action = menu.addAction("Change Dynamic")
        change_dynamic_action.triggered.connect(
            lambda _=False, p=parent, name="Change Dynamic": (
                self._handle_score_editor_action(name, p)
            )
        )

        menu.addSeparator()

        add_ornament_action = menu.addAction("Add Ornament")
        add_ornament_action.triggered.connect(
            lambda _=False, p=parent, name="Add Ornament": (
                self._handle_score_editor_action(name, p)
            )
        )

        toggle_tie_action = menu.addAction("Toggle Tie")
        toggle_tie_action.triggered.connect(
            lambda _=False, p=parent, name="Toggle Tie": (
                self._handle_score_editor_action(name, p)
            )
        )

        menu.addSeparator()

        change_clef_action = menu.addAction("Change Clef")
        change_clef_action.triggered.connect(
            lambda _=False, p=parent, name="Change Clef": (
                self._handle_score_editor_action(name, p)
            )
        )

        return menu
