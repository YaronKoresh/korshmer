from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from korshmer.constants import HELP_TEXTS
from korshmer.ui.viewmodels.editor_vm import (
    MAX_CLIP_GAIN,
    MAX_TRACK_GAIN,
    clamp_pan,
    clamp_track_gain,
)
from korshmer.ui.widgets.material import (
    MAT_ERROR,
    MAT_PRIMARY,
    MAT_TEXT,
    MAT_WARNING,
    HelpButton,
    MaterialButton,
    MaterialToggleButton,
)
from korshmer.utils.soundfont_ranges import (
    instrument_label,
    playable_range,
    range_text,
)
from korshmer.utils.text_input import limit_line_edit
from korshmer.utils.track_layout import DRUMS_NAME, PROGRAM_NAMES

ROW_HEIGHT = 34
SMALL_BUTTON = 26
DRUMS_LABEL = DRUMS_NAME
LANE_HELP = HELP_TEXTS.get(
    "lane",
    "Each lane is one instrument track. Click a lane to make it the target for new notes.",
)
CLIP_HELP = HELP_TEXTS.get(
    "clip",
    "Audio clip placed on the editor timeline. Drag to move, double-click to mute.",
)


def swatch_style(color):
    return (
        f"QPushButton {{ background-color: {color}; border: 1px solid #333; "
        f"border-radius: 8px; }}"
    )


def clip_swatch_style(color):
    return f"background-color: {color}; border-radius: 8px;"


def make_small_toggle(text, color, tooltip):
    button = MaterialToggleButton(text, color)
    button.setFixedSize(SMALL_BUTTON, SMALL_BUTTON)
    button.setToolTip(tooltip)
    return button


def make_small_button(text, tooltip):
    button = QPushButton(text)
    button.setFixedSize(SMALL_BUTTON, SMALL_BUTTON)
    button.setToolTip(tooltip)
    button.setCursor(Qt.PointingHandCursor)
    return button


SILENT_NOTES_NOTICE = (
    "{count} notes on {track} are outside what {instrument} can play "
    "({keys}) and stay silent"
)
SILENT_NOTICE_LINES = 3
SILENT_NOTES_STYLE = "color: #C5221F; font-size: 11px; font-weight: bold;"
LEVEL_STEP = 5
MIX_TOLERANCE = 1e-9
PAN_STEPS = 100
PAN_SLIDER_WIDTH = 44
INSTRUMENT_MIN_WIDTH = 64
PAN_HELP = "Drag to pan left or right. Double-click to return to the centre."
GESTURE_KEYS = (QEvent.Type.KeyPress, QEvent.Type.Wheel)
LEVEL_BOX_WIDTH = 50
LEVEL_HELP = (
    "Level in percent. Type a number and press Enter, or use the Up and "
    "Down arrow keys to move in steps of 5."
)


def pan_text(pan):
    percent = int(round(abs(pan) * PAN_STEPS))
    if percent == 0:
        return "Centre"
    return f"{percent}% {'left' if pan < 0 else 'right'}"


def show_value(widget, value):
    if widget.value() == value:
        return
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)


def show_checked(button, checked):
    if button.isChecked() == checked:
        return
    button.blockSignals(True)
    button.setChecked(checked)
    button.blockSignals(False)
    button.update_style()


class PanSlider(QSlider):
    def __init__(self, owner_name, pan, on_change):
        super().__init__(Qt.Horizontal)
        self.setRange(-PAN_STEPS, PAN_STEPS)
        self.setSingleStep(LEVEL_STEP)
        self.setPageStep(LEVEL_STEP)
        self.setFixedWidth(PAN_SLIDER_WIDTH)
        self.show_pan(pan)
        self.name_owner(owner_name)
        self.valueChanged.connect(self._describe)
        self.valueChanged.connect(
            lambda value: on_change(value / float(PAN_STEPS))
        )

    def name_owner(self, owner_name):
        self.setAccessibleName(f"Pan of {owner_name}")

    def show_pan(self, pan):
        show_value(self, int(round(pan * PAN_STEPS)))
        self._describe(self.value())

    def _describe(self, value):
        position = pan_text(value / float(PAN_STEPS))
        self.setToolTip(f"Pan: {position}\n{PAN_HELP}")

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self.setValue(0)


def make_level_box(percent, maximum, on_change):
    box = QSpinBox()
    box.setRange(0, int(maximum))
    box.setSingleStep(LEVEL_STEP)
    box.setSuffix("%")
    box.setValue(int(round(percent)))
    box.setFixedWidth(LEVEL_BOX_WIDTH)
    box.setButtonSymbols(QAbstractSpinBox.NoButtons)
    box.setAlignment(Qt.AlignCenter)
    box.setKeyboardTracking(False)
    box.setToolTip(LEVEL_HELP)
    box.valueChanged.connect(on_change)
    return box


class TrackRow(QFrame):
    def __init__(self, panel, track):
        super().__init__(panel)
        self.panel = panel
        self.track_id = track.id
        self.setFixedHeight(ROW_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.swatch = QPushButton()
        self.swatch.setFixedSize(16, 16)
        self.swatch.setToolTip("Click to cycle the track color")
        self.swatch.setStyleSheet(swatch_style(track.color))
        self.swatch.clicked.connect(
            lambda: self.panel.cycle_track_color(self.track_id)
        )

        self.name_edit = limit_line_edit(QLineEdit(track.name))
        self.name_edit.setMinimumWidth(60)
        self.name_edit.setToolTip(self._name_help(track))
        self.name_edit.editingFinished.connect(self._commit_name)

        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(PROGRAM_NAMES)
        self.instrument_combo.addItem(DRUMS_LABEL)
        self.instrument_combo.setMinimumWidth(INSTRUMENT_MIN_WIDTH)
        self.instrument_combo.setMaximumWidth(140)
        self.instrument_combo.setToolTip(
            f"Instrument: {track.instrument_label()}"
        )
        self.instrument_combo.setCurrentIndex(
            128 if track.is_drum else track.program
        )
        self.instrument_combo.currentIndexChanged.connect(
            lambda index: self.panel.change_track_instrument(
                self.track_id, index
            )
        )

        self.btn_mute = make_small_toggle("M", MAT_ERROR, "Mute track")
        self.btn_mute.setChecked(track.muted)
        self.btn_mute.toggled.connect(
            lambda checked: self.panel.set_track_mute(self.track_id, checked)
        )

        self.btn_solo = make_small_toggle("S", MAT_WARNING, "Solo track")
        self.btn_solo.setChecked(track.solo)
        self.btn_solo.toggled.connect(
            lambda checked: self.panel.set_track_solo(self.track_id, checked)
        )

        self.btn_visible = make_small_toggle(
            "\U0001f441", MAT_PRIMARY, "Show or hide the lane"
        )
        self.btn_visible.setChecked(track.visible)
        self.btn_visible.toggled.connect(
            lambda checked: self.panel.set_track_visible(self.track_id, checked)
        )

        self.btn_delete = make_small_button("✕", "Delete track")
        self.btn_delete.clicked.connect(
            lambda: self.panel.delete_track(self.track_id)
        )

        self.level_box = make_level_box(
            track.gain * 100,
            MAX_TRACK_GAIN * 100,
            lambda value: self.panel.set_track_gain(
                self.track_id, value / 100.0
            ),
        )
        self.level_box.setAccessibleName(f"Level of {track.name}")
        self.pan_slider = PanSlider(
            track.name,
            track.pan,
            lambda pan: self.panel.set_track_pan(self.track_id, pan),
        )
        panel.watch_mix_control(self.level_box)
        panel.watch_mix_control(self.pan_slider)

        layout.addWidget(self.swatch)
        layout.addWidget(self.name_edit, 1)
        layout.addWidget(self.instrument_combo)
        layout.addWidget(self.level_box)
        layout.addWidget(self.pan_slider)
        layout.addWidget(self.btn_mute)
        layout.addWidget(self.btn_solo)
        layout.addWidget(self.btn_visible)
        layout.addWidget(self.btn_delete)
        self.set_active(False)

    def _commit_name(self):
        self.panel.rename_track(self.track_id, self.name_edit.text())

    def _name_help(self, track):
        note_total = len(self.panel.model.notes_for_track(track.id))
        return (
            f"{LANE_HELP}\n{note_total} notes · double-click the row to "
            "select them"
        )

    def show_track(self, track):
        self.swatch.setStyleSheet(swatch_style(track.color))
        if self.name_edit.text() != track.name:
            self.name_edit.setText(track.name)
        self.name_edit.setToolTip(self._name_help(track))
        self.instrument_combo.setToolTip(
            f"Instrument: {track.instrument_label()}"
        )
        index = 128 if track.is_drum else track.program
        if self.instrument_combo.currentIndex() != index:
            self.instrument_combo.blockSignals(True)
            self.instrument_combo.setCurrentIndex(index)
            self.instrument_combo.blockSignals(False)
        show_checked(self.btn_mute, track.muted)
        show_checked(self.btn_solo, track.solo)
        show_checked(self.btn_visible, track.visible)
        show_value(self.level_box, int(round(track.gain * 100)))
        self.level_box.setAccessibleName(f"Level of {track.name}")
        self.pan_slider.show_pan(track.pan)
        self.pan_slider.name_owner(track.name)

    def set_active(self, active):
        border = MAT_PRIMARY if active else "transparent"
        background = "#E8F0FE" if active else "transparent"
        self.setStyleSheet(
            f"TrackRow {{ border: 1px solid {border}; border-radius: 6px; "
            f"background-color: {background}; }}"
        )

    def mousePressEvent(self, event):
        self.panel.set_active_track(self.track_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.panel.request_select_track_notes(self.track_id)
        super().mouseDoubleClickEvent(event)


class ClipRow(QFrame):
    def __init__(self, panel, clip):
        super().__init__(panel)
        self.panel = panel
        self.clip_id = clip.id
        self.setFixedHeight(ROW_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.swatch = QLabel()
        self.swatch.setFixedSize(16, 16)
        self.swatch.setStyleSheet(clip_swatch_style(clip.color))

        self.name_label = QLabel(clip.name)
        self.name_label.setToolTip(f"{CLIP_HELP}\n{clip.path}")
        self.name_label.setStyleSheet(f"color: {MAT_TEXT}; font-weight: bold;")

        self.btn_mute = make_small_toggle("M", MAT_ERROR, "Mute clip")
        self.btn_mute.setChecked(clip.muted)
        self.btn_mute.toggled.connect(
            lambda checked: self.panel.set_clip_mute(self.clip_id, checked)
        )

        self.btn_solo = make_small_toggle("S", MAT_WARNING, "Solo clip")
        self.btn_solo.setChecked(clip.solo)
        self.btn_solo.toggled.connect(
            lambda checked: self.panel.set_clip_solo(self.clip_id, checked)
        )

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, int(MAX_CLIP_GAIN * 100))
        self.gain_slider.setValue(int(round(clip.gain * 100)))
        self.gain_slider.setFixedWidth(80)
        self.gain_slider.setToolTip(f"Clip gain {self.gain_slider.value()}%")
        self.gain_slider.valueChanged.connect(self._gain_changed)
        self.level_box = make_level_box(
            self.gain_slider.value(),
            MAX_CLIP_GAIN * 100,
            self.gain_slider.setValue,
        )
        self.gain_slider.setAccessibleName(f"Level of {clip.name}")
        self.level_box.setAccessibleName(f"Level of {clip.name} in percent")
        self.pan_slider = PanSlider(
            clip.name,
            clip.pan,
            lambda pan: self.panel.set_clip_pan(self.clip_id, pan),
        )
        for control in (self.gain_slider, self.level_box, self.pan_slider):
            panel.watch_mix_control(control)

        self.btn_remove = make_small_button("✕", "Remove clip")
        self.btn_remove.clicked.connect(
            lambda: self.panel.remove_clip(self.clip_id)
        )

        layout.addWidget(self.swatch)
        layout.addWidget(self.name_label, 1)
        layout.addWidget(self.btn_mute)
        layout.addWidget(self.btn_solo)
        layout.addWidget(self.gain_slider)
        layout.addWidget(self.level_box)
        layout.addWidget(self.pan_slider)
        layout.addWidget(self.btn_remove)

    def _gain_changed(self, value):
        self.gain_slider.setToolTip(f"Clip gain {value}%")
        show_value(self.level_box, value)
        self.panel.set_clip_gain(self.clip_id, value / 100.0)

    def show_clip(self, clip):
        self.swatch.setStyleSheet(clip_swatch_style(clip.color))
        if self.name_label.text() != clip.name:
            self.name_label.setText(clip.name)
        self.name_label.setToolTip(f"{CLIP_HELP}\n{clip.path}")
        show_checked(self.btn_mute, clip.muted)
        show_checked(self.btn_solo, clip.solo)
        percent = int(round(clip.gain * 100))
        show_value(self.gain_slider, percent)
        self.gain_slider.setToolTip(f"Clip gain {percent}%")
        show_value(self.level_box, percent)
        self.pan_slider.show_pan(clip.pan)
        self.pan_slider.name_owner(clip.name)


class TrackPanel(QWidget):
    active_track_changed = Signal(int)
    tracks_changed = Signal()
    mix_changed = Signal()
    select_track_notes_requested = Signal(int)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.active_track_id = None
        self._rows = []
        self._clip_rows = []
        self._signature = None
        self._status_text = ""
        self._gesture_owner = None
        self._gesture_by_mouse = False
        self._gesture_step = None
        self.mix_changed.connect(self.refresh)

        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel("TRACKS")
        title.setStyleSheet(f"color: {MAT_TEXT}; font-weight: bold;")
        header.addWidget(title)
        header.addWidget(HelpButton(LANE_HELP), alignment=Qt.AlignLeft)
        header.addStretch()
        self.btn_add_track = MaterialButton("✚ Track", "tonal")
        self.btn_add_track.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Preferred
        )
        self.btn_add_track.clicked.connect(self.add_track)
        header.addWidget(self.btn_add_track)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.rows_layout = QVBoxLayout(self.container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        self.rows_layout.addStretch()
        self.scroll.setWidget(self.container)

        self.lbl_status = QLabel("")
        self._status_style = f"color: {MAT_TEXT}; font-size: 11px;"
        self.lbl_status.setStyleSheet(self._status_style)
        self.lbl_status.setWordWrap(True)

        layout.addLayout(header)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self.lbl_status)
        self.refresh()

    def set_model(self, model):
        self.model = model
        self.active_track_id = None
        self._signature = None
        self._end_mix_gesture()
        self.refresh()

    def _structure_signature(self):
        tracks = tuple(
            (
                t.id,
                t.name,
                t.program,
                t.is_drum,
                t.color,
                t.muted,
                t.solo,
                t.visible,
                round(t.gain, 3),
                round(t.pan, 3),
            )
            for t in self.model.tracks.values()
        )
        clips = tuple(
            (
                c.id,
                c.name,
                c.path,
                c.color,
                c.muted,
                c.solo,
                round(c.gain, 3),
                round(c.pan, 3),
            )
            for c in self.model.clips
        )
        return tracks, clips

    def _shown_clip_rows(self):
        return [row for row in self._clip_rows if isinstance(row, ClipRow)]

    def _rows_match_the_song(self):
        return [row.track_id for row in self._rows] == list(
            self.model.tracks
        ) and [row.clip_id for row in self._shown_clip_rows()] == [
            clip.id for clip in self.model.clips
        ]

    def _update_rows(self):
        for row in self._rows:
            row.show_track(self.model.tracks[row.track_id])
        for row, clip in zip(
            self._shown_clip_rows(), self.model.clips, strict=True
        ):
            row.show_clip(clip)
        self._highlight_active()

    def _clear_rows(self):
        for row in self._rows + self._clip_rows:
            self.rows_layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._rows = []
        self._clip_rows = []
        self._end_mix_gesture()

    def refresh(self):
        signature = self._structure_signature()
        changed = signature != self._signature
        if changed:
            self._signature = signature
            if self._rows_match_the_song():
                self._update_rows()
            else:
                self._rebuild_rows()
        self.update_status()
        return changed

    def _rebuild_rows(self):
        self._clear_rows()
        insert_at = 0
        for track in self.model.tracks.values():
            row = TrackRow(self, track)
            self.rows_layout.insertWidget(insert_at, row)
            self._rows.append(row)
            insert_at += 1
        if self.model.clips:
            header = QLabel("AUDIO CLIPS")
            header.setStyleSheet(
                f"color: {MAT_TEXT}; font-weight: bold; padding-top: 6px;"
            )
            header.setToolTip(CLIP_HELP)
            self.rows_layout.insertWidget(insert_at, header)
            self._clip_rows.append(header)
            insert_at += 1
            for clip in self.model.clips:
                row = ClipRow(self, clip)
                self.rows_layout.insertWidget(insert_at, row)
                self._clip_rows.append(row)
                insert_at += 1
        self._highlight_active()

    def update_status(self):
        text = self.model.summary_text()
        silent = self.model.silent_notes(playable_range)
        for track_id, count in list(silent.items())[:SILENT_NOTICE_LINES]:
            track = self.model.tracks[track_id]
            keys = playable_range(track.program, track.is_drum)
            notice = SILENT_NOTES_NOTICE.format(
                count=count,
                track=track.name,
                instrument=instrument_label(track),
                keys=range_text(keys),
            )
            text = f"{text}\n{notice}"
        if text != self._status_text:
            self._status_text = text
            self.lbl_status.setText(text)
            self.lbl_status.setStyleSheet(
                SILENT_NOTES_STYLE if silent else self._status_style
            )

    def _highlight_active(self):
        if self.active_track_id not in self.model.tracks:
            self.active_track_id = self.model.first_track_id()
        for row in self._rows:
            row.set_active(row.track_id == self.active_track_id)

    def set_active_track(self, track_id):
        if track_id not in self.model.tracks:
            return
        self.active_track_id = track_id
        self._highlight_active()
        self.active_track_changed.emit(track_id)

    def _commit(self):
        self.refresh()
        self.tracks_changed.emit()

    def add_track(self):
        self.model.snapshot()
        track = self.model.add_track(f"Track {len(self.model.tracks) + 1}")
        self.active_track_id = track.id
        self._commit()
        self.active_track_changed.emit(track.id)
        return track

    def rename_track(self, track_id, name):
        track = self.model.tracks.get(track_id)
        if track is None or track.name == name.strip() or not name.strip():
            return
        self.model.snapshot()
        self.model.rename_track(track_id, name)
        self._commit()

    def change_track_instrument(self, track_id, index):
        track = self.model.tracks.get(track_id)
        if track is None:
            return
        is_drum = index >= 128
        program = track.program if is_drum else max(0, index)
        if track.is_drum == is_drum and track.program == program:
            return
        self.model.snapshot()
        self.model.set_track_drum(track_id, is_drum)
        self.model.set_track_program(track_id, program)
        self._commit()

    def set_track_mute(self, track_id, muted):
        self.model.snapshot()
        self.model.set_track_mute(track_id, muted)
        self._commit()

    def set_track_solo(self, track_id, solo):
        self.model.snapshot()
        self.model.set_track_solo(track_id, solo)
        self._commit()

    def set_track_visible(self, track_id, visible):
        self.model.snapshot()
        self.model.set_track_visible(track_id, visible)
        self._commit()

    def cycle_track_color(self, track_id):
        track = self.model.tracks.get(track_id)
        if track is None:
            return
        self.model.snapshot()
        self.model.set_track_color(
            track_id, self.model.next_palette_color(track.color)
        )
        self._commit()

    def delete_track(self, track_id):
        if track_id not in self.model.tracks:
            return
        self.model.snapshot()
        self.model.remove_track(track_id)
        self._commit()

    def set_clip_mute(self, clip_id, muted):
        self.model.snapshot()
        self.model.set_clip_mute(clip_id, muted)
        self._commit()

    def set_clip_solo(self, clip_id, solo):
        self.model.snapshot()
        self.model.set_clip_solo(clip_id, solo)
        self._commit()

    def request_select_track_notes(self, track_id):
        if track_id in self.model.tracks:
            self.set_active_track(track_id)
            self.select_track_notes_requested.emit(track_id)

    def watch_mix_control(self, control):
        control.installEventFilter(self)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            self._begin_mix_gesture(watched, by_mouse=True)
        elif kind == QEvent.Type.MouseButtonDblClick:
            if self._gesture_owner is not watched:
                self._begin_mix_gesture(watched, by_mouse=True)
        elif kind in GESTURE_KEYS and (
            self._gesture_owner is not watched or self._gesture_by_mouse
        ):
            self._begin_mix_gesture(watched, by_mouse=False)
        elif kind == QEvent.Type.FocusOut and self._gesture_owner is watched:
            self._end_mix_gesture()
        return super().eventFilter(watched, event)

    def _begin_mix_gesture(self, owner, by_mouse):
        self._gesture_owner = owner
        self._gesture_by_mouse = by_mouse
        self._gesture_step = None

    def _end_mix_gesture(self):
        self._gesture_owner = None
        self._gesture_by_mouse = False
        self._gesture_step = None

    def _save_before_mix_edit(self, target):
        step = (target, self.model.history_mark())
        if self._gesture_owner is not None and self._gesture_step == step:
            return
        self.model.snapshot()
        self._gesture_step = (target, self.model.history_mark())

    def _edit_mix(self, target, current, wanted, apply):
        if current is None or abs(current - wanted) < MIX_TOLERANCE:
            return
        self._save_before_mix_edit(target)
        apply()
        self.mix_changed.emit()

    def set_track_gain(self, track_id, gain):
        track = self.model.tracks.get(track_id)
        self._edit_mix(
            ("track gain", track_id),
            None if track is None else track.gain,
            clamp_track_gain(gain),
            lambda: self.model.set_track_gain(track_id, gain),
        )

    def set_track_pan(self, track_id, pan):
        track = self.model.tracks.get(track_id)
        self._edit_mix(
            ("track pan", track_id),
            None if track is None else track.pan,
            clamp_pan(pan),
            lambda: self.model.set_track_pan(track_id, pan),
        )

    def set_clip_gain(self, clip_id, gain):
        clip = self.model.clip_by_id(clip_id)
        self._edit_mix(
            ("clip gain", clip_id),
            None if clip is None else clip.gain,
            max(0.0, min(MAX_CLIP_GAIN, float(gain))),
            lambda: self.model.set_clip_gain(clip_id, gain),
        )

    def set_clip_pan(self, clip_id, pan):
        clip = self.model.clip_by_id(clip_id)
        self._edit_mix(
            ("clip pan", clip_id),
            None if clip is None else clip.pan,
            clamp_pan(pan),
            lambda: self.model.set_clip_pan(clip_id, pan),
        )

    def remove_clip(self, clip_id):
        if self.model.clip_by_id(clip_id) is None:
            return
        self.model.snapshot()
        self.model.remove_clip(clip_id)
        self._commit()
