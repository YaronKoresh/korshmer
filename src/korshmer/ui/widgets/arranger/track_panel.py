from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from korshmer.constants import HELP_TEXTS
from korshmer.ui.widgets.material import (
    MAT_ERROR,
    MAT_PRIMARY,
    MAT_TEXT,
    MAT_WARNING,
    HelpButton,
    MaterialButton,
    MaterialToggleButton,
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
        note_total = len(panel.model.notes_for_track(track.id))
        self.name_edit.setToolTip(
            f"{LANE_HELP}\n{note_total} notes · double-click the row to "
            "select them"
        )
        self.name_edit.editingFinished.connect(self._commit_name)

        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(PROGRAM_NAMES)
        self.instrument_combo.addItem(DRUMS_LABEL)
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

        layout.addWidget(self.swatch)
        layout.addWidget(self.name_edit, 1)
        layout.addWidget(self.instrument_combo)
        layout.addWidget(self.btn_mute)
        layout.addWidget(self.btn_solo)
        layout.addWidget(self.btn_visible)
        layout.addWidget(self.btn_delete)
        self.set_active(False)

    def _commit_name(self):
        self.panel.rename_track(self.track_id, self.name_edit.text())

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

        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(
            f"background-color: {clip.color}; border-radius: 8px;"
        )

        name = QLabel(clip.name)
        name.setToolTip(f"{CLIP_HELP}\n{clip.path}")
        name.setStyleSheet(f"color: {MAT_TEXT}; font-weight: bold;")

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
        self.gain_slider.setRange(0, 200)
        self.gain_slider.setValue(int(round(clip.gain * 100)))
        self.gain_slider.setFixedWidth(80)
        self.gain_slider.setToolTip(f"Clip gain {self.gain_slider.value()}%")
        self.gain_slider.valueChanged.connect(self._gain_changed)

        self.btn_remove = make_small_button("✕", "Remove clip")
        self.btn_remove.clicked.connect(
            lambda: self.panel.remove_clip(self.clip_id)
        )

        layout.addWidget(swatch)
        layout.addWidget(name, 1)
        layout.addWidget(self.btn_mute)
        layout.addWidget(self.btn_solo)
        layout.addWidget(self.gain_slider)
        layout.addWidget(self.btn_remove)

    def _gain_changed(self, value):
        self.gain_slider.setToolTip(f"Clip gain {value}%")
        self.panel.set_clip_gain(self.clip_id, value / 100.0)


class TrackPanel(QWidget):
    active_track_changed = Signal(int)
    tracks_changed = Signal()
    select_track_notes_requested = Signal(int)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.active_track_id = None
        self._rows = []
        self._clip_rows = []
        self._signature = None
        self._status_text = ""

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
        self.lbl_status.setStyleSheet(f"color: {MAT_TEXT}; font-size: 11px;")
        self.lbl_status.setWordWrap(True)

        layout.addLayout(header)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self.lbl_status)
        self.refresh()

    def set_model(self, model):
        self.model = model
        self.active_track_id = None
        self._signature = None
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
            )
            for t in self.model.tracks.values()
        )
        clips = tuple(
            (c.id, c.name, c.color, c.muted, c.solo, round(c.gain, 3))
            for c in self.model.clips
        )
        return tracks, clips

    def _clear_rows(self):
        for row in self._rows + self._clip_rows:
            self.rows_layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._rows = []
        self._clip_rows = []

    def refresh(self):
        signature = self._structure_signature()
        changed = signature != self._signature
        if changed:
            self._signature = signature
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
        if text != self._status_text:
            self._status_text = text
            self.lbl_status.setText(text)

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

    def set_clip_gain(self, clip_id, gain):
        self.model.set_clip_gain(clip_id, gain)
        self.update_status()
        self.tracks_changed.emit()

    def remove_clip(self, clip_id):
        if self.model.clip_by_id(clip_id) is None:
            return
        self.model.snapshot()
        self.model.remove_clip(clip_id)
        self._commit()
