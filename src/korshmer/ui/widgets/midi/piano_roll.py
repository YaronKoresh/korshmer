import math
import time

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from korshmer.constants import GRID_RESOLUTIONS
from korshmer.exceptions import InvalidFileError
from korshmer.ui.utils.throttle import EventThrottle
from korshmer.ui.viewmodels.editor_vm import (
    MIN_NOTE_DURATION,
    EditorModel,
    clamp_velocity,
)
from korshmer.ui.viewmodels.piano_roll_vm import PianoRollViewModel
from korshmer.ui.widgets.material import MAT_BG, MAT_ERROR, MAT_PRIMARY
from korshmer.ui.widgets.midi.audio_clip_item import (
    CLIP_LANE_HEIGHT,
    CLIP_LANE_PADDING,
    AudioClipItem,
    ClipLaneBackgroundItem,
)
from korshmer.ui.widgets.midi.grid_painter import GridPainter
from korshmer.ui.widgets.midi.piano_roll_renderer import (
    MUTED_TRACK_ALPHA,
    NoteRenderer,
)
from korshmer.utils.filesystem import sanitize_path
from korshmer.utils.tempo import validate_bpm

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a")
MIDI_EXTENSIONS = (".mid", ".midi")
CLICK_DRAG_THRESHOLD_PX = 3
DEFAULT_GRID_DIVISIONS = GRID_RESOLUTIONS["1/16"]
DEFAULT_SCENE_WIDTH = 5000
SCENE_RIGHT_MARGIN = 400
VELOCITY_WHEEL_STEP = 5
VELOCITY_WHEEL_GROUP_SECONDS = 0.8


MIN_ZOOM = 0.1
MAX_ZOOM = 100.0


def clamp_zoom(value, lower=MIN_ZOOM, upper=MAX_ZOOM):
    if not math.isfinite(value) or value <= 0:
        return None
    return max(lower, min(upper, value))


def is_midi_path(path):
    return str(path).lower().endswith(MIDI_EXTENSIONS)


def is_audio_path(path):
    return str(path).lower().endswith(AUDIO_EXTENSIONS)


def alt_modifier_held():
    return bool(QApplication.keyboardModifiers() & Qt.AltModifier)


class PianoKeyItem(QGraphicsRectItem):
    def __init__(
        self,
        pitch,
        x_pos,
        width,
        velocity,
        grid_x,
        grid_y=20,
        track_id=0,
        color=None,
        note_id=None,
    ):
        y = (127 - pitch) * grid_y
        super().__init__(0, 0, width, grid_y)
        self.setPos(x_pos, y)

        self.pitch = pitch
        self.velocity = clamp_velocity(velocity)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.track_id = track_id
        self.note_id = note_id
        self.color = QColor(color) if color else QColor(MAT_PRIMARY)
        self.track_name = ""
        self.grid_divisions = 1
        self.grid_origin = 0.0
        self.snap_enabled = True
        self.dim_alpha = 1.0
        self.total_rows = 128

        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.resize_mode = None
        self.resize_start_x = 0
        self.initial_rect_width = 0
        self.initial_pos_x = 0
        self.refresh_style()

    @property
    def snap_step(self):
        return self.grid_x / max(1, int(self.grid_divisions))

    def snapping_active(self):
        return self.snap_enabled and not alt_modifier_held()

    def set_track_style(self, color, track_name, muted=False):
        if color:
            self.color = QColor(color)
        self.track_name = track_name or ""
        self.dim_alpha = MUTED_TRACK_ALPHA if muted else 1.0
        self.refresh_style()

    def set_velocity(self, velocity):
        self.velocity = clamp_velocity(velocity)
        self.refresh_style()
        self.update()

    def refresh_style(self):
        fill, pen_color = NoteRenderer.note_colors(
            self.color, self.velocity, self.dim_alpha
        )
        self.setBrush(QBrush(fill))
        self.setPen(QPen(pen_color, 1))
        self.update_tooltip()

    def current_pitch(self):
        pitch = int(round(127 - self.y() / self.grid_y))
        return max(0, min(127, pitch))

    def update_tooltip(self):
        self.setToolTip(
            NoteRenderer.note_tooltip(
                self.current_pitch(), self.velocity, self.track_name
            )
        )

    def paint(self, painter, _option, _widget=None):
        rect = self.rect()
        label = None
        if rect.width() > 24:
            label = NoteRenderer.note_label(self.y(), self.grid_y)
        NoteRenderer.paint_note(
            painter,
            rect,
            self.brush(),
            self.pen(),
            self.isSelected(),
            label,
            self.dim_alpha,
        )

    def hoverMoveEvent(self, event):
        margin = 10
        x = event.pos().x()
        width = self.rect().width()

        if x <= margin or x >= width - margin:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            margin = 10
            x = event.pos().x()
            width = self.rect().width()

            if x <= margin:
                self.resize_mode = "left"
            elif x >= width - margin:
                self.resize_mode = "right"
            else:
                self.resize_mode = None

            if self.resize_mode:
                self.resize_start_x = event.scenePos().x()
                self.initial_rect_width = width
                self.initial_pos_x = self.pos().x()
                event.accept()
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def _resize_step(self):
        if self.snapping_active():
            return max(1.0, self.snap_step)
        return 1.0

    def mouseMoveEvent(self, event):
        if self.resize_mode:
            mouse_delta = event.scenePos().x() - self.resize_start_x
            subdivision = self._resize_step()

            if self.resize_mode == "right":
                new_width = self.initial_rect_width + mouse_delta
                snapped_width = round(new_width / subdivision) * subdivision
                snapped_width = max(subdivision, snapped_width)

                if abs(snapped_width - self.rect().width()) > 0.1:
                    self.prepareGeometryChange()
                    self.setRect(0, 0, snapped_width, self.rect().height())

            elif self.resize_mode == "left":
                new_pos_x = self.initial_pos_x + mouse_delta
                snapped_pos_x = self._snap_x(new_pos_x, subdivision)

                end_x = self.initial_pos_x + self.initial_rect_width
                new_width = end_x - snapped_pos_x

                if new_width >= subdivision and snapped_pos_x >= 0:
                    self.prepareGeometryChange()
                    self.setPos(snapped_pos_x, self.y())
                    self.setRect(0, 0, new_width, self.rect().height())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.resize_mode = None
        super().mouseReleaseEvent(event)

    def _clamp_row_y(self, y):
        snapped = round(y / self.grid_y) * self.grid_y
        max_y = (self.total_rows - 1) * self.grid_y
        return max(0, min(max_y, snapped))

    def _snap_x(self, x, step):
        return round((x - self.grid_origin) / step) * step + self.grid_origin

    def snapGrid(self, new_pos):
        step = self.snap_step
        x = self._snap_x(new_pos.x(), step)
        if x < 0:
            x += math.ceil(-x / step - 1e-9) * step
        y = self._clamp_row_y(new_pos.y())
        x = max(0, x)
        return QPointF(x, y)

    def clampPosition(self, new_pos):
        return QPointF(max(0, new_pos.x()), self._clamp_row_y(new_pos.y()))

    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.ItemPositionChange
            and self.scene()
            and not self.resize_mode
        ):
            if self.snapping_active():
                return self.snapGrid(value)
            return self.clampPosition(value)

        return super().itemChange(change, value)


class InteractivePianoRoll(QGraphicsView):
    midi_dropped = Signal(str)
    audio_dropped = Signal(str)
    notes_changed = Signal()
    clips_changed = Signal()
    model_changed = Signal()
    context_menu_requested = Signal(QPoint)
    viewport_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = PianoRollViewModel()
        self.model = EditorModel(tempo=self.vm.tempo)
        self._pending_zoom_x = 1.0
        self._pending_zoom_y = 1.0
        self._zoom_throttle = EventThrottle(
            self._apply_pending_zoom, parent=self
        )

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QColor(MAT_BG))
        self.setFocusPolicy(Qt.StrongFocus)

        self.snap_enabled = True
        self.grid_divisions = DEFAULT_GRID_DIVISIONS
        self.default_note_length_beats = 1.0
        self._default_velocity = clamp_velocity(100)
        self.playback_offset = 0
        self.is_monitoring = False

        self._current_track_id = None
        self._note_items = {}
        self._clip_items = {}
        self._lane_item = None
        self._clipboard = []
        self._press_pos = None
        self._press_item = None
        self._drag_snapshot_pending = False
        self._last_velocity_wheel = 0.0

        self.playhead = self._create_playhead()
        self.scene.setSceneRect(
            0,
            0,
            DEFAULT_SCENE_WIDTH,
            self.vm.total_divisions * self.vm.grid_size,
        )
        self.setAcceptDrops(True)

        self.scene.changed.connect(lambda: self.notes_changed.emit())

    def _create_playhead(self):
        playhead = QGraphicsLineItem(0, 0, 0, self._rows_height())
        playhead.setPen(QPen(QColor(MAT_ERROR), 2))
        playhead.setZValue(100)
        self.scene.addItem(playhead)
        return playhead

    @property
    def default_velocity(self):
        return self._default_velocity

    @default_velocity.setter
    def default_velocity(self, value):
        self._default_velocity = clamp_velocity(value)

    @property
    def grid_size(self):
        return self.vm.grid_size

    @grid_size.setter
    def grid_size(self, value):
        self.vm.grid_size = value

    @property
    def pixels_per_second(self):
        return self.vm.pixels_per_second

    @pixels_per_second.setter
    def pixels_per_second(self, value):
        self.vm.pixels_per_second = value
        for item in getattr(self, "_note_items", {}).values():
            item.grid_origin = self.vm.grid_offset_x

    @property
    def tempo(self):
        return self.vm.tempo

    @tempo.setter
    def tempo(self, value):
        value = validate_bpm(value)
        self.vm.tempo = value
        self.model.tempo = value

    @property
    def divisions_per_octave(self):
        return self.vm.divisions_per_octave

    @property
    def total_divisions(self):
        return self.vm.total_divisions

    @property
    def root_note(self):
        return self.vm._root_note_idx

    @property
    def root_note_name(self):
        return self.vm.root_note

    @property
    def _root_note_idx(self):
        return self.vm._root_note_idx

    @property
    def scale_intervals(self):
        return self.vm.scale_intervals

    @property
    def current_track_id(self):
        if self._current_track_id in self.model.tracks:
            return self._current_track_id
        first = self.model.first_track_id()
        return first if first is not None else -1

    @current_track_id.setter
    def current_track_id(self, value):
        self._current_track_id = value

    def set_scale_view(self, root_name, scale_name):
        self.vm.set_scale(root_name, scale_name)
        self.viewport().update()

    def set_grid_size(self, size):
        self.vm.grid_size = size
        if self.model.notes or self.model.clips:
            self.load_model(keep_selection=True)

    def set_grid_divisions(self, divisions):
        self.grid_divisions = max(1, int(divisions))
        for item in self._note_items.values():
            item.grid_divisions = self.grid_divisions

    def set_snap(self, enabled):
        self.snap_enabled = bool(enabled)
        for item in self._note_items.values():
            item.snap_enabled = self.snap_enabled

    def snapping_active(self):
        return self.snap_enabled and not alt_modifier_held()

    def snap_x(self, x):
        return self.vm.snap_to_grid(x, self.grid_divisions)

    @property
    def grid_offset(self):
        return self.vm.grid_offset

    def set_grid_offset(self, seconds):
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value) or value < 0:
            value = 0.0
        self.vm.grid_offset = value
        self.model.grid_offset = value
        for item in self._note_items.values():
            item.grid_origin = self.vm.grid_offset_x
        self.viewport().update()

    def detect_grid_offset(self, midi):
        from korshmer.midi_generation.song_match import estimate_offset

        try:
            offset = estimate_offset(midi, self.vm.tempo)
        except (ValueError, TypeError):
            offset = 0.0
        self.set_grid_offset(offset)
        return offset

    def _clip_snap_x(self, x):
        if self.snapping_active():
            return self.snap_x(x)
        return max(0.0, x)

    def grid_step_seconds(self):
        return self.vm.grid_step_seconds(self.grid_divisions)

    def _rows_height(self):
        return self.vm.total_divisions * self.vm.grid_size

    def _view_scales(self):
        transform = self.transform()
        scale_x = transform.m11() if transform.m11() else 1.0
        scale_y = transform.m22() if transform.m22() else 1.0
        return scale_x, scale_y

    def _lane_geometry(self):
        scale_x, scale_y = self._view_scales()
        top = max(0.0, self.mapToScene(0, 0).y())
        height = CLIP_LANE_HEIGHT / scale_y
        padding = CLIP_LANE_PADDING / scale_y
        return top, height, padding

    def visible_clip_ids(self):
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        return [
            clip_id
            for clip_id, item in self._clip_items.items()
            if item.sceneBoundingRect().intersects(visible)
        ]

    def _sync_clip_lane(self):
        if not self._clip_items:
            if self._lane_item is not None:
                self._lane_item.setVisible(False)
            return
        top, height, padding = self._lane_geometry()
        scale_x, scale_y = self._view_scales()
        if self._lane_item is None:
            self._lane_item = ClipLaneBackgroundItem()
            self.scene.addItem(self._lane_item)
        self._lane_item.setVisible(True)
        rect = self.scene.sceneRect()
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        self._lane_item.set_geometry(
            rect.left(),
            top,
            rect.width(),
            height,
            scale_x,
            scale_y,
            caption_right=visible.right(),
        )
        for item in self._clip_items.values():
            item.set_lane(top, height, padding)

        self.viewport_changed.emit()

    def has_clip_lane(self):
        return bool(self._clip_items)

    def lane_visible(self):
        return self._lane_item is not None and self._lane_item.isVisible()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if self._clip_items:
            self._sync_clip_lane()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._clip_items:
            self._sync_clip_lane()

    def scale(self, sx, sy):
        transform = self.transform()
        cur_x = transform.m11()
        cur_y = transform.m22()
        adj_x = sx
        adj_y = sy

        if cur_x > 0 and math.isfinite(sx) and sx > 0:
            target = clamp_zoom(cur_x * sx)
            adj_x = (target / cur_x) if target is not None else 1.0
        if cur_y > 0 and math.isfinite(sy) and sy > 0:
            target = clamp_zoom(cur_y * sy)
            adj_y = (target / cur_y) if target is not None else 1.0

        super().scale(adj_x, adj_y)
        if self._clip_items:
            self._sync_clip_lane()

    def has_content(self):
        if self.model.notes or self.model.clips:
            return True
        return any(
            isinstance(item, PianoKeyItem) for item in self.scene.items()
        )

    def _update_scene_rect(self):
        bounds = self.scene.itemsBoundingRect()
        width = max(DEFAULT_SCENE_WIDTH, bounds.right() + SCENE_RIGHT_MARGIN)
        height = self._rows_height()
        self.scene.setSceneRect(0, 0, width, height)
        line = self.playhead.line()
        self.playhead.setLine(line.x1(), 0, line.x1(), height)
        self._sync_clip_lane()

    def set_divisions_per_octave(self, divisions):
        old = self.vm.set_divisions(divisions)

        ratio = self.vm.divisions_per_octave / old
        for item in self.scene.items():
            if isinstance(item, PianoKeyItem):
                item.total_rows = self.vm.total_divisions
                new_y = (
                    self.vm.total_divisions - 1 - item.pitch * ratio
                ) * self.vm.grid_size
                item.setY(new_y)

        self._update_scene_rect()
        self.scene.update()
        self.viewport().update()

    def set_tempo(self, new_bpm):
        result = self.vm.compute_tempo_change(new_bpm)
        self.model.tempo = self.vm.tempo
        if result is None:
            return

        old_beat_width, new_beat_width = result
        ratio = new_beat_width / old_beat_width if old_beat_width > 0 else 1.0

        for item in self.scene.items():
            if isinstance(item, PianoKeyItem):
                beats_from_start = (
                    item.x() / old_beat_width if old_beat_width > 0 else 0
                )
                item.setX(beats_from_start * new_beat_width)

                rect = item.rect()
                duration_in_beats = (
                    rect.width() / old_beat_width if old_beat_width > 0 else 1.0
                )
                item.setRect(
                    0, 0, duration_in_beats * new_beat_width, rect.height()
                )
                item.grid_x = new_beat_width

        self.model.scale_note_times(ratio)
        self.model.grid_offset = self.vm.grid_offset
        for item in self._note_items.values():
            item.grid_origin = self.vm.grid_offset_x
        self._update_scene_rect()
        self.scene.update()
        self.viewport().update()

    def get_beat_width(self):
        return self.vm.beat_width

    def stretch_notes(self, ratio):
        if ratio == 1.0 or ratio <= 0:
            return

        for item in self.scene.items():
            if isinstance(item, PianoKeyItem):
                new_x = item.x() * ratio
                current_rect = item.rect()
                new_width = current_rect.width() * ratio
                item.setRect(0, 0, new_width, current_rect.height())
                item.setPos(new_x, item.y())

        self.model.scale_note_times(ratio)
        self._update_scene_rect()

    def zoom(self, factor_x=1.0, factor_y=1.0):
        transform = self.transform()
        scale_x = transform.m11()
        scale_y = transform.m22()
        applied_x = 1.0
        applied_y = 1.0

        if (
            factor_x
            and math.isfinite(factor_x)
            and factor_x > 0
            and scale_x > 0
        ):
            target = clamp_zoom(scale_x * factor_x)
            if target is not None:
                applied_x = target / scale_x

        if (
            factor_y
            and math.isfinite(factor_y)
            and factor_y > 0
            and scale_y > 0
        ):
            total_scene_height = self._rows_height()
            viewport_height = self.viewport().height()
            min_scale_y = (
                viewport_height / total_scene_height
                if total_scene_height > 0
                else MIN_ZOOM
            )
            lower_y = max(MIN_ZOOM, min_scale_y)
            upper_y = max(lower_y, MAX_ZOOM)
            target = clamp_zoom(scale_y * factor_y, lower_y, upper_y)
            if target is not None:
                applied_y = target / scale_y

        if applied_x != 1.0 or applied_y != 1.0:
            self.scale(applied_x, applied_y)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, self.backgroundBrush())
        GridPainter.draw_background(
            painter,
            rect,
            self.transform(),
            self.mapToScene(self.viewport().rect()).boundingRect(),
            self.vm,
            self.grid_divisions,
            CLIP_LANE_HEIGHT if self.has_clip_lane() else 0,
        )

    def _begin_edit_snapshot(self):
        self.model.snapshot()
        self._drag_snapshot_pending = True

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.RightButton:
            self.context_menu_requested.emit(event.pos())
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item is self.playhead:
                item = None
            self._press_pos = event.pos()
            self._press_item = item
            if item is not None:
                self._begin_edit_snapshot()
                if event.modifiers() & Qt.ShiftModifier:
                    item.setSelected(True)
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton or self._press_pos is None:
            return

        press_pos = self._press_pos
        press_item = self._press_item
        snapshot_pending = self._drag_snapshot_pending
        self._press_pos = None
        self._press_item = None
        self._drag_snapshot_pending = False

        moved = (event.pos() - press_pos).manhattanLength() > (
            CLICK_DRAG_THRESHOLD_PX
        )

        if press_item is not None:
            changed = self.sync_model_from_scene()
            if changed:
                self._update_scene_rect()
                if isinstance(press_item, AudioClipItem):
                    self.clips_changed.emit()
                self.model_changed.emit()
            elif snapshot_pending:
                self.model.discard_last_snapshot()
            return

        blocked = event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)
        if not moved and not blocked:
            self._create_note_at(self.mapToScene(press_pos))

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, AudioClipItem):
            self._toggle_clip_mute(item.clip_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _ensure_current_track(self):
        track = self.model.tracks.get(self._current_track_id)
        if track is None:
            track = self.model.ensure_default_track()
            self._current_track_id = track.id
        return track

    def _create_note_at(self, scene_pos):
        if scene_pos.y() < 0:
            return None
        pitch = self.vm.pitch_from_y(scene_pos.y())
        if not 0 <= pitch <= 127:
            return None

        x = scene_pos.x()
        if self.snapping_active():
            x = self.snap_x(x)
        x = max(0.0, x)
        pps = self.vm.pixels_per_second
        start = x / pps if pps > 0 else 0.0
        duration = max(
            MIN_NOTE_DURATION,
            (60.0 / self.vm.tempo) * self.default_note_length_beats,
        )

        self.model.snapshot()
        track = self._ensure_current_track()
        note = self.model.add_note(
            track.id, pitch, start, duration, self.default_velocity
        )
        item = self._make_note_item(note, track)
        self.scene.addItem(item)
        self._note_items[note.id] = item
        self.scene.clearSelection()
        item.setSelected(True)
        self._update_scene_rect()
        self.model_changed.emit()
        return note

    def _make_note_item(self, note, track):
        gs = self.vm.grid_size
        pps = self.vm.pixels_per_second
        x, width, y = NoteRenderer.note_geometry(
            note,
            pps,
            gs,
            self.vm.divisions_per_octave,
            self.vm.total_divisions,
        )
        item = PianoKeyItem(
            note.pitch,
            x,
            width,
            note.velocity,
            self.vm.beat_width,
            gs,
            track_id=note.track_id,
            color=track.color if track is not None else None,
            note_id=note.id,
        )
        item.grid_divisions = self.grid_divisions
        item.grid_origin = self.vm.grid_offset_x
        item.snap_enabled = self.snap_enabled
        item.total_rows = self.vm.total_divisions
        if track is not None:
            item.set_track_style(track.color, track.name, track.muted)
        if self.vm.divisions_per_octave != 12:
            item.setY(y)
        return item

    def _add_clip_item(self, clip):
        top, height, padding = self._lane_geometry()
        item = AudioClipItem(
            clip,
            self.vm.pixels_per_second,
            top,
            height,
            snap_provider=self._clip_snap_x,
            mute_toggle=self._toggle_clip_mute,
        )
        item.set_lane(top, height, padding)
        self.scene.addItem(item)
        self._clip_items[clip.id] = item
        return item

    def add_clip(self, clip):
        item = self._add_clip_item(clip)
        self._update_scene_rect()
        self.viewport().update()
        return item

    def refresh_clip(self, clip_id):
        item = self._clip_items.get(clip_id)
        if item is None:
            return
        item.refresh_geometry(self.vm.pixels_per_second)
        self._update_scene_rect()
        self.viewport().update()

    def _toggle_clip_mute(self, clip_id):
        clip = self.model.clip_by_id(clip_id)
        if clip is None:
            return
        self.model.snapshot()
        self.model.set_clip_mute(clip_id, not clip.muted)
        item = self._clip_items.get(clip_id)
        if item is not None:
            item.update_tooltip()
            item.update()
        self.clips_changed.emit()
        self.model_changed.emit()

    def load_notes(self, midi_obj):
        self.model.snapshot()
        if midi_obj is None:
            self.model.tracks.clear()
            self.model.notes.clear()
        else:
            self.model.load_midi(midi_obj, replace=True)
        self._current_track_id = self.model.first_track_id()
        self.load_model()
        self.model_changed.emit()

    def load_model(self, keep_selection=False):
        selected_notes = set()
        selected_clips = set()
        if keep_selection:
            selected_notes = set(self._selected_note_ids())
            selected_clips = set(self._selected_clip_ids())

        self.vm.grid_offset = getattr(self.model, "grid_offset", 0.0)
        self._note_items.clear()
        self._clip_items.clear()
        self._lane_item = None
        self._press_item = None
        self.scene.clear()
        self.playhead = self._create_playhead()

        tracks = self.model.tracks
        for note in self.model.notes:
            track = tracks.get(note.track_id)
            if track is not None and not track.visible:
                continue
            item = self._make_note_item(note, track)
            self.scene.addItem(item)
            self._note_items[note.id] = item
            if note.id in selected_notes:
                item.setSelected(True)

        for clip in self.model.clips:
            item = self._add_clip_item(clip)
            if clip.id in selected_clips:
                item.setSelected(True)

        self._update_scene_rect()
        self.viewport().update()

    def _adopt_orphan_items(self):
        orphans = [
            item
            for item in self.scene.items()
            if isinstance(item, PianoKeyItem) and item.note_id is None
        ]
        if not orphans:
            return
        pps = self.vm.pixels_per_second
        track = self._ensure_current_track()
        for item in orphans:
            note = self.model.add_note(
                track.id,
                item.current_pitch(),
                item.x() / pps if pps > 0 else 0.0,
                item.rect().width() / pps if pps > 0 else MIN_NOTE_DURATION,
                item.velocity,
            )
            item.note_id = note.id
            item.track_id = track.id
            item.set_track_style(track.color, track.name, track.muted)
            self._note_items[note.id] = item

    def sync_model_from_scene(self):
        self._adopt_orphan_items()
        pps = self.vm.pixels_per_second
        if pps <= 0:
            return False
        changed = False
        notes_by_id = self.model.notes_by_id()
        twelve_tet = self.vm.divisions_per_octave == 12

        for note_id, item in self._note_items.items():
            note = notes_by_id.get(note_id)
            if note is None:
                continue
            start = max(0.0, item.x() / pps)
            duration = max(MIN_NOTE_DURATION, item.rect().width() / pps)
            pitch = item.current_pitch() if twelve_tet else note.pitch
            velocity = clamp_velocity(item.velocity)
            if (
                abs(start - note.start) > 1e-6
                or abs(duration - note.duration) > 1e-6
                or pitch != note.pitch
                or velocity != note.velocity
            ):
                note.start = start
                note.duration = duration
                note.pitch = pitch
                note.velocity = velocity
                changed = True
            if item.pitch != pitch:
                item.pitch = pitch
                item.update_tooltip()

        for clip_id, item in self._clip_items.items():
            clip = self.model.clip_by_id(clip_id)
            if clip is None:
                continue
            start = item.start_seconds()
            if abs(start - clip.start) > 1e-6:
                self.model.move_clip(clip_id, start)
                item.update_tooltip()
                changed = True

        return changed

    def _selected_note_ids(self):
        return [
            item.note_id
            for item in self.scene.selectedItems()
            if isinstance(item, PianoKeyItem) and item.note_id is not None
        ]

    def _selected_clip_ids(self):
        return [
            item.clip_id
            for item in self.scene.selectedItems()
            if isinstance(item, AudioClipItem)
        ]

    def _select_note_ids(self, note_ids):
        self.scene.clearSelection()
        for note_id in note_ids:
            item = self._note_items.get(note_id)
            if item is not None:
                item.setSelected(True)

    def _apply_edit(self, operation, keep_selection=True):
        self.sync_model_from_scene()
        self.model.snapshot()
        result = operation()
        self.load_model(keep_selection=keep_selection)
        self.model_changed.emit()
        return result

    def select_all(self):
        for item in self._note_items.values():
            item.setSelected(True)
        for item in self._clip_items.values():
            item.setSelected(True)
        return len(self._note_items) + len(self._clip_items)

    def select_track_notes(self, track_id):
        self.scene.clearSelection()
        count = 0
        for item in self._note_items.values():
            if item.track_id == track_id:
                item.setSelected(True)
                count += 1
        return count

    def delete_selected(self):
        self._adopt_orphan_items()
        note_ids = self._selected_note_ids()
        clip_ids = self._selected_clip_ids()
        if not note_ids and not clip_ids:
            return 0
        self.sync_model_from_scene()
        self.model.snapshot()
        self.model.remove_notes(note_ids)
        for clip_id in clip_ids:
            self.model.remove_clip(clip_id)
        self.load_model()
        if clip_ids:
            self.clips_changed.emit()
        self.model_changed.emit()
        return len(note_ids) + len(clip_ids)

    def copy_selected(self):
        self.sync_model_from_scene()
        note_ids = self._selected_note_ids()
        if not note_ids:
            return 0
        self._clipboard = self.model.copy_notes(note_ids)
        return len(self._clipboard)

    def has_clipboard(self):
        return bool(self._clipboard)

    def paste(self, at_time=None):
        if not self._clipboard:
            return []
        if at_time is None:
            pps = self.vm.pixels_per_second
            at_time = self.playhead.line().x1() / pps if pps > 0 else 0.0
        clipboard = list(self._clipboard)
        track_id = self.current_track_id

        def operation():
            return self.model.paste_notes(clipboard, at_time, track_id)

        created = self._apply_edit(operation, keep_selection=False)
        self._select_note_ids([n.id for n in created])
        return created

    def duplicate_selected(self):
        self.sync_model_from_scene()
        note_ids = self._selected_note_ids()
        if not note_ids:
            return []
        beat_seconds = 60.0 / self.vm.tempo

        def operation():
            return self.model.duplicate_notes(note_ids, beat_seconds)

        created = self._apply_edit(operation, keep_selection=False)
        self._select_note_ids([n.id for n in created])
        return created

    def transpose_selected(self, semitones):
        self.sync_model_from_scene()
        note_ids = self._selected_note_ids()
        if not note_ids or not semitones:
            return 0
        self._apply_edit(
            lambda: self.model.transpose_notes(note_ids, semitones)
        )
        return len(note_ids)

    def nudge_selected(self, steps):
        self.sync_model_from_scene()
        note_ids = self._selected_note_ids()
        if not note_ids or not steps:
            return 0
        seconds = self.grid_step_seconds() * steps
        self._apply_edit(lambda: self.model.shift_notes(note_ids, seconds))
        return len(note_ids)

    def _selection_or_all(self):
        self.sync_model_from_scene()
        note_ids = self._selected_note_ids()
        if note_ids:
            return note_ids
        return [note.id for note in self.model.notes]

    def quantize_selected(self, grid_divisions=None, strength=1.0):
        note_ids = self._selection_or_all()
        if not note_ids:
            return 0
        divisions = grid_divisions or self.grid_divisions
        grid_seconds = self.vm.grid_step_seconds(divisions)
        self._apply_edit(
            lambda: self.model.quantize_notes(
                note_ids, grid_seconds, strength, self.vm.grid_offset
            )
        )
        return len(note_ids)

    def humanize_selected(self, timing_sec=0.01, velocity_amt=8, seed=None):
        note_ids = self._selection_or_all()
        if not note_ids:
            return 0
        self._apply_edit(
            lambda: self.model.humanize_notes(
                note_ids, timing_sec, velocity_amt, seed
            )
        )
        return len(note_ids)

    def set_selected_velocity(self, velocity):
        self.sync_model_from_scene()
        note_ids = self._selected_note_ids()
        if not note_ids:
            return 0
        self._apply_edit(
            lambda: self.model.set_notes_velocity(note_ids, velocity)
        )
        return len(note_ids)

    def _wheel_velocity(self, item, delta_sign):
        now = time.monotonic()
        if now - self._last_velocity_wheel > VELOCITY_WHEEL_GROUP_SECONDS:
            self.sync_model_from_scene()
            self.model.snapshot()
        self._last_velocity_wheel = now
        targets = [item]
        if item.isSelected():
            targets = [
                other
                for other in self._note_items.values()
                if other.isSelected()
            ]
        note_ids = []
        for target in targets:
            target.set_velocity(
                target.velocity + delta_sign * VELOCITY_WHEEL_STEP
            )
            if target.note_id is not None:
                note_ids.append(target.note_id)
        self.model.adjust_notes_velocity(
            note_ids, delta_sign * VELOCITY_WHEEL_STEP
        )
        self.model_changed.emit()

    def undo(self):
        if not self.model.undo():
            return False
        self.load_model()
        self.clips_changed.emit()
        self.model_changed.emit()
        return True

    def redo(self):
        if not self.model.redo():
            return False
        self.load_model()
        self.clips_changed.emit()
        self.model_changed.emit()
        return True

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            try:
                sanitize_path(path)
            except InvalidFileError:
                continue
            if is_midi_path(path):
                event.acceptProposedAction()
                self.midi_dropped.emit(path)
                return
            if is_audio_path(path):
                event.acceptProposedAction()
                self.audio_dropped.emit(path)
                return

    def _is_editor_shortcut(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key in (
            Qt.Key_Delete,
            Qt.Key_Backspace,
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Left,
            Qt.Key_Right,
        ):
            return True
        if modifiers & Qt.ControlModifier:
            return key in (
                Qt.Key_A,
                Qt.Key_C,
                Qt.Key_V,
                Qt.Key_D,
                Qt.Key_Z,
                Qt.Key_Y,
            )
        return False

    def event(self, event):
        if event.type() == QEvent.ShortcutOverride and self._is_editor_shortcut(
            event
        ):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        ctrl = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)
        handled = True

        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            self.delete_selected()
        elif key == Qt.Key_Escape:
            self.scene.clearSelection()
        elif ctrl and key == Qt.Key_A:
            self.select_all()
        elif ctrl and key == Qt.Key_C:
            self.copy_selected()
        elif ctrl and key == Qt.Key_V:
            self.paste()
        elif ctrl and key == Qt.Key_D:
            self.duplicate_selected()
        elif ctrl and key == Qt.Key_Z and shift:
            self.redo()
        elif ctrl and key == Qt.Key_Z:
            self.undo()
        elif ctrl and key == Qt.Key_Y:
            self.redo()
        elif key == Qt.Key_Up:
            self.transpose_selected(12 if shift else 1)
        elif key == Qt.Key_Down:
            self.transpose_selected(-12 if shift else -1)
        elif key == Qt.Key_Left:
            self.nudge_selected(-1)
        elif key == Qt.Key_Right:
            self.nudge_selected(1)
        else:
            handled = False

        if handled:
            event.accept()
            return
        super().keyPressEvent(event)

    def set_playhead_time(self, seconds):
        x = self.vm.time_to_x(max(0.0, float(seconds)))
        self.playhead.setLine(x, 0, x, self._rows_height())

    def update_playhead(self, ms):
        if not self.is_monitoring:
            return
        x = self.vm.playhead_x(ms)
        self.playhead.setLine(x, 0, x, self._rows_height())
        if (
            x
            > self.mapToScene(self.viewport().rect()).boundingRect().right()
            - 50
        ):
            self.centerOn(
                x,
                self.mapToScene(self.viewport().rect())
                .boundingRect()
                .center()
                .y(),
            )

    def _queue_zoom(self, factor_x, factor_y):

        self._pending_zoom_x *= factor_x
        self._pending_zoom_y *= factor_y
        self._zoom_throttle.submit()

    def _apply_pending_zoom(self):
        factor_x, factor_y = self._pending_zoom_x, self._pending_zoom_y
        self._pending_zoom_x = 1.0
        self._pending_zoom_y = 1.0
        if factor_x != 1.0 or factor_y != 1.0:
            self.zoom(factor_x, factor_y)

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1.0 / zoom_factor

        if event.modifiers() & Qt.ControlModifier:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, PianoKeyItem):
                self._wheel_velocity(
                    item, 1 if event.angleDelta().y() > 0 else -1
                )
            else:
                self._queue_zoom(zoom_factor, 1.0)
            event.accept()

        elif event.modifiers() & Qt.ShiftModifier:
            self._queue_zoom(1.0, zoom_factor)
            event.accept()

        else:
            super().wheelEvent(event)
