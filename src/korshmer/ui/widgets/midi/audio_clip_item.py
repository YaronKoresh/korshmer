import math
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem

CLIP_LANE_HEIGHT = 64
CLIP_LANE_PADDING = 4
MIN_CLIP_WIDTH = 40.0
EDGE_GRAB_PIXELS = 8.0
TRIM_HINT = "Drag the left or right edge to trim the clip."
CLIP_CORNER_RADIUS = 6


@lru_cache(maxsize=1)
def caption_font():
    return QFont("Tahoma", 8, QFont.Bold)


class AudioClipItem(QGraphicsRectItem):
    def __init__(
        self,
        clip,
        pixels_per_second,
        lane_top,
        lane_height=CLIP_LANE_HEIGHT,
        snap_provider=None,
        mute_toggle=None,
        trim_request=None,
    ):
        self.trim_request = trim_request
        self._trim_edge = None
        self.clip = clip
        self.clip_id = clip.id
        self.pixels_per_second = float(pixels_per_second)
        self.lane_top = float(lane_top)
        self.lane_height = float(lane_height)
        self.lane_padding = float(CLIP_LANE_PADDING)
        self.snap_provider = snap_provider
        self.mute_toggle = mute_toggle
        self._waveform_shape = None
        super().__init__(
            0,
            0,
            self._width_for_duration(),
            self.lane_height - 2 * self.lane_padding,
        )
        self.setPos(self._x_for_start(), self.lane_top + self.lane_padding)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setZValue(5)
        self.setCursor(Qt.OpenHandCursor)
        self.update_tooltip()

    def _width_for_duration(self):
        return max(MIN_CLIP_WIDTH, self.clip.duration * self.pixels_per_second)

    def _x_for_start(self):
        return max(0.0, self.clip.start * self.pixels_per_second)

    def refresh_geometry(self, pixels_per_second=None):
        if pixels_per_second:
            self.pixels_per_second = float(pixels_per_second)
        self.prepareGeometryChange()
        self.setRect(0, 0, self._width_for_duration(), self.rect().height())
        self.setPos(self._x_for_start(), self.lane_top + self.lane_padding)
        self.update_tooltip()
        self.update()

    def set_lane(self, lane_top, lane_height, padding=CLIP_LANE_PADDING):
        self.lane_top = float(lane_top)
        self.lane_height = float(lane_height)
        self.lane_padding = float(padding)
        self.prepareGeometryChange()
        self.setRect(
            0,
            0,
            self.rect().width(),
            max(4.0, self.lane_height - 2 * self.lane_padding),
        )
        self.setPos(self.x(), self.lane_top + self.lane_padding)
        self.update()

    def update_tooltip(self):
        state = "muted" if self.clip.muted else f"gain {self.clip.gain:.2f}x"
        trimmed = ""
        if self.clip.is_trimmed():
            trimmed = f" of {self.clip.length:.2f}s"
        self.setToolTip(
            f"{self.clip.name} ({self.clip.kind})\n"
            f"start {self.clip.start:.2f}s · "
            f"{self.clip.duration:.2f}s{trimmed} · {state}\n{TRIM_HINT}"
        )

    def _edge_reach(self):
        views = self.scene().views() if self.scene() else []
        scale = views[0].transform().m11() if views else 1.0
        reach = EDGE_GRAB_PIXELS / (scale or 1.0)
        return min(reach, self.rect().width() / 3.0)

    def edge_at(self, x):
        reach = self._edge_reach()
        if x <= reach:
            return "start"
        if x >= self.rect().width() - reach:
            return "end"
        return None

    def hoverMoveEvent(self, event):
        edge = self.edge_at(event.pos().x()) if self.trim_request else None
        self.setCursor(Qt.SizeHorCursor if edge else Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        edge = None
        if event.button() == Qt.LeftButton and self.trim_request is not None:
            edge = self.edge_at(event.pos().x())
        if edge is None:
            super().mousePressEvent(event)
            return
        self._trim_edge = edge
        self.setSelected(True)
        self.trim_request(self.clip_id, None, None)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._trim_edge is None:
            super().mouseMoveEvent(event)
            return
        x = max(0.0, event.scenePos().x())
        if self.snap_provider is not None:
            x = max(0.0, self.snap_provider(x))
        self.trim_edge_to(self._trim_edge, x / self.pixels_per_second)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._trim_edge is None:
            super().mouseReleaseEvent(event)
            return
        self._trim_edge = None
        event.accept()

    def trim_edge_to(self, edge, seconds):
        start, end = self.clip.start, self.clip.end
        if edge == "start":
            start = seconds
        else:
            end = seconds
        self.trim_request(self.clip_id, start, end)
        self.refresh_geometry()

    def start_seconds(self):
        if self.pixels_per_second <= 0:
            return 0.0
        return max(0.0, self.x() / self.pixels_per_second)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            x = max(0.0, value.x())
            if self.snap_provider is not None:
                x = max(0.0, self.snap_provider(x))
            return QPointF(x, self.lane_top + self.lane_padding)
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        on_edge = (
            self.trim_request is not None
            and self.edge_at(event.pos().x()) is not None
        )
        if event.button() == Qt.LeftButton and on_edge:
            self.mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton and self.mute_toggle is not None:
            self.mute_toggle(self.clip_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _fill_color(self):
        color = QColor(self.clip.color)
        if not color.isValid():
            color = QColor("#EA4335")
        color.setAlphaF(0.35 if self.clip.muted else 0.85)
        return color

    def paint(self, painter, _option, _widget=None):
        rect = self.rect()
        fill = self._fill_color()
        painter.setBrush(QBrush(fill))
        outline = QPen(Qt.white if self.isSelected() else fill.darker(150), 2)
        outline.setCosmetic(True)
        painter.setPen(outline)
        painter.drawRoundedRect(rect, CLIP_CORNER_RADIUS, CLIP_CORNER_RADIUS)
        self._paint_waveform(painter, rect)
        if self.clip.muted:
            self._paint_hatching(painter, rect)
        self._paint_label(painter, rect)

    def _visible_peaks(self):
        points = self.clip.waveform
        if points and self.clip.length > 0:
            first = int(len(points) * self.clip.offset / self.clip.length)
            last = int(
                math.ceil(
                    len(points)
                    * (self.clip.offset + self.clip.duration)
                    / self.clip.length
                )
            )
            points = points[first : max(first + 2, last)]
        return points

    def _build_waveform(self, rect):
        points = self._visible_peaks()
        if not points:
            return None
        mid_y = rect.height() / 2.0
        half = max(1.0, mid_y - 4)
        step = rect.width() / max(1, len(points) - 1)
        upper = []
        lower = []
        for index, value in enumerate(points):
            amplitude = max(0.0, min(1.0, float(value))) * half
            x = index * step
            upper.append(QPointF(x, mid_y - amplitude))
            lower.append(QPointF(x, mid_y + amplitude))
        lower.reverse()
        return QPolygonF(upper + lower)

    def waveform_polygon(self):
        rect = self.rect()
        clip = self.clip
        peaks = clip.waveform
        key = (
            len(peaks),
            clip.offset,
            clip.duration,
            clip.length,
            rect.width(),
            rect.height(),
        )
        cached = self._waveform_shape
        if cached is None or cached[0] is not peaks or cached[1] != key:
            cached = (peaks, key, self._build_waveform(rect))
            self._waveform_shape = cached
        return cached[2]

    def _paint_waveform(self, painter, rect):
        if rect.width() < 8:
            return
        polygon = self.waveform_polygon()
        if polygon is None:
            return
        painter.setPen(Qt.NoPen)
        painter.setBrush(
            QBrush(QColor(255, 255, 255, 60 if self.clip.muted else 150))
        )
        painter.drawPolygon(polygon)

    def _paint_hatching(self, painter, rect):
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(0, 0, 0, 90), 1))
        height = rect.height()
        x = -height
        while x < rect.width():
            painter.drawLine(QPointF(x, height), QPointF(x + height, 0))
            x += 8
        painter.restore()

    def _paint_label(self, painter, rect):
        if rect.width() < 24:
            return
        painter.setPen(QColor(255, 255, 255, 230))
        painter.setFont(caption_font())
        label = self.clip.name
        if self.clip.muted:
            label = f"{label} (muted)"
        text_rect = QRectF(6, 2, max(0.0, rect.width() - 12), 14)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label)


class ClipLaneBackgroundItem(QGraphicsRectItem):
    def __init__(self):
        super().__init__(0, 0, 1, CLIP_LANE_HEIGHT)
        self.view_scale_x = 1.0
        self.view_scale_y = 1.0
        self.caption_right = None
        self.setZValue(4)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setPen(QPen(Qt.NoPen))

    def set_geometry(
        self, left, top, width, height, scale_x, scale_y, caption_right=None
    ):
        self.view_scale_x = float(scale_x) or 1.0
        self.view_scale_y = float(scale_y) or 1.0
        self.caption_right = caption_right
        self.prepareGeometryChange()
        self.setRect(0, 0, max(1.0, width), max(1.0, height))
        self.setPos(left, top)

    def paint(self, painter, _option, _widget=None):
        rect = self.rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(238, 241, 245, 236)))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(0, 0, 0, 90), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        font = caption_font()
        caption = "AUDIO CLIPS"
        text_width = QFontMetrics(font).horizontalAdvance(caption)
        if self.caption_right is None:
            anchor_x = rect.right() - (text_width + 8) / self.view_scale_x
        else:
            anchor_x = (
                self.caption_right
                - self.x()
                - (text_width + 8) / self.view_scale_x
            )
        painter.save()
        painter.translate(anchor_x, rect.top() + 13 / self.view_scale_y)
        painter.scale(1.0 / self.view_scale_x, 1.0 / self.view_scale_y)
        painter.setPen(QColor(60, 64, 67, 200))
        painter.setFont(font)
        painter.drawText(0, 0, caption)
        painter.restore()
