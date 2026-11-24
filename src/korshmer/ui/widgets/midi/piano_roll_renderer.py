from functools import lru_cache

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPen, QPolygonF

from korshmer.constants import NOTE_NAMES
from korshmer.ui.viewmodels.editor_vm import note_name_for_pitch

MUTED_TRACK_ALPHA = 0.4
MIN_NOTE_WIDTH_PX = 2.0
SLIDE_MARK_ALPHA = 0.9
SLIDE_MARK_WIDTH_PX = 14.0
MIN_LABEL_ALPHA = 0.35


@lru_cache(maxsize=1)
def label_font():
    return QFont("Tahoma", 8, QFont.Bold)


@lru_cache(maxsize=1)
def selection_pen():
    pen = QPen(Qt.white, 2)
    pen.setCosmetic(True)
    return pen


@lru_cache(maxsize=8)
def label_pen(dim_alpha):
    color = QColor(255, 255, 255)
    color.setAlphaF(max(MIN_LABEL_ALPHA, dim_alpha))
    return QPen(color)


class NoteRenderer:
    @staticmethod
    def note_colors(base_color, velocity, dim_alpha):
        ratio = max(0.0, min(1.0, float(velocity) / 127.0))
        hue, _sat, _val, _alpha = base_color.getHsvF()
        fill = QColor.fromHsvF(
            max(0.0, hue), 0.45 + 0.55 * ratio, 0.55 + 0.45 * ratio
        )
        fill.setAlphaF((0.55 + 0.45 * ratio) * dim_alpha)
        pen_color = fill.darker(160)
        return fill, pen_color

    @staticmethod
    def note_tooltip(pitch, velocity, track_name, slide=None):
        name = note_name_for_pitch(pitch)
        text = f"{name} · vel {velocity}"
        if track_name:
            text = f"{text} · {track_name}"
        if slide:
            text = f"{text} · {slide.summary()}"
        return text

    @staticmethod
    def slide_marks(rect, slide):
        width = min(SLIDE_MARK_WIDTH_PX, rect.width() / 2)
        left, right = rect.left(), rect.right()
        top, bottom = rect.top(), rect.bottom()
        marks = []
        lead = slide.lead() if slide else 0
        fall = slide.fall() if slide else 0
        if lead:
            near, far = (bottom, top) if lead < 0 else (top, bottom)
            marks.append(
                QPolygonF(
                    [
                        QPointF(left, near),
                        QPointF(left + width, far),
                        QPointF(left, far),
                    ]
                )
            )
        if fall:
            near, far = (top, bottom) if fall < 0 else (bottom, top)
            marks.append(
                QPolygonF(
                    [
                        QPointF(right - width, near),
                        QPointF(right, far),
                        QPointF(right, near),
                    ]
                )
            )
        return marks

    @staticmethod
    def paint_slide(painter, rect, slide, dim_alpha):
        mark = QColor(255, 255, 255)
        mark.setAlphaF(SLIDE_MARK_ALPHA * max(0.35, dim_alpha))
        painter.setPen(Qt.NoPen)
        painter.setBrush(mark)
        for polygon in NoteRenderer.slide_marks(rect, slide):
            painter.drawPolygon(polygon)

    @staticmethod
    def note_label(y, grid_y):
        pitch = int(127 - (y // grid_y))
        if 0 <= pitch <= 127:
            return NOTE_NAMES[pitch % 12] + str(pitch // 12 - 1)
        return None

    @staticmethod
    def paint_note(painter, rect, fill_brush, pen, selected, label, dim_alpha):
        if selected:
            painter.setBrush(fill_brush.color().lighter(130))
            painter.setPen(selection_pen())
        else:
            painter.setBrush(fill_brush)
            painter.setPen(pen)

        painter.drawRoundedRect(rect, 4, 4)

        if rect.width() > 24 and label:
            painter.setPen(label_pen(dim_alpha))
            painter.setFont(label_font())
            painter.drawText(rect, Qt.AlignCenter, label)

    @staticmethod
    def note_geometry(
        note,
        pixels_per_second,
        grid_size,
        divisions_per_octave,
        total_divisions,
    ):
        x = note.start * pixels_per_second
        width = max(MIN_NOTE_WIDTH_PX, note.duration * pixels_per_second)
        ratio = divisions_per_octave / 12
        y = (total_divisions - 1 - note.pitch * ratio) * grid_size
        return x, width, y
