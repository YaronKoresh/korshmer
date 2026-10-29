from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPen

from korshmer.constants import NOTE_NAMES
from korshmer.ui.viewmodels.editor_vm import note_name_for_pitch

MUTED_TRACK_ALPHA = 0.4
MIN_NOTE_WIDTH_PX = 2.0


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
    def note_tooltip(pitch, velocity, track_name):
        name = note_name_for_pitch(pitch)
        text = f"{name} · vel {velocity}"
        if track_name:
            text = f"{text} · {track_name}"
        return text

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
            painter.setPen(QPen(Qt.white, 2))
        else:
            painter.setBrush(fill_brush)
            painter.setPen(pen)

        painter.drawRoundedRect(rect, 4, 4)

        if rect.width() > 24 and label:
            text_color = QColor(255, 255, 255)
            text_color.setAlphaF(max(0.35, dim_alpha))
            painter.setPen(text_color)
            painter.setFont(QFont("Tahoma", 8, QFont.Bold))
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
