import math
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPen

DEFAULT_BEATS_PER_BAR = 4
EDGE_PAD_PX = 3.0
TOP_LABEL_REACH_PX = 96.0
PITCH_LABEL_REACH_PX = 72.0
PITCH_LABEL_HEIGHT_PX = 28.0
MIN_TIME_LABEL_GAP_PX = 50.0
TIME_LABEL_STEPS = (5, 10, 30, 60, 300, 600, 1800, 3600)


@lru_cache(maxsize=1)
def _pens():
    return {
        "bar": QPen(QColor(0, 0, 0, 120), 1),
        "beat": QPen(QColor(0, 0, 0, 40), 1),
        "grid": QPen(QColor(0, 0, 0, 18), 1),
        "row": QPen(QColor(0, 0, 0, 60), 1),
        "number": QPen(QColor(0, 0, 0, 200), 1),
        "text": QPen(QColor(Qt.black), 1),
        "faint": QPen(QColor(0, 0, 0, 100), 1),
    }


@lru_cache(maxsize=1)
def _row_colors():
    return {
        "root": QColor(0, 0, 255, 40),
        "scale": QColor(255, 255, 255, 40),
        "other": QColor(0, 0, 0, 40),
    }


@lru_cache(maxsize=1)
def _fonts():
    return {
        "time": QFont("Monospace", 8, QFont.Bold),
        "scale": QFont("Tahoma", 9, QFont.Bold),
        "other": QFont("Tahoma", 8),
    }


class GridPainter:
    @staticmethod
    def beats_per_bar(vm):
        return max(1, int(getattr(vm, "beats_per_bar", DEFAULT_BEATS_PER_BAR)))

    @staticmethod
    def time_label_step(second_px):
        if second_px > 80:
            return 1
        for step in TIME_LABEL_STEPS:
            if step * second_px >= MIN_TIME_LABEL_GAP_PX:
                return step
        return TIME_LABEL_STEPS[-1]

    @staticmethod
    def beat_lines(vm, left, right, grid_divisions, scale_x):
        beat_width = vm.beat_width
        if beat_width <= 5:
            return []
        mapped = vm.mapped_lines(left, right, grid_divisions)
        if mapped is not None:
            return mapped
        origin = vm.grid_offset_x
        steps = max(1, int(grid_divisions))
        fine = steps > 1 and scale_x * beat_width / steps > 6
        if not fine:
            steps = 1
        step = beat_width / steps
        first = math.floor((left - origin) / step)
        last = math.ceil((right - origin) / step)
        lines = []
        for index in range(first, last + 1):
            x = origin + index * step
            if not left <= x < right:
                continue
            if index % steps == 0:
                lines.append((x, index // steps))
            if fine:
                lines.append((x, None))
        return lines

    @staticmethod
    def draw_background(
        painter,
        rect,
        view_transform,
        viewport_rect,
        vm,
        grid_divisions,
        lane_px,
    ):
        scale_x = view_transform.m11() if view_transform.m11() != 0 else 1
        scale_y = view_transform.m22() if view_transform.m22() != 0 else 1
        pad_x = EDGE_PAD_PX / scale_x + 1.5
        pad_y = EDGE_PAD_PX / scale_y + 1.0

        view_left = max(0.0, viewport_rect.left())
        view_right = viewport_rect.right()
        line_top = math.floor(viewport_rect.top()) - 1
        line_bottom = math.ceil(viewport_rect.bottom()) + 1
        left = max(
            view_left, rect.left() - pad_x - TOP_LABEL_REACH_PX / scale_x
        )
        right = min(view_right, rect.right() + pad_x)

        pens = _pens()
        fonts = _fonts()
        texts = []
        per_bar = GridPainter.beats_per_bar(vm)
        beat_width = vm.beat_width
        fine = grid_divisions > 1 and (
            scale_x * beat_width / max(1, grid_divisions) > 6
        )
        beats_shown = scale_x * beat_width > 10
        number_y = int(viewport_rect.top() * scale_y + 30 + lane_px)

        pen = None
        for line_x, beat in GridPainter.beat_lines(
            vm, left, right, grid_divisions, scale_x
        ):
            bar = beat is not None and beat % per_bar == 0
            top = line_top
            if beat is None:
                if not fine:
                    continue
                wanted = "grid"
                top = max(line_top, 0)
            elif bar:
                if line_x <= 0.5:
                    continue
                wanted = "bar"
            elif beats_shown:
                wanted = "beat"
            else:
                continue
            if wanted != pen:
                pen = wanted
                painter.setPen(pens[pen])
            painter.drawLine(int(line_x), top, int(line_x), line_bottom)
            if bar and scale_x > 0.3:
                texts.append(
                    (
                        "number",
                        None,
                        float(int(line_x * scale_x + 5)),
                        float(number_y),
                        str(beat // per_bar + 1),
                    )
                )

        second_width = vm.pixels_per_second
        label_step = GridPainter.time_label_step(scale_x * second_width)
        label_y = viewport_rect.top() * scale_y + 10 + lane_px
        first_second = max(1, math.ceil(left / second_width))
        for second in range(first_second, math.ceil(right / second_width)):
            if second % label_step:
                continue
            minutes, seconds = divmod(second, 60)
            texts.append(
                (
                    "text",
                    "time",
                    second * second_width * scale_x + 2,
                    label_y,
                    f"{minutes:02}:{seconds:02}",
                )
            )

        labels_on_screen = view_left < viewport_rect.left() + (50 / scale_x)
        labels_exposed = labels_on_screen and (
            rect.left() - pad_x
            < viewport_rect.left() + PITCH_LABEL_REACH_PX / scale_x
        )
        if labels_exposed:
            pad_y += PITCH_LABEL_HEIGHT_PX / scale_y

        colors = _row_colors()
        gs = vm.grid_size
        row_left = math.floor(view_left)
        row_right = math.ceil(view_right)
        row_span = QRectF(row_left, 0.0, row_right - row_left, gs)
        label_x = viewport_rect.left() * scale_x + 5
        first_row = max(0, math.floor((rect.top() - pad_y) / gs))
        last_row = min(127, math.floor((rect.bottom() + pad_y) / gs))
        if first_row <= last_row:
            painter.setPen(pens["row"])
        for row in range(first_row, last_row + 1):
            y = row * gs
            pitch = 127 - row
            info = vm.pitch_info(pitch)
            if info["is_root"]:
                kind = "root"
            elif info["in_scale"]:
                kind = "scale"
            else:
                kind = "other"

            row_span.moveTop(y)
            painter.fillRect(row_span, colors[kind])
            painter.drawLine(row_left, y, row_right, y)

            if (
                labels_exposed
                and (kind != "other" or scale_y > 0.8)
                and (scale_y > 0.8 or pitch % 12 == 0)
            ):
                plain = kind == "other"
                texts.append(
                    (
                        "faint" if plain else "text",
                        "other" if plain else "scale",
                        label_x,
                        (y + gs) * scale_y - 5,
                        info["name"],
                    )
                )

        if not texts:
            return
        painter.save()
        painter.scale(1.0 / scale_x, 1.0 / scale_y)
        pen = font = None
        for wanted_pen, wanted_font, x, y, text in texts:
            if wanted_pen != pen:
                pen = wanted_pen
                painter.setPen(pens[pen])
            if wanted_font is not None and wanted_font != font:
                font = wanted_font
                painter.setFont(fonts[font])
            painter.drawText(QPointF(x, y), text)
        painter.restore()
