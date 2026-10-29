import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPen


class GridPainter:
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

        draw_left = max(0, viewport_rect.left())
        draw_right = viewport_rect.right()
        draw_top = rect.top()
        draw_bottom = rect.bottom()

        pen_bar = QPen(QColor(0, 0, 0, 120), 1)
        pen_beat = QPen(QColor(0, 0, 0, 40), 1)
        pen_grid = QPen(QColor(0, 0, 0, 18), 1)

        font_time = QFont("Monospace", 8, QFont.Bold)

        beat_width = vm.beat_width
        origin = getattr(vm, "grid_offset_x", 0.0)
        if beat_width > 5:
            start_beat_idx = math.floor((draw_left - origin) / beat_width)
            current_x = origin + start_beat_idx * beat_width
            beat_idx = start_beat_idx

            while current_x < draw_right:
                if current_x >= draw_left:
                    if beat_idx % 4 == 0:
                        bt = beat_idx // 4 + 1
                        if current_x > 0.5:
                            painter.setPen(pen_bar)
                            painter.drawLine(
                                int(current_x),
                                int(draw_top),
                                int(current_x),
                                int(draw_bottom),
                            )

                            if scale_x > 0.3:
                                painter.save()
                                painter.scale(1 / scale_x, 1 / scale_y)
                                painter.setPen(QColor(0, 0, 0, 200))
                                txt_x = (current_x * scale_x) + 5
                                txt_y = (
                                    (viewport_rect.top() * scale_y)
                                    + 30
                                    + lane_px
                                )
                                painter.drawText(
                                    int(txt_x), int(txt_y), str(bt)
                                )
                                painter.restore()
                    else:
                        if scale_x * beat_width > 10:
                            painter.setPen(pen_beat)
                            painter.drawLine(
                                int(current_x),
                                int(draw_top),
                                int(current_x),
                                int(draw_bottom),
                            )

                current_x += beat_width
                beat_idx += 1

            step = beat_width / max(1, grid_divisions)
            if grid_divisions > 1 and scale_x * step > 6:
                painter.setPen(pen_grid)
                grid_x = origin + math.floor((draw_left - origin) / step) * step
                while grid_x < draw_right:
                    if grid_x >= draw_left:
                        painter.drawLine(
                            int(grid_x),
                            int(max(draw_top, 0)),
                            int(grid_x),
                            int(draw_bottom),
                        )
                    grid_x += step

        sec_start = math.floor(draw_left / vm.pixels_per_second)
        curr_sec_x = sec_start * vm.pixels_per_second

        while curr_sec_x < draw_right:
            if curr_sec_x >= draw_left:
                seconds = curr_sec_x / vm.pixels_per_second
                m, s = divmod(int(seconds), 60)
                time_str = f"{m:02}:{s:02}"

                painter.save()
                painter.translate(
                    curr_sec_x + (2 / scale_x),
                    viewport_rect.top() + ((10 + lane_px) / scale_y),
                )
                painter.scale(1.0 / scale_x, 1.0 / scale_y)

                if (s % 5 == 0 or scale_x > 0.8) and time_str != "00:00":
                    painter.setPen(Qt.black)
                    painter.setFont(font_time)
                    painter.drawText(0, 0, time_str)
                painter.restore()
            curr_sec_x += vm.pixels_per_second

        gs = vm.grid_size
        start_y = int(draw_top) - (int(draw_top) % gs)
        for y in range(start_y, int(draw_bottom), gs):
            if y < 0:
                continue
            pitch = 127 - (y // gs)
            if pitch < 0:
                continue

            info = vm.pitch_info(pitch)

            if info["is_root"]:
                painter.fillRect(
                    int(draw_left),
                    y,
                    int(draw_right - draw_left),
                    gs,
                    QColor(0, 0, 255, 40),
                )
            elif info["in_scale"]:
                painter.fillRect(
                    int(draw_left),
                    y,
                    int(draw_right - draw_left),
                    gs,
                    QColor(255, 255, 255, 40),
                )
            else:
                painter.fillRect(
                    int(draw_left),
                    y,
                    int(draw_right - draw_left),
                    gs,
                    QColor(0, 0, 0, 40),
                )

            painter.setPen(QColor(0, 0, 0, 60))
            painter.drawLine(int(draw_left), y, int(draw_right), y)

            if (
                (info["is_root"] or info["in_scale"] or scale_y > 0.8)
                and (draw_left < viewport_rect.left() + (50 / scale_x))
                and (scale_y > 0.8 or pitch % 12 == 0)
            ):
                painter.save()
                painter.translate(
                    viewport_rect.left() + (5 / scale_x),
                    y + gs - (5 / scale_y),
                )
                painter.scale(1.0 / scale_x, 1.0 / scale_y)

                if info["is_root"]:
                    painter.setPen(Qt.black)
                    painter.setFont(QFont("Tahoma", 9, QFont.Bold))
                elif info["in_scale"]:
                    painter.setPen(Qt.black)
                    painter.setFont(QFont("Tahoma", 9, QFont.Bold))
                else:
                    painter.setPen(QColor(0, 0, 0, 100))
                    painter.setFont(QFont("Tahoma", 8))

                painter.drawText(0, 0, info["name"])
                painter.restore()
