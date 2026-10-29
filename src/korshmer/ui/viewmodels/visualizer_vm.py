import math
import time


class VisualizerViewModel:
    def __init__(self, bars_count=32):
        self.bars_count = bars_count
        self.values = [0.0] * bars_count
        self.rms_data = None
        self.duration_sec = 0.0
        self.current_time_sec = 0.0
        self.is_playing = False
        self.volume_scale = 1.0
        self.current_energy = 0.0
        self.flash_state = 0.0

    def set_analysis(self, data, duration):
        self.rms_data = data
        self.duration_sec = duration

    def clear_analysis(self):
        self.rms_data = None
        self.duration_sec = 0.0

    def set_position(self, ms):
        self.current_time_sec = ms / 1000.0

    def tick(self):
        base_energy = 0.0

        if (
            self.is_playing
            and self.rms_data is not None
            and self.duration_sec > 0
        ):
            total_frames = len(self.rms_data)
            idx = int(
                (self.current_time_sec / self.duration_sec) * total_frames
            )
            if 0 <= idx < total_frames:
                raw_val = self.rms_data[idx] * self.volume_scale
                base_energy = (raw_val**3) * 3.0

        self.current_energy = self.current_energy * 0.4 + base_energy * 0.6

        self.flash_state = min(1.0, self.current_energy * 0.5)

        speed_mult = self.current_energy / 5
        t = time.time() * speed_mult

        for i in range(self.bars_count):
            if self.current_energy > 0.01:
                noise = math.sin(i * 0.9 + t) * math.cos(i * 0.4 - t * 0.5)
                dist = 1.0 - abs(
                    (i - self.bars_count / 2) / (self.bars_count / 2)
                )
                dist = dist**0.5
                target = min(
                    1.0, self.current_energy * (0.5 + 0.5 * noise) * dist * 1.5
                )
            else:
                target = 0.0
            self.values[i] += (target - self.values[i]) * 0.6

    def bar_color_rgb(self):
        energy = min(1.0, self.current_energy)
        c_low = (26, 115, 232)
        c_mid = (128, 0, 128)
        c_high = (255, 0, 0)

        if energy < 0.5:
            ratio = energy * 2.0
            a, b = c_low, c_mid
        else:
            ratio = (energy - 0.5) * 2.0
            a, b = c_mid, c_high

        r = int(a[0] + ratio * (b[0] - a[0]))
        g = int(a[1] + ratio * (b[1] - a[1]))
        bl = int(a[2] + ratio * (b[2] - a[2]))
        return r, g, bl

    def flash_alpha(self):
        alpha = self.flash_state
        if alpha < 0.3:
            alpha *= 0.4
        elif alpha < 0.5:
            alpha *= 0.8
        elif alpha < 0.7:
            alpha *= 1.2
        else:
            alpha = (alpha + 0.05) * 1.2
        return min(1.0, alpha)
