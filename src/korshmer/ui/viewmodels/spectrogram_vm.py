import numpy as np

from korshmer.utils.filesystem import is_midi_path


class SpectrogramViewModel:
    MEL_MAX = 2595 * np.log10(1 + 8000.0 / 700.0)

    def __init__(self):
        self.duration = 0.0
        self.loading = False
        self.has_image = False

    def set_loading(self, audio_path):
        if is_midi_path(audio_path):
            self.has_image = False
            self.loading = False
            return False

        self.loading = True
        self.has_image = False
        return True

    def on_ready(self, has_image):
        self.loading = False
        self.has_image = has_image

    def hz_to_y(self, freq, height):
        if freq <= 0:
            return height
        mel_f = 2595 * np.log10(1 + freq / 700.0)
        norm = mel_f / self.MEL_MAX
        return height * (1 - norm)

    def y_to_hz(self, y, height):
        norm = 1 - (y / height)
        mel_val = norm * self.MEL_MAX
        return 700 * (10 ** (mel_val / 2595.0) - 1)

    def hover_info(self, x, y, width, height):
        if not self.has_image:
            return None

        if not (0 <= y <= height and 0 <= x <= width):
            return None

        freq = self.y_to_hz(y, height)
        time_sec = (x / width) * self.duration if self.duration > 0 else 0
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)

        return {
            "freq": freq,
            "time_str": f"{minutes:02}:{seconds:02}",
            "freq_int": int(freq),
        }
