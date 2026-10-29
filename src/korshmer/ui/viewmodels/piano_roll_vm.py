import logging
import math

from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.utils.tempo import MAX_BPM, MIN_BPM

_logger = logging.getLogger(__name__)


class NoteData:
    __slots__ = ("pitch", "x", "width", "velocity", "grid_x", "grid_y")

    def __init__(self, pitch, x, width, velocity, grid_x, grid_y=20):
        self.pitch = pitch
        self.x = x
        self.width = width
        self.velocity = velocity
        self.grid_x = grid_x
        self.grid_y = grid_y

    @property
    def y(self):
        return (127 - self.pitch) * self.grid_y

    def snap_position(self, new_x, new_y):
        x = round(new_x / self.grid_x) * self.grid_x
        y = round(new_y / self.grid_y) * self.grid_y
        max_y = 127 * self.grid_y
        y = max(0, min(max_y, y))
        x = max(0, x)
        return x, y


class PianoRollViewModel:
    def __init__(self):
        self.grid_size = 20
        self.pixels_per_second = 100.0
        self.tempo = 120.0
        self.divisions_per_octave = 12
        self.root_note = "C"
        self._root_note_idx = 0
        self.scale_intervals = SCALES["Chromatic"]
        self.grid_offset = 0.0

    @property
    def grid_offset_x(self):
        return self.grid_offset * self.pixels_per_second

    @property
    def total_divisions(self):
        return 128 * (self.divisions_per_octave // 12)

    @property
    def beat_width(self):
        return (60.0 / self.tempo) * self.pixels_per_second

    def set_scale(self, root_name, scale_name):
        if root_name == "Auto":
            root_name = "C"
        try:
            self._root_note_idx = NOTE_NAMES.index(root_name)
            self.root_note = root_name
        except ValueError:
            self._root_note_idx = 0
            self.root_note = "C"

        if scale_name == "Auto (Detect)":
            scale_name = "Chromatic"
        self.scale_intervals = SCALES.get(scale_name, SCALES["Chromatic"])

    def set_divisions(self, divisions):
        if divisions not in (12, 24, 48, 96):
            _logger.warning(f"Unsupported divisions: {divisions}. Using 12.")
            return 12
        old = self.divisions_per_octave
        self.divisions_per_octave = divisions
        return old

    def compute_tempo_change(self, new_bpm):
        try:
            new_bpm = float(new_bpm)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(new_bpm) or new_bpm <= 0:
            return None

        new_bpm = max(MIN_BPM, min(MAX_BPM, new_bpm))
        old_bpm = self.tempo
        ratio = old_bpm / new_bpm

        self.tempo = new_bpm

        if abs(ratio - 1.0) < 1e-9:
            return None

        self.grid_offset *= ratio
        old_beat_width = (60.0 / old_bpm) * self.pixels_per_second
        new_beat_width = (60.0 / new_bpm) * self.pixels_per_second
        return old_beat_width, new_beat_width

    def pitch_info(self, pitch):
        relative = (pitch - self._root_note_idx) % 12
        return {
            "in_scale": relative in self.scale_intervals,
            "is_root": relative == 0,
            "name": f"{NOTE_NAMES[pitch % 12]}{(pitch // 12) - 1}",
        }

    def time_to_x(self, seconds):
        return seconds * self.pixels_per_second

    def x_to_time(self, x):
        return x / self.pixels_per_second if self.pixels_per_second > 0 else 0

    def snap_to_beat(self, x):
        bw = self.beat_width
        origin = self.grid_offset_x
        return round((x - origin) / bw) * bw + origin

    def grid_step(self, divisions=1):
        return self.beat_width / max(1, int(divisions))

    def grid_step_seconds(self, divisions=1):
        return (60.0 / self.tempo) / max(1, int(divisions))

    def snap_to_grid(self, x, divisions=1):
        step = self.grid_step(divisions)
        if step <= 0:
            return max(0.0, x)
        origin = self.grid_offset_x
        snapped = round((x - origin) / step) * step + origin
        if snapped < 0:
            snapped += math.ceil(-snapped / step - 1e-9) * step
        return max(0.0, snapped)

    def pitch_from_y(self, y):
        return int(127 - (y // self.grid_size))

    def y_from_pitch(self, pitch):
        return (127 - pitch) * self.grid_size

    def playhead_x(self, ms):
        return (ms / 1000.0) * self.pixels_per_second

    def note_from_midi(self, midi_note):
        return NoteData(
            pitch=midi_note.pitch,
            x=midi_note.start * self.pixels_per_second,
            width=(midi_note.end - midi_note.start) * self.pixels_per_second,
            velocity=midi_note.velocity,
            grid_x=self.beat_width,
            grid_y=self.grid_size,
        )
