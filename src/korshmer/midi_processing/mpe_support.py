import math
import numbers
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .note_validation import validate_midi_note

MAX_CURVE_POINTS = 100_000


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _point_count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if not 1 <= value <= MAX_CURVE_POINTS:
        raise ValueError(
            f"{name} must be between 1 and {MAX_CURVE_POINTS}, got {value!r}"
        )
    return int(value)


class MPEZone(Enum):
    LOWER = "lower"
    UPPER = "upper"


@dataclass
class MPENote:
    pitch: int
    start_time: float
    duration: float
    velocity: int
    channel: int
    pressure: list[tuple[float, int]]
    pitch_bend: list[tuple[float, float]]
    timbre: list[tuple[float, int]]
    slide: list[tuple[float, int]]

    def __post_init__(self):
        self.pitch = validate_midi_note(self.pitch)

    def get_pressure_at_time(self, time: float) -> int:
        if not self.pressure:
            return 0

        for i in range(len(self.pressure) - 1):
            t1, p1 = self.pressure[i]
            t2, p2 = self.pressure[i + 1]
            if t1 <= time < t2:
                factor = (time - t1) / (t2 - t1) if t2 > t1 else 0
                return int(p1 + factor * (p2 - p1))

        return self.pressure[-1][1]

    def get_pitch_bend_at_time(self, time: float) -> float:
        if not self.pitch_bend:
            return 0.0

        for i in range(len(self.pitch_bend) - 1):
            t1, b1 = self.pitch_bend[i]
            t2, b2 = self.pitch_bend[i + 1]
            if t1 <= time < t2:
                factor = (time - t1) / (t2 - t1) if t2 > t1 else 0
                return b1 + factor * (b2 - b1)

        return self.pitch_bend[-1][1]


class MPEChannelExhaustedError(RuntimeError):
    pass


@dataclass
class MPEConfiguration:
    lower_zone_channels: list[int]
    upper_zone_channels: list[int]
    pitch_bend_range: int = 48
    master_channel_lower: int = 0
    master_channel_upper: int = 15

    def __post_init__(self):
        masters = {self.master_channel_lower, self.master_channel_upper}
        self.lower_zone_channels = [
            ch for ch in self.lower_zone_channels if ch not in masters
        ]
        self.upper_zone_channels = [
            ch for ch in self.upper_zone_channels if ch not in masters
        ]

    def get_zone_for_channel(self, channel: int) -> MPEZone | None:
        if channel in self.lower_zone_channels:
            return MPEZone.LOWER
        elif channel in self.upper_zone_channels:
            return MPEZone.UPPER
        return None


class MPEProcessor:
    PITCH_BEND_CENTER = 8192
    PITCH_BEND_MAX = 16383

    def __init__(self, config: MPEConfiguration | None = None):
        if config is None:
            self.config = MPEConfiguration(
                lower_zone_channels=list(range(1, 8)),
                upper_zone_channels=list(range(9, 15)),
                pitch_bend_range=48,
            )
        else:
            self.config = config

        self.active_notes: dict[int, MPENote] = {}
        self.channel_allocator = MPEChannelAllocator(self.config)
        self.zone_setup_messages = self.build_zone_setup_messages()

    @classmethod
    def from_controller_mapper(
        cls, mapper: "MPEControllerMapper"
    ) -> "MPEProcessor":
        config = MPEConfiguration(
            lower_zone_channels=list(range(1, 8)),
            upper_zone_channels=list(range(9, 15)),
            pitch_bend_range=mapper.get_pitch_bend_range(),
        )
        return cls(config)

    def build_zone_setup_messages(self) -> list[dict]:
        messages = []
        zones = (
            (self.config.master_channel_lower, self.config.lower_zone_channels),
            (self.config.master_channel_upper, self.config.upper_zone_channels),
        )
        for master, members in zones:
            if not members:
                continue
            for control, value in (
                (101, 0),
                (100, 6),
                (6, len(members)),
            ):
                messages.append(
                    {
                        "type": "control_change",
                        "channel": master,
                        "control": control,
                        "value": value,
                        "time": 0.0,
                    }
                )
        return messages

    def create_mpe_note(
        self,
        pitch: int,
        start_time: float,
        duration: float,
        velocity: int,
        zone: MPEZone = MPEZone.LOWER,
    ) -> MPENote:
        channel = self.channel_allocator.allocate_channel(zone)

        note = MPENote(
            pitch=pitch,
            start_time=start_time,
            duration=duration,
            velocity=velocity,
            channel=channel,
            pressure=[],
            pitch_bend=[],
            timbre=[],
            slide=[],
        )

        self.active_notes[id(note)] = note
        return note

    def add_pressure_automation(
        self, note: MPENote, time_values: list[tuple[float, int]]
    ):
        note.pressure.extend(time_values)
        note.pressure.sort(key=lambda x: x[0])

    def add_pitch_bend_automation(
        self, note: MPENote, time_values: list[tuple[float, float]]
    ):
        note.pitch_bend.extend(time_values)
        note.pitch_bend.sort(key=lambda x: x[0])

    def add_timbre_automation(
        self, note: MPENote, time_values: list[tuple[float, int]]
    ):
        note.timbre.extend(time_values)
        note.timbre.sort(key=lambda x: x[0])

    def add_slide_automation(
        self, note: MPENote, time_values: list[tuple[float, int]]
    ):
        note.slide.extend(time_values)
        note.slide.sort(key=lambda x: x[0])

    def convert_to_midi(self, note: MPENote) -> list[dict]:
        messages = []

        messages.append(
            {
                "type": "note_on",
                "channel": note.channel,
                "pitch": note.pitch,
                "velocity": note.velocity,
                "time": note.start_time,
            }
        )

        for time, pressure in note.pressure:
            messages.append(
                {
                    "type": "polyphonic_aftertouch",
                    "channel": note.channel,
                    "pitch": note.pitch,
                    "pressure": pressure,
                    "time": time,
                }
            )

        for time, bend in note.pitch_bend:
            normalized_bend = bend / float(self.config.pitch_bend_range)
            normalized_bend = max(-1.0, min(1.0, normalized_bend))
            bend_value = int(
                self.PITCH_BEND_CENTER
                + normalized_bend * self.PITCH_BEND_CENTER
            )
            bend_value = max(0, min(self.PITCH_BEND_MAX, bend_value))
            messages.append(
                {
                    "type": "pitch_bend",
                    "channel": note.channel,
                    "value": bend_value,
                    "time": time,
                }
            )

        for time, timbre in note.timbre:
            messages.append(
                {
                    "type": "control_change",
                    "channel": note.channel,
                    "control": 74,
                    "value": timbre,
                    "time": time,
                }
            )

        for time, slide in note.slide:
            messages.append(
                {
                    "type": "control_change",
                    "channel": note.channel,
                    "control": 70,
                    "value": slide,
                    "time": time,
                }
            )

        messages.append(
            {
                "type": "note_off",
                "channel": note.channel,
                "pitch": note.pitch,
                "velocity": 64,
                "time": note.start_time + note.duration,
            }
        )

        messages.sort(key=lambda x: x["time"])
        return messages

    def release_note(self, note: MPENote):
        if id(note) in self.active_notes:
            self.channel_allocator.release_channel(note.channel)
            del self.active_notes[id(note)]


class MPEChannelAllocator:
    def __init__(self, config: MPEConfiguration = None):
        self.config = config
        self.available_lower = (
            set(config.lower_zone_channels) if config else set()
        )
        self.available_upper = (
            set(config.upper_zone_channels) if config else set()
        )
        self.allocated = {}

    def allocate_channel(self, zone: MPEZone) -> int:
        if self.config is None:
            return 1
        available = (
            self.available_lower
            if zone == MPEZone.LOWER
            else self.available_upper
        )
        if not available:
            raise MPEChannelExhaustedError(
                f"No free channel left in the {zone.value} MPE zone"
            )
        channel = min(available)
        available.remove(channel)
        self.allocated[channel] = zone
        return channel

    def release_channel(self, channel: int):
        if channel in self.allocated:
            zone = self.allocated[channel]
            if zone == MPEZone.LOWER:
                self.available_lower.add(channel)
            else:
                self.available_upper.add(channel)
            del self.allocated[channel]


class MPEVisualizer:
    def __init__(self):
        pass

    def get_pressure_curve_data(self, note: MPENote) -> list[tuple[float, int]]:
        return note.pressure.copy()

    def get_pitch_bend_curve_data(
        self, note: MPENote
    ) -> list[tuple[float, float]]:
        return note.pitch_bend.copy()

    def get_timbre_curve_data(self, note: MPENote) -> list[tuple[float, int]]:
        return note.timbre.copy()

    def interpolate_curve(
        self, curve_data: list[tuple[float, float]], num_points: int = 100
    ) -> np.ndarray:
        if not curve_data:
            return np.array([])

        curve_data = sorted(curve_data, key=lambda x: x[0])
        times = np.array([t for t, _ in curve_data])
        values = np.array([v for _, v in curve_data])

        if len(times) < 2:
            return np.array([[times[0], values[0]]] * num_points)

        interp_times = np.linspace(times[0], times[-1], num_points)
        interp_values = np.interp(interp_times, times, values)

        return np.column_stack([interp_times, interp_values])


class MPEToStandardMIDIConverter:
    PITCH_BEND_RANGE = 24

    def __init__(self, pitch_bend_range: int | None = None):
        if pitch_bend_range is not None:
            self.PITCH_BEND_RANGE = pitch_bend_range

    def convert_note(self, mpe_note: MPENote) -> list[dict]:
        messages = []

        base_pitch = mpe_note.pitch

        messages.append(
            {
                "type": "note_on",
                "channel": 0,
                "pitch": base_pitch,
                "velocity": mpe_note.velocity,
                "time": mpe_note.start_time,
            }
        )

        if mpe_note.pressure:
            avg_pressure = int(np.mean([p for _, p in mpe_note.pressure]))
            messages.append(
                {
                    "type": "channel_aftertouch",
                    "channel": 0,
                    "pressure": avg_pressure,
                    "time": mpe_note.start_time,
                }
            )

        if mpe_note.pitch_bend:
            for time, bend in mpe_note.pitch_bend:
                normalized_bend = bend / float(self.PITCH_BEND_RANGE)
                normalized_bend = max(-1.0, min(1.0, normalized_bend))
                bend_value = int(8192 + normalized_bend * 8192)
                bend_value = max(0, min(16383, bend_value))
                messages.append(
                    {
                        "type": "pitch_bend",
                        "channel": 0,
                        "value": bend_value,
                        "time": time,
                    }
                )

        messages.append(
            {
                "type": "note_off",
                "channel": 0,
                "pitch": base_pitch,
                "velocity": 64,
                "time": mpe_note.start_time + mpe_note.duration,
            }
        )

        messages.sort(key=lambda x: x["time"])
        return messages


class MPEControllerMapper:
    CONTROLLER_PRESETS = {
        "roli_seaboard": {
            "pitch_bend_range": 48,
            "pressure_cc": None,
            "timbre_cc": 74,
            "slide_cc": 70,
        },
        "linnstrument": {
            "pitch_bend_range": 24,
            "pressure_cc": None,
            "timbre_cc": 74,
            "slide_cc": 70,
        },
        "haken_continuum": {
            "pitch_bend_range": 96,
            "pressure_cc": None,
            "timbre_cc": 74,
            "slide_cc": 70,
        },
    }

    def __init__(self, controller_type: str = "roli_seaboard"):
        self.settings = self.CONTROLLER_PRESETS.get(
            controller_type, self.CONTROLLER_PRESETS["roli_seaboard"]
        )

    def get_pitch_bend_range(self) -> int:
        return self.settings["pitch_bend_range"]

    def get_timbre_cc(self) -> int:
        return self.settings["timbre_cc"]

    def get_slide_cc(self) -> int:
        return self.settings["slide_cc"]


class MPEGlissandoGenerator:
    def __init__(self):
        pass

    def generate_glissando(
        self,
        start_pitch: int,
        end_pitch: int,
        start_time: float,
        duration: float,
        num_steps: int = 50,
    ) -> MPENote:
        start_pitch = validate_midi_note(start_pitch, "start_pitch")
        end_pitch = validate_midi_note(end_pitch, "end_pitch")
        start_time = _finite_number(start_time, "start_time")
        duration = _finite_number(duration, "duration")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration!r}")
        num_steps = _point_count(num_steps, "num_steps")
        note = MPENote(
            pitch=start_pitch,
            start_time=start_time,
            duration=duration,
            velocity=100,
            channel=1,
            pressure=[],
            pitch_bend=[],
            timbre=[],
            slide=[],
        )

        pitch_diff = end_pitch - start_pitch

        for i in range(num_steps + 1):
            t = i / num_steps
            time = start_time + t * duration
            bend = pitch_diff * t
            note.pitch_bend.append((time, bend))

        return note

    def generate_vibrato(
        self, note: MPENote, rate_hz: float = 5.0, depth_semitones: float = 0.5
    ):
        rate_hz = _finite_number(rate_hz, "rate_hz")
        depth_semitones = _finite_number(depth_semitones, "depth_semitones")
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
        if note.duration <= 0:
            return

        num_points = int(note.duration * rate_hz * 10)
        if num_points > MAX_CURVE_POINTS:
            raise ValueError(
                f"rate_hz {rate_hz!r} over a duration of {note.duration!r} "
                f"needs more than {MAX_CURVE_POINTS} points"
            )

        if num_points < 2:
            return

        for i in range(num_points):
            t = i / (num_points - 1)
            time = note.start_time + t * note.duration

            phase = 2 * np.pi * rate_hz * t * note.duration
            bend = depth_semitones * np.sin(phase)

            note.pitch_bend.append((time, bend))
