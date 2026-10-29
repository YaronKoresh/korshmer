import math
import numbers
import random
from dataclasses import dataclass
from enum import Enum

from .note_validation import validate_midi_note


class DrumArticulation(Enum):
    NORMAL = "normal"
    GHOST = "ghost"
    ACCENT = "accent"
    FLAM = "flam"
    DRAG = "drag"
    ROLL = "roll"


GM_MAP_NAME = "GM"
MAX_PATTERN_VARIATIONS = 1000


def _validate_map_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Drum map name must be text, got {name!r}")
    if name == GM_MAP_NAME:
        raise ValueError(f"The {GM_MAP_NAME} drum map cannot be changed")
    return name


def _validate_pattern(pattern, name="pattern") -> list[tuple[int, float, int]]:
    if not isinstance(pattern, (list, tuple)):
        raise ValueError(f"Drum {name} must be a list, got {pattern!r}")
    checked = []
    for entry in pattern:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                f"Drum {name} entries must be (pitch, time, velocity), "
                f"got {entry!r}"
            )
        pitch, time, velocity = entry
        pitch = validate_midi_note(pitch, "drum note")
        if (
            isinstance(time, bool)
            or not isinstance(time, numbers.Real)
            or not math.isfinite(time)
            or time < 0
        ):
            raise ValueError(
                f"Drum {name} time must be a finite number of 0 or more, "
                f"got {time!r}"
            )
        velocity = validate_midi_note(velocity, "velocity")
        if velocity < 1:
            raise ValueError(f"Drum {name} velocity must be 1 or more")
        checked.append((pitch, float(time), velocity))
    return checked


@dataclass
class DrumNote:
    pitch: int
    start_time: float
    velocity: int
    articulation: DrumArticulation = DrumArticulation.NORMAL

    def __post_init__(self):
        self.pitch = validate_midi_note(self.pitch)

    @property
    def duration(self) -> float:
        if self.articulation == DrumArticulation.ROLL:
            return 0.5
        return 0.1


@dataclass
class DrumSample:
    name: str
    midi_note: int
    file_path: str | None = None
    velocity_layers: list[str] = None

    def __post_init__(self):
        if self.velocity_layers is None:
            self.velocity_layers = []


GM_DRUM_MAP = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi-Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}


class DrumMapEditor:
    def __init__(self):
        self.custom_maps: dict[str, dict[int, DrumSample]] = {}
        self.current_map = "GM"
        self.gm_map = {
            note: DrumSample(name, note) for note, name in GM_DRUM_MAP.items()
        }

    def create_custom_map(self, name: str) -> dict[int, DrumSample]:
        name = _validate_map_name(name)
        self.custom_maps[name] = {}
        return self.custom_maps[name]

    def assign_sample(self, map_name: str, midi_note: int, sample: DrumSample):
        map_name = _validate_map_name(map_name)
        midi_note = validate_midi_note(midi_note, "drum note")
        if not isinstance(sample, DrumSample):
            raise ValueError(f"Sample must be a DrumSample, got {sample!r}")
        if map_name not in self.custom_maps:
            self.custom_maps[map_name] = {}

        self.custom_maps[map_name][midi_note] = sample

    def get_sample(self, map_name: str, midi_note: int) -> DrumSample | None:
        if map_name == "GM":
            return self.gm_map.get(midi_note)

        if map_name in self.custom_maps:
            return self.custom_maps[map_name].get(midi_note)

        return None

    def get_drum_name(self, midi_note: int) -> str:
        return GM_DRUM_MAP.get(midi_note, f"Unknown ({midi_note})")


class DrumPatternLibrary:
    ROCK_BASIC = [
        (36, 0.0, 100),
        (42, 0.0, 80),
        (42, 0.5, 80),
        (38, 1.0, 100),
        (42, 1.0, 80),
        (42, 1.5, 80),
        (36, 2.0, 100),
        (42, 2.0, 80),
        (42, 2.5, 80),
        (38, 3.0, 100),
        (42, 3.0, 80),
        (42, 3.5, 80),
    ]

    JAZZ_SWING = [
        (36, 0.0, 90),
        (42, 0.0, 70),
        (51, 0.333, 85),
        (38, 1.0, 95),
        (51, 1.333, 85),
        (36, 2.0, 90),
        (51, 2.333, 85),
        (38, 3.0, 95),
        (51, 3.333, 85),
    ]

    @staticmethod
    def _get_electronic_4on4():
        return (
            [(36, i * 0.25, 110) for i in range(16)]
            + [(42, i * 0.125, 90 if i % 2 == 0 else 70) for i in range(32)]
            + [(38, 1.0 + i * 2.0, 105) for i in range(2)]
        )

    def __init__(self):
        self.patterns: dict[str, list[tuple[int, float, int]]] = {
            "rock_basic": list(self.ROCK_BASIC),
            "jazz_swing": list(self.JAZZ_SWING),
            "electronic_4on4": self._get_electronic_4on4(),
        }

    def get_pattern(self, name: str) -> list[tuple[int, float, int]]:
        return list(self.patterns.get(name, []))

    def add_pattern(self, name: str, pattern: list[tuple[int, float, int]]):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Drum pattern name must be text, got {name!r}")
        self.patterns[name] = _validate_pattern(pattern)

    def generate_pattern_variations(
        self,
        base_pattern: list[tuple[int, float, int]],
        num_variations: int = 4,
    ) -> list[list[tuple[int, float, int]]]:
        if (
            isinstance(num_variations, bool)
            or not isinstance(num_variations, numbers.Integral)
            or not 0 <= num_variations <= MAX_PATTERN_VARIATIONS
        ):
            raise ValueError(
                "Number of variations must be a whole number from 0 to "
                f"{MAX_PATTERN_VARIATIONS}, got {num_variations!r}"
            )
        base_pattern = _validate_pattern(base_pattern, "base pattern")
        variations = []

        for _ in range(num_variations):
            variation = []
            for pitch, time, velocity in base_pattern:
                if random.random() > 0.1:
                    new_velocity = max(
                        1, min(127, velocity + random.randint(-15, 15))
                    )
                    variation.append((pitch, time, new_velocity))
            variations.append(variation)

        return variations


FILL_TYPES = ("tom_roll", "snare_roll", "crash_build", "generic")


class DrumFillGenerator:
    def __init__(self):
        self.intensity_levels = {"low": 0.3, "medium": 0.6, "high": 0.9}

    def generate_fill(
        self,
        duration: float = 2.0,
        intensity: str = "medium",
        fill_type: str = "tom_roll",
    ) -> list[DrumNote]:
        if intensity not in self.intensity_levels:
            raise ValueError(
                f"Unknown fill intensity {intensity!r}; "
                f"use one of {', '.join(self.intensity_levels)}"
            )
        if fill_type not in FILL_TYPES:
            raise ValueError(
                f"Unknown fill type {fill_type!r}; "
                f"use one of {', '.join(FILL_TYPES)}"
            )
        if (
            isinstance(duration, bool)
            or not isinstance(duration, numbers.Real)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError(
                f"Fill duration must be a finite number of at least 0, "
                f"got {duration!r}"
            )
        intensity_value = self.intensity_levels[intensity]

        if fill_type == "tom_roll":
            return self._generate_tom_roll(duration, intensity_value)
        elif fill_type == "snare_roll":
            return self._generate_snare_roll(duration, intensity_value)
        elif fill_type == "crash_build":
            return self._generate_crash_build(duration, intensity_value)
        else:
            return self._generate_generic_fill(duration, intensity_value)

    def _generate_tom_roll(
        self, duration: float, intensity: float
    ) -> list[DrumNote]:
        toms = [50, 48, 47, 45, 43, 41]
        notes = []

        num_hits = int(duration * 8 * intensity)

        for i in range(num_hits):
            tom_index = int((i / num_hits) * len(toms))
            tom_index = min(tom_index, len(toms) - 1)

            time = (i / num_hits) * duration
            velocity = int(80 + 40 * intensity * (i / num_hits))

            notes.append(
                DrumNote(
                    pitch=toms[tom_index],
                    start_time=time,
                    velocity=velocity,
                    articulation=DrumArticulation.NORMAL,
                )
            )

        return notes

    def _generate_snare_roll(
        self, duration: float, intensity: float
    ) -> list[DrumNote]:
        notes = []

        num_hits = int(duration * 16 * intensity)

        for i in range(num_hits):
            time = (i / num_hits) * duration
            velocity = int(60 + 50 * intensity)

            notes.append(
                DrumNote(
                    pitch=38,
                    start_time=time,
                    velocity=velocity,
                    articulation=DrumArticulation.ROLL,
                )
            )

        return notes

    def _generate_crash_build(
        self, duration: float, intensity: float
    ) -> list[DrumNote]:
        notes = []

        num_hits = int(duration * 4)

        for i in range(num_hits):
            time = (i / num_hits) * duration
            velocity = int(60 + 60 * intensity * (i / num_hits))

            notes.append(
                DrumNote(
                    pitch=42,
                    start_time=time,
                    velocity=velocity,
                    articulation=DrumArticulation.ACCENT,
                )
            )

        notes.append(
            DrumNote(
                pitch=49,
                start_time=duration,
                velocity=120,
                articulation=DrumArticulation.ACCENT,
            )
        )

        return notes

    def _generate_generic_fill(
        self, duration: float, intensity: float
    ) -> list[DrumNote]:
        drums = [38, 50, 48, 47, 45]
        notes = []

        num_hits = int(duration * 4)

        for i in range(num_hits):
            drum = drums[i % len(drums)]
            time = (i / num_hits) * duration
            velocity = int(80 + 40 * intensity)

            notes.append(
                DrumNote(
                    pitch=drum,
                    start_time=time,
                    velocity=velocity,
                    articulation=DrumArticulation.NORMAL,
                )
            )

        return notes


HIHAT_PATTERNS = ("eighth_notes", "sixteenth_notes")
ROLL_TYPES = ("single_stroke", "double_stroke")


class HiHatVariationGenerator:
    def __init__(self):
        self.closed_hihat = 42
        self.pedal_hihat = 44
        self.open_hihat = 46

    def generate_variation(
        self,
        duration: float = 4.0,
        pattern: str = "eighth_notes",
        openness: float = 0.2,
    ) -> list[DrumNote]:
        if pattern not in HIHAT_PATTERNS:
            raise ValueError(
                f"Unknown hi-hat pattern {pattern!r}; "
                f"use one of {', '.join(HIHAT_PATTERNS)}"
            )
        notes = []

        if pattern == "eighth_notes":
            num_notes = int(duration * 2)
            for i in range(num_notes):
                time = i * 0.5

                if random.random() < openness:
                    pitch = self.open_hihat
                    velocity = 90
                else:
                    pitch = self.closed_hihat
                    velocity = 80 if i % 2 == 0 else 60

                notes.append(
                    DrumNote(
                        pitch=pitch,
                        start_time=time,
                        velocity=velocity,
                        articulation=DrumArticulation.NORMAL,
                    )
                )

        elif pattern == "sixteenth_notes":
            num_notes = int(duration * 4)
            for i in range(num_notes):
                time = i * 0.25

                if i % 4 == 2 and random.random() < openness:
                    pitch = self.open_hihat
                    velocity = 95
                else:
                    pitch = self.closed_hihat
                    velocity = 85 if i % 4 == 0 else 65

                notes.append(
                    DrumNote(
                        pitch=pitch,
                        start_time=time,
                        velocity=velocity,
                        articulation=DrumArticulation.NORMAL,
                    )
                )

        return notes


class DrumHumanizer:
    def __init__(self):
        self.timing_variation = 0.02
        self.velocity_variation = 10

    def humanize_pattern(
        self, notes: list[DrumNote], add_ghost_notes: bool = True
    ) -> list[DrumNote]:
        timing = self.timing_variation
        if (
            isinstance(timing, bool)
            or not isinstance(timing, numbers.Real)
            or not math.isfinite(timing)
            or timing < 0
        ):
            raise ValueError(
                "Timing variation must be a finite number of 0 or more, "
                f"got {timing!r}"
            )
        spread = self.velocity_variation
        if (
            isinstance(spread, bool)
            or not isinstance(spread, numbers.Integral)
            or spread < 0
        ):
            raise ValueError(
                "Velocity variation must be a whole number of 0 or more, "
                f"got {spread!r}"
            )
        humanized = []

        for note in notes:
            time_offset = random.uniform(
                -self.timing_variation, self.timing_variation
            )
            new_time = max(0, note.start_time + time_offset)

            velocity_offset = random.randint(
                -self.velocity_variation, self.velocity_variation
            )
            new_velocity = max(1, min(127, note.velocity + velocity_offset))

            humanized.append(
                DrumNote(
                    pitch=note.pitch,
                    start_time=new_time,
                    velocity=new_velocity,
                    articulation=note.articulation,
                )
            )

        if add_ghost_notes:
            ghost_notes = self._add_ghost_notes(notes)
            humanized.extend(ghost_notes)

        humanized.sort(key=lambda n: n.start_time)
        return humanized

    def _add_ghost_notes(self, notes: list[DrumNote]) -> list[DrumNote]:
        ghost_notes = []

        snare_notes = [n for n in notes if n.pitch == 38]

        for snare in snare_notes:
            if random.random() < 0.3:
                ghost_time = snare.start_time - 0.125
                if ghost_time > 0:
                    ghost_notes.append(
                        DrumNote(
                            pitch=38,
                            start_time=ghost_time,
                            velocity=random.randint(30, 50),
                            articulation=DrumArticulation.GHOST,
                        )
                    )

        return ghost_notes


class DrumRollGenerator:
    def __init__(self):
        pass

    def generate_roll(
        self,
        pitch: int,
        start_time: float,
        duration: float,
        roll_type: str = "single_stroke",
    ) -> list[DrumNote]:
        if roll_type not in ROLL_TYPES:
            raise ValueError(
                f"Unknown roll type {roll_type!r}; "
                f"use one of {', '.join(ROLL_TYPES)}"
            )
        notes = []

        if roll_type == "single_stroke":
            subdivisions = int(duration * 32)
            for i in range(subdivisions):
                time = start_time + (i / subdivisions) * duration
                velocity = 70 + int(20 * (i / subdivisions))

                notes.append(
                    DrumNote(
                        pitch=pitch,
                        start_time=time,
                        velocity=velocity,
                        articulation=DrumArticulation.ROLL,
                    )
                )

        elif roll_type == "double_stroke":
            subdivisions = int(duration * 16)
            for i in range(subdivisions):
                time = start_time + (i / subdivisions) * duration
                velocity = 70 + int(20 * (i / subdivisions))

                notes.append(
                    DrumNote(
                        pitch=pitch,
                        start_time=time,
                        velocity=velocity,
                        articulation=DrumArticulation.ROLL,
                    )
                )

                notes.append(
                    DrumNote(
                        pitch=pitch,
                        start_time=time + 0.015,
                        velocity=velocity - 10,
                        articulation=DrumArticulation.ROLL,
                    )
                )

        return notes
