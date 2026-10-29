import math
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from korshmer.audio_processing.validation import validate_time_signature
from korshmer.utils.tempo import validate_bpm

from .note_validation import (
    MIDI_NOTE_MAX,
    validate_midi_cc_number,
    validate_midi_note,
)

MAX_PATTERN_STEPS: int = 4096
MAX_ARPEGGIO_NOTES: int = 100000
MAX_RATCHET_REPEATS: int = 64
CURVE_TYPES: tuple[str, ...] = (
    "linear",
    "exponential",
    "logarithmic",
    "s-curve",
)


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(
        value, (int, float, np.integer, np.floating)
    )


def _validate_finite_number(value, label: str, allow_zero: bool) -> float:
    if not _is_number(value):
        raise ValueError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < 0
        or (result == 0 and not allow_zero)
    ):
        bound = "0 or more" if allow_zero else "above 0"
        raise ValueError(
            f"{label} must be a finite number {bound}, got {value!r}"
        )
    return result


def _validate_count(value, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a whole number, got {value!r}")
    result = int(value)
    if result < minimum or result > MAX_PATTERN_STEPS:
        raise ValueError(
            f"{label} must be from {minimum} to {MAX_PATTERN_STEPS}, "
            f"got {value!r}"
        )
    return result


def _validate_repeats(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"Ratchet repeats must be a whole number, got {value!r}"
        )
    if value > MAX_RATCHET_REPEATS:
        raise ValueError(
            f"Ratchet repeats must be at most {MAX_RATCHET_REPEATS}, "
            f"got {value!r}"
        )
    return int(value)


def _validate_cc_value(value, label: str) -> int:
    if not _is_number(value) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return int(max(0, min(127, round(value))))


class ArpPattern(Enum):
    UP = "up"
    DOWN = "down"
    UP_DOWN = "up_down"
    DOWN_UP = "down_up"
    RANDOM = "random"
    AS_PLAYED = "as_played"
    CHORD = "chord"


class StepMode(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    PINGPONG = "pingpong"
    RANDOM = "random"


@dataclass
class ArpNote:
    pitch: int
    start_time: float
    duration: float
    velocity: int
    gate_length: float = 1.0

    def __post_init__(self):
        self.pitch = validate_midi_note(self.pitch)


@dataclass
class Pattern:
    name: str
    steps: list[bool]
    velocities: list[int]
    gate_lengths: list[float]
    pitch_offsets: list[int]

    def __len__(self):
        return len(self.steps)


class Arpeggiator:
    def __init__(
        self, bpm: float = 120.0, time_signature: tuple[int, int] = (4, 4)
    ):
        self.bpm = validate_bpm(bpm)
        self.time_signature = validate_time_signature(time_signature)
        self.octave_range = 1
        self.gate_length = 0.8
        self.pattern = ArpPattern.UP

    def generate_arpeggio(
        self,
        notes: list[int],
        duration: float,
        subdivision: float = 0.25,
        velocity: int = 100,
    ) -> list[ArpNote]:
        if not notes:
            return []

        notes = [validate_midi_note(n, "note") for n in notes]
        duration = _validate_finite_number(duration, "Duration", True)
        subdivision = _validate_finite_number(subdivision, "Subdivision", False)

        beat_duration = 60.0 / self.bpm
        step_duration = beat_duration * subdivision
        num_steps = int(duration / step_duration)
        arp_sequence = self._generate_sequence(notes)
        per_step = len(arp_sequence) if self.pattern == ArpPattern.CHORD else 1
        if num_steps * per_step > MAX_ARPEGGIO_NOTES:
            raise ValueError(
                f"Arpeggio would have {num_steps * per_step} notes, "
                f"the limit is {MAX_ARPEGGIO_NOTES}"
            )

        arp_notes = []
        for i in range(num_steps):
            if self.pattern == ArpPattern.CHORD:
                pitches = arp_sequence
            else:
                pitches = [arp_sequence[i % len(arp_sequence)]]

            for pitch in pitches:
                arp_notes.append(
                    ArpNote(
                        pitch=pitch,
                        start_time=i * step_duration,
                        duration=step_duration * self.gate_length,
                        velocity=velocity,
                        gate_length=self.gate_length,
                    )
                )

        return arp_notes

    def _generate_sequence(self, notes: list[int]) -> list[int]:
        sorted_notes = sorted(set(notes))

        sequence = []
        for octave in range(self.octave_range):
            octave_offset = octave * 12

            shifted = [
                n + octave_offset
                for n in sorted_notes
                if n + octave_offset <= MIDI_NOTE_MAX
            ]
            if not shifted:
                break

            if self.pattern == ArpPattern.CHORD:
                sequence.extend(shifted)
                break
            elif self.pattern == ArpPattern.UP:
                sequence.extend(shifted)
            elif self.pattern == ArpPattern.DOWN:
                sequence.extend(reversed(shifted))
            elif self.pattern == ArpPattern.UP_DOWN:
                up = shifted
                down = list(reversed(shifted[:-1]))
                sequence.extend(up + down)
            elif self.pattern == ArpPattern.DOWN_UP:
                down = list(reversed(shifted))
                up = shifted[1:]
                sequence.extend(down + up)
            elif self.pattern == ArpPattern.RANDOM:
                sequence.extend(random.sample(shifted, len(shifted)))
            elif self.pattern == ArpPattern.AS_PLAYED:
                sequence.extend(
                    [
                        n + octave_offset
                        for n in notes
                        if n + octave_offset <= MIDI_NOTE_MAX
                    ]
                )

        return sequence if sequence else sorted_notes

    def set_octave_range(self, octaves: int):
        if isinstance(octaves, bool) or not isinstance(
            octaves, (int, np.integer)
        ):
            raise ValueError(
                f"Octave range must be a whole number, got {octaves!r}"
            )
        self.octave_range = max(1, min(4, int(octaves)))

    def set_gate_length(self, length: float):
        length = _validate_finite_number(length, "Gate length", True)
        self.gate_length = min(1.0, max(0.1, length))


class StepSequencer:
    def __init__(self, num_steps: int = 16, bpm: float = 120.0):
        self.num_steps = _validate_count(num_steps, "Step count", 1)
        self.bpm = validate_bpm(bpm)
        self.patterns: list[Pattern] = []
        self.current_pattern = 0

    def create_pattern(
        self,
        name: str,
        steps: list[bool] | None = None,
        velocities: list[int] | None = None,
    ) -> Pattern:
        if steps is None:
            steps = [True] * self.num_steps
        else:
            steps = steps[: self.num_steps] + [False] * (
                self.num_steps - len(steps)
            )

        if velocities is None:
            velocities = [100] * self.num_steps
        else:
            velocities = velocities[: self.num_steps] + [100] * (
                self.num_steps - len(velocities)
            )

        pattern = Pattern(
            name=name,
            steps=steps,
            velocities=velocities,
            gate_lengths=[0.8] * self.num_steps,
            pitch_offsets=[0] * self.num_steps,
        )

        self.patterns.append(pattern)
        return pattern

    def generate_from_pattern(
        self, pattern: Pattern, base_pitch: int, duration: float
    ) -> list[ArpNote]:
        base_pitch = validate_midi_note(base_pitch, "base pitch")
        duration = _validate_finite_number(duration, "Duration", True)
        if not pattern.steps:
            return []
        step_duration = duration / len(pattern.steps)

        notes = []
        for i, active in enumerate(pattern.steps):
            if active:
                notes.append(
                    ArpNote(
                        pitch=base_pitch + pattern.pitch_offsets[i],
                        start_time=i * step_duration,
                        duration=step_duration * pattern.gate_lengths[i],
                        velocity=pattern.velocities[i],
                        gate_length=pattern.gate_lengths[i],
                    )
                )

        return notes

    def apply_probability(
        self, pattern: Pattern, probabilities: list[float]
    ) -> Pattern:
        new_steps = []
        for i, step in enumerate(pattern.steps):
            prob = probabilities[i] if i < len(probabilities) else 1.0
            new_steps.append(step and random.random() < prob)

        return Pattern(
            name=f"{pattern.name} (Probabilistic)",
            steps=new_steps,
            velocities=pattern.velocities.copy(),
            gate_lengths=pattern.gate_lengths.copy(),
            pitch_offsets=pattern.pitch_offsets.copy(),
        )


class RatchetGenerator:
    def __init__(self):
        pass

    def apply_ratcheting(
        self, notes: list[ArpNote], ratchet_pattern: list[int]
    ) -> list[ArpNote]:
        if not ratchet_pattern:
            return notes

        ratcheted = []

        for i, note in enumerate(notes):
            repeats = _validate_repeats(
                ratchet_pattern[i % len(ratchet_pattern)]
            )

            if repeats <= 1:
                ratcheted.append(note)
            else:
                subdivision = note.duration / repeats
                for r in range(repeats):
                    ratcheted.append(
                        ArpNote(
                            pitch=note.pitch,
                            start_time=note.start_time + r * subdivision,
                            duration=subdivision * note.gate_length,
                            velocity=note.velocity,
                            gate_length=note.gate_length,
                        )
                    )

        return ratcheted


class EuclideanRhythmGenerator:
    def __init__(self):
        pass

    def generate_euclidean(self, pulses: int, steps: int) -> list[bool]:
        pulses = _validate_count(pulses, "Pulses", 0)
        steps = _validate_count(steps, "Steps", 0)
        if pulses >= steps:
            return [True] * steps
        if pulses == 0:
            return [False] * steps

        pattern = []
        bucket = 0

        for _i in range(steps):
            bucket += pulses
            if bucket >= steps:
                bucket -= steps
                pattern.append(True)
            else:
                pattern.append(False)

        return pattern

    def rotate_pattern(self, pattern: list[bool], rotation: int) -> list[bool]:
        if isinstance(rotation, bool) or not isinstance(
            rotation, (int, np.integer)
        ):
            raise ValueError(
                f"Rotation must be a whole number, got {rotation!r}"
            )
        if not pattern:
            return []
        rotation = int(rotation) % len(pattern)
        return pattern[rotation:] + pattern[:rotation]


class PolyrhythmGenerator:
    def __init__(self):
        pass

    def generate_polyrhythm(
        self, ratios: list[tuple[int, int]], duration: float, base_pitch: int
    ) -> list[list[ArpNote]]:
        duration = _validate_finite_number(duration, "Duration", False)
        base_pitch = validate_midi_note(base_pitch, "base pitch")
        tracks = []

        for numerator, _denominator in ratios:
            numerator = _validate_count(numerator, "Ratio numerator", 1)
            step_duration = duration / numerator
            notes = []

            for i in range(numerator):
                notes.append(
                    ArpNote(
                        pitch=base_pitch,
                        start_time=i * step_duration,
                        duration=step_duration * 0.8,
                        velocity=100,
                        gate_length=0.8,
                    )
                )

            tracks.append(notes)

        return tracks


class PatternChainer:
    def __init__(self):
        self.patterns: list[Pattern] = []
        self.variations: dict[str, Pattern] = {}

    def add_pattern(self, pattern: Pattern):
        self.patterns.append(pattern)

    def chain_patterns(self, pattern_indices: list[int]) -> Pattern:
        if not pattern_indices:
            return None

        combined_steps = []
        combined_velocities = []
        combined_gates = []
        combined_offsets = []

        for idx in pattern_indices:
            if 0 <= idx < len(self.patterns):
                pattern = self.patterns[idx]
                combined_steps.extend(pattern.steps)
                combined_velocities.extend(pattern.velocities)
                combined_gates.extend(pattern.gate_lengths)
                combined_offsets.extend(pattern.pitch_offsets)

        return Pattern(
            name="Chained Pattern",
            steps=combined_steps,
            velocities=combined_velocities,
            gate_lengths=combined_gates,
            pitch_offsets=combined_offsets,
        )

    def create_variation(
        self, pattern: Pattern, mutation_rate: float = 0.2
    ) -> Pattern:
        new_steps = []
        for step in pattern.steps:
            if random.random() < mutation_rate:
                new_steps.append(not step)
            else:
                new_steps.append(step)

        new_velocities = []
        for vel in pattern.velocities:
            if random.random() < mutation_rate:
                new_velocities.append(
                    max(1, min(127, vel + random.randint(-20, 20)))
                )
            else:
                new_velocities.append(vel)

        variation = Pattern(
            name=f"{pattern.name} (Variation)",
            steps=new_steps,
            velocities=new_velocities,
            gate_lengths=pattern.gate_lengths.copy(),
            pitch_offsets=pattern.pitch_offsets.copy(),
        )

        self.variations[variation.name] = variation
        return variation


class MidiCCAutomation:
    @dataclass
    class CCPoint:
        time: float
        value: int
        cc_number: int

    def __init__(self):
        self.automation_data: dict[int, list[MidiCCAutomation.CCPoint]] = {}

    def add_point(self, cc_number: int, time: float, value: int):
        cc_number = validate_midi_cc_number(cc_number)
        time = _validate_finite_number(time, "Time", True)
        value = _validate_cc_value(value, "CC value")
        if cc_number not in self.automation_data:
            self.automation_data[cc_number] = []

        point = self.CCPoint(time=time, value=value, cc_number=cc_number)
        self.automation_data[cc_number].append(point)
        self.automation_data[cc_number].sort(key=lambda p: p.time)

    def get_value_at_time(self, cc_number: int, time: float) -> int | None:
        if not _is_number(time) or not math.isfinite(time):
            raise ValueError(f"Time must be a finite number, got {time!r}")
        if cc_number not in self.automation_data:
            return None

        points = self.automation_data[cc_number]
        if not points:
            return None

        if time <= points[0].time:
            return points[0].value
        if time >= points[-1].time:
            return points[-1].value

        for i in range(len(points) - 1):
            if points[i].time <= time < points[i + 1].time:
                t = (time - points[i].time) / (
                    points[i + 1].time - points[i].time
                )
                value = points[i].value + t * (
                    points[i + 1].value - points[i].value
                )
                return int(value)

        return points[-1].value

    def generate_curve(
        self,
        cc_number: int,
        start_time: float,
        end_time: float,
        start_value: int,
        end_value: int,
        curve_type: str = "linear",
    ):
        cc_number = validate_midi_cc_number(cc_number)
        start_time = _validate_finite_number(start_time, "Start time", True)
        end_time = _validate_finite_number(end_time, "End time", True)
        if end_time < start_time:
            raise ValueError(
                f"End time {end_time!r} is before start time {start_time!r}"
            )
        start_value = _validate_cc_value(start_value, "Start value")
        end_value = _validate_cc_value(end_value, "End value")
        if curve_type not in CURVE_TYPES:
            raise ValueError(
                f"Unknown curve type {curve_type!r}, use one of "
                f"{', '.join(CURVE_TYPES)}"
            )
        num_points = 32
        for i in range(num_points + 1):
            t = i / num_points
            time = start_time + t * (end_time - start_time)

            if curve_type == "linear":
                value = start_value + t * (end_value - start_value)
            elif curve_type == "exponential":
                value = start_value + (end_value - start_value) * (t**2)
            elif curve_type == "logarithmic":
                value = start_value + (end_value - start_value) * np.sqrt(t)
            else:
                s = 3 * t**2 - 2 * t**3
                value = start_value + (end_value - start_value) * s

            self.add_point(cc_number, time, int(value))
