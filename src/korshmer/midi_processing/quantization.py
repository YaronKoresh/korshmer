import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import librosa
import numpy as np

from korshmer.audio_processing.validation import (
    is_empty_audio,
    validate_sample_rate,
    validate_time_signature,
)
from korshmer.utils.tempo import validate_bpm

from .note_validation import validate_midi_note


class QuantizeMode(Enum):
    NEAREST = "nearest"
    FLOOR = "floor"
    CEIL = "ceil"
    PROBABILISTIC = "probabilistic"


class NoteType(Enum):
    QUARTER = 1.0
    EIGHTH = 0.5
    SIXTEENTH = 0.25
    THIRTY_SECOND = 0.125
    DOTTED_QUARTER = 1.5
    DOTTED_EIGHTH = 0.75
    DOTTED_SIXTEENTH = 0.375
    TRIPLET_QUARTER = 2.0 / 3.0
    TRIPLET_EIGHTH = 1.0 / 3.0
    TRIPLET_SIXTEENTH = 1.0 / 6.0


@dataclass
class Note:
    start_time: float
    end_time: float
    pitch: int
    velocity: int

    def __post_init__(self):
        self.pitch = validate_midi_note(self.pitch)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class GrooveTemplate:
    name: str
    timing_offsets: dict[float, float]
    velocity_curve: dict[float, float]
    swing_amount: float = 0.0

    def apply_timing(self, position: float) -> float:
        if position in self.timing_offsets:
            return position + self.timing_offsets[position]

        keys = sorted(self.timing_offsets.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= position < keys[i + 1]:
                t = (position - keys[i]) / (keys[i + 1] - keys[i])
                offset1 = self.timing_offsets[keys[i]]
                offset2 = self.timing_offsets[keys[i + 1]]
                return position + (offset1 + t * (offset2 - offset1))

        return position

    def apply_velocity(self, position: float, velocity: int) -> int:
        if position in self.velocity_curve:
            factor = self.velocity_curve[position]
            return int(np.clip(velocity * factor, 1, 127))
        return velocity


def validate_grid_division(grid_division) -> float:
    if isinstance(grid_division, bool) or not isinstance(
        grid_division, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Grid division must be a positive number, got {grid_division!r}"
        )
    value = float(grid_division)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Grid division must be a finite number above 0, got {grid_division!r}"
        )
    return value


def validate_quantize_strength(strength) -> float:
    if isinstance(strength, bool) or not isinstance(
        strength, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Quantize strength must be a number from 0 to 1, got {strength!r}"
        )
    value = float(strength)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"Quantize strength must be from 0 to 1, got {strength!r}"
        )
    return value


def _validate_unit_fraction(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a number from 0 to 1, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be from 0 to 1, got {value!r}")
    return result


def validate_swing_amount(swing_amount) -> float:
    return _validate_unit_fraction(swing_amount, "Swing amount")


def validate_selective_threshold(threshold) -> float:
    return _validate_unit_fraction(threshold, "Selective quantize threshold")


def validate_grid_offset(offset) -> float:
    if isinstance(offset, bool) or not isinstance(
        offset, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Grid offset must be a number of seconds, got {offset!r}"
        )
    value = float(offset)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"Grid offset must be a finite number of 0 or more, got {offset!r}"
        )
    return value


class AdvancedQuantizer:
    def __init__(
        self,
        bpm: float = 120.0,
        time_signature: tuple[int, int] = (4, 4),
        grid_offset: float = 0.0,
    ):
        self.bpm = validate_bpm(bpm)
        self.grid_offset = grid_offset
        self.time_signature = validate_time_signature(time_signature)
        self.groove_template: GrooveTemplate | None = None
        self.quantize_strength = 1.0

    @property
    def grid_offset(self) -> float:
        return self._grid_offset

    @grid_offset.setter
    def grid_offset(self, value) -> None:
        self._grid_offset = validate_grid_offset(value)

    @property
    def quantize_strength(self) -> float:
        return self._quantize_strength

    @quantize_strength.setter
    def quantize_strength(self, value) -> None:
        self._quantize_strength = validate_quantize_strength(value)

    def quantize_notes(
        self,
        notes: list[Note],
        grid_division: float = 0.25,
        mode: QuantizeMode = QuantizeMode.NEAREST,
        humanize: bool = False,
    ) -> list[Note]:
        grid_division = validate_grid_division(grid_division)
        quantized = []

        for note in notes:
            q_start = self._quantize_time(note.start_time, grid_division, mode)
            q_end = self._quantize_time(note.end_time, grid_division, mode)

            if humanize:
                q_start = self._humanize_time(q_start, grid_division)
                q_end = self._humanize_time(q_end, grid_division)

            if self.groove_template:
                q_start = self.groove_template.apply_timing(q_start)
                q_end = self.groove_template.apply_timing(q_end)

            q_start = max(q_start, 0.0)

            quantized.append(
                Note(
                    start_time=q_start,
                    end_time=max(q_end, q_start + 0.01),
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )

        return quantized

    def _quantize_time(
        self, time: float, grid_division: float, mode: QuantizeMode
    ) -> float:
        beat_duration = 60.0 / self.bpm
        grid_size = beat_duration * grid_division
        time = time - self.grid_offset

        if mode == QuantizeMode.NEAREST:
            quantized = round(time / grid_size) * grid_size
        elif mode == QuantizeMode.FLOOR:
            quantized = np.floor(time / grid_size) * grid_size
        elif mode == QuantizeMode.CEIL:
            quantized = np.ceil(time / grid_size) * grid_size
        elif mode == QuantizeMode.PROBABILISTIC:
            lower = np.floor(time / grid_size) * grid_size
            upper = np.ceil(time / grid_size) * grid_size
            t = (time - lower) / (upper - lower) if upper > lower else 0
            quantized = lower if np.random.random() > t else upper
        else:
            quantized = time

        blend = self.quantize_strength
        quantized += self.grid_offset
        time += self.grid_offset
        return time * (1 - blend) + quantized * blend

    def _humanize_time(self, time: float, grid_division: float) -> float:
        beat_duration = 60.0 / self.bpm
        grid_size = beat_duration * grid_division

        max_deviation = grid_size * 0.1
        deviation = np.random.uniform(-max_deviation, max_deviation)

        return time + deviation

    def quantize_with_strength_curve(
        self,
        notes: list[Note],
        grid_division: float = 0.25,
        strength_curve: Callable[[float], float] | None = None,
    ) -> list[Note]:
        grid_division = validate_grid_division(grid_division)
        quantized = []

        for note in notes:
            beat_position = self._get_beat_position(note.start_time)

            if strength_curve:
                local_strength = float(strength_curve(beat_position))
                if math.isnan(local_strength):
                    raise ValueError("Strength curve returned NaN")
                local_strength = min(1.0, max(0.0, local_strength))
            else:
                local_strength = self.quantize_strength

            old_strength = self.quantize_strength
            self.quantize_strength = local_strength
            try:
                q_note = self.quantize_notes([note], grid_division)[0]
            finally:
                self.quantize_strength = old_strength
            quantized.append(q_note)

        return quantized

    def _get_beat_position(self, time: float) -> float:
        beat_duration = 60.0 / self.bpm
        return ((time - self.grid_offset) % beat_duration) / beat_duration

    def apply_swing(
        self, notes: list[Note], swing_amount: float = 0.5
    ) -> list[Note]:
        swing_amount = validate_swing_amount(swing_amount)
        beat_duration = 60.0 / self.bpm
        eighth_note = beat_duration * 0.5

        swung_notes = []
        for note in notes:
            beats_elapsed = (note.start_time - self.grid_offset) / beat_duration
            position_in_beat = beats_elapsed % 1.0

            is_offbeat = 0.4 < position_in_beat < 0.6

            if is_offbeat:
                delay = eighth_note * swing_amount * 0.25
                new_start = note.start_time + delay
                new_end = note.end_time + delay
            else:
                new_start = note.start_time
                new_end = note.end_time

            swung_notes.append(
                Note(
                    start_time=new_start,
                    end_time=new_end,
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )

        return swung_notes

    def selective_quantize(
        self,
        notes: list[Note],
        grid_division: float = 0.25,
        threshold: float = 0.1,
    ) -> list[Note]:
        grid_division = validate_grid_division(grid_division)
        threshold = validate_selective_threshold(threshold)
        beat_duration = 60.0 / self.bpm
        grid_size = beat_duration * grid_division

        quantized = []
        for note in notes:
            origin = self.grid_offset
            nearest_grid = (
                round((note.start_time - origin) / grid_size) * grid_size
                + origin
            )
            distance = abs(note.start_time - nearest_grid)

            if distance > grid_size * threshold:
                q_note = self.quantize_notes([note], grid_division)[0]
                quantized.append(q_note)
            else:
                quantized.append(note)

        return quantized


class GrooveExtractor:
    def __init__(self, bpm: float = 120.0):
        self.bpm = validate_bpm(bpm)

    def extract_from_midi(
        self, notes: list[Note], reference_grid: float = 0.25
    ) -> GrooveTemplate:
        reference_grid = validate_grid_division(reference_grid)
        beat_duration = 60.0 / self.bpm
        grid_size = beat_duration * reference_grid

        timing_offsets = {}
        velocity_curve = {}

        for note in notes:
            nearest_grid = round(note.start_time / grid_size) * grid_size
            offset = note.start_time - nearest_grid

            grid_position = nearest_grid
            if grid_position not in timing_offsets:
                timing_offsets[grid_position] = []
            timing_offsets[grid_position].append(offset)

            if grid_position not in velocity_curve:
                velocity_curve[grid_position] = []
            velocity_curve[grid_position].append(note.velocity / 127.0)

        avg_timing = {
            pos: np.mean(offsets) for pos, offsets in timing_offsets.items()
        }
        avg_velocity = {
            pos: np.mean(vels) for pos, vels in velocity_curve.items()
        }

        swing = self._estimate_swing(timing_offsets)

        return GrooveTemplate(
            name="Extracted Groove",
            timing_offsets=avg_timing,
            velocity_curve=avg_velocity,
            swing_amount=swing,
        )

    def _estimate_swing(
        self, timing_offsets: dict[float, list[float]]
    ) -> float:
        offbeat_offsets = []
        beat_duration = 60.0 / self.bpm

        for pos, offsets in timing_offsets.items():
            beat_position = (pos / beat_duration) % 1.0
            if 0.4 < beat_position < 0.6:
                offbeat_offsets.extend(offsets)

        if offbeat_offsets:
            avg_delay = np.mean(offbeat_offsets)
            return float(np.clip(avg_delay / (beat_duration * 0.125), 0, 1))

        return 0.0

    def extract_from_audio(
        self, audio: np.ndarray, sr: int, reference_grid: float = 0.25
    ) -> GrooveTemplate:
        reference_grid = validate_grid_division(reference_grid)
        sr = validate_sample_rate(sr)
        if not is_empty_audio(audio):
            audio = np.asarray(audio)
            if audio.ndim != 1:
                raise ValueError(
                    f"Audio must be mono (1-D), got {audio.ndim} dimensions"
                )
            if not np.isfinite(audio).all():
                raise ValueError("Audio contains NaN or infinite values")
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        beat_times = librosa.frames_to_time(beats, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])

        if not math.isfinite(tempo) or tempo <= 0 or len(beat_times) == 0:
            return GrooveTemplate(
                name="Audio Extracted Groove",
                timing_offsets={},
                velocity_curve={},
                swing_amount=0.0,
            )

        beat_duration = 60.0 / tempo
        grid_size = beat_duration * reference_grid

        timing_offsets = {}
        for beat_time in beat_times:
            nearest_grid = float(round(beat_time / grid_size) * grid_size)
            offset = float(beat_time - nearest_grid)

            if nearest_grid not in timing_offsets:
                timing_offsets[nearest_grid] = []
            timing_offsets[nearest_grid].append(offset)

        avg_timing = {
            pos: float(np.mean(offsets))
            for pos, offsets in timing_offsets.items()
        }

        return GrooveTemplate(
            name="Audio Extracted Groove",
            timing_offsets=avg_timing,
            velocity_curve={},
            swing_amount=0.0,
        )


class TripletQuantizer(AdvancedQuantizer):
    def quantize_to_triplets(self, notes: list[Note]) -> list[Note]:
        beat_duration = 60.0 / self.bpm
        triplet_grid = beat_duration / 3.0

        quantized = []
        for note in notes:
            q_start = round(note.start_time / triplet_grid) * triplet_grid
            q_end = round(note.end_time / triplet_grid) * triplet_grid

            quantized.append(
                Note(
                    start_time=q_start,
                    end_time=max(q_end, q_start + triplet_grid * 0.5),
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )

        return quantized


class DottedNoteQuantizer(AdvancedQuantizer):
    def quantize_to_dotted(
        self, notes: list[Note], base_division: float = 0.5
    ) -> list[Note]:
        base_division = validate_grid_division(base_division)
        beat_duration = 60.0 / self.bpm
        dotted_grid = beat_duration * base_division * 1.5

        quantized = []
        for note in notes:
            q_start = round(note.start_time / dotted_grid) * dotted_grid
            q_end = round(note.end_time / dotted_grid) * dotted_grid

            quantized.append(
                Note(
                    start_time=q_start,
                    end_time=max(q_end, q_start + dotted_grid * 0.5),
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )

        return quantized
