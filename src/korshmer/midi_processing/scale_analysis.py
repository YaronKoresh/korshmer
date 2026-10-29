import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from korshmer.utils.scale_names import sanitize_scale_name

from .note_validation import validate_midi_note

MAX_DIVISIONS = 1000
MAX_SCALE_INTERVAL = 23


def _finite_number(value, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return float(value)


def _whole_number(value, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} must be a whole number, got {value!r}")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(
            f"{label} must be from {minimum} to {maximum}, got {value!r}"
        )
    return number


class ScaleType(Enum):
    MAJOR = "major"
    NATURAL_MINOR = "natural_minor"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    PHRYGIAN_DOMINANT = "phrygian_dominant"
    HUNGARIAN_MINOR = "hungarian_minor"
    NEAPOLITAN_MAJOR = "neapolitan_major"
    NEAPOLITAN_MINOR = "neapolitan_minor"
    WHOLE_TONE = "whole_tone"
    DIMINISHED = "diminished"
    CHROMATIC = "chromatic"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    BLUES = "blues"
    ARABIC = "arabic"
    JAPANESE = "japanese"
    CHINESE = "chinese"
    CUSTOM = "custom"


@dataclass
class Scale:
    root: int
    scale_type: ScaleType
    intervals: tuple[int, ...]
    degrees: tuple[int, ...]

    def get_notes(self) -> list[int]:
        return [self.root + interval for interval in self.intervals]

    def get_name(self) -> str:
        note_names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        root_name = note_names[self.root % 12]
        scale_name = self.scale_type.value.replace("_", " ").title()
        return f"{root_name} {scale_name}"

    def get_formula(self) -> str:
        degree_names = [
            "1",
            "b2",
            "2",
            "b3",
            "3",
            "4",
            "b5",
            "5",
            "b6",
            "6",
            "b7",
            "7",
        ]
        extension_map = {
            "b2": "b9",
            "2": "9",
            "4": "11",
            "b6": "b13",
            "6": "13",
        }
        formula = []
        for interval in self.intervals:
            if interval < len(degree_names):
                formula.append(degree_names[interval])
            else:
                base_name = degree_names[interval % 12]
                extended_name = extension_map.get(base_name, base_name)
                formula.append(extended_name)
        return " - ".join(formula)

    def contains_note(self, note: int) -> bool:
        return (note % 12) in [(self.root + i) % 12 for i in self.intervals]


class ScaleDetector:
    SCALE_TEMPLATES = {
        ScaleType.MAJOR: (0, 2, 4, 5, 7, 9, 11),
        ScaleType.NATURAL_MINOR: (0, 2, 3, 5, 7, 8, 10),
        ScaleType.HARMONIC_MINOR: (0, 2, 3, 5, 7, 8, 11),
        ScaleType.MELODIC_MINOR: (0, 2, 3, 5, 7, 9, 11),
        ScaleType.DORIAN: (0, 2, 3, 5, 7, 9, 10),
        ScaleType.PHRYGIAN: (0, 1, 3, 5, 7, 8, 10),
        ScaleType.LYDIAN: (0, 2, 4, 6, 7, 9, 11),
        ScaleType.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
        ScaleType.AEOLIAN: (0, 2, 3, 5, 7, 8, 10),
        ScaleType.LOCRIAN: (0, 1, 3, 5, 6, 8, 10),
        ScaleType.PHRYGIAN_DOMINANT: (0, 1, 4, 5, 7, 8, 10),
        ScaleType.HUNGARIAN_MINOR: (0, 2, 3, 6, 7, 8, 11),
        ScaleType.NEAPOLITAN_MAJOR: (0, 1, 3, 5, 7, 9, 11),
        ScaleType.NEAPOLITAN_MINOR: (0, 1, 3, 5, 7, 8, 11),
        ScaleType.WHOLE_TONE: (0, 2, 4, 6, 8, 10),
        ScaleType.DIMINISHED: (0, 2, 3, 5, 6, 8, 9, 11),
        ScaleType.CHROMATIC: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        ScaleType.PENTATONIC_MAJOR: (0, 2, 4, 7, 9),
        ScaleType.PENTATONIC_MINOR: (0, 3, 5, 7, 10),
        ScaleType.BLUES: (0, 3, 5, 6, 7, 10),
        ScaleType.ARABIC: (0, 1, 4, 5, 7, 8, 11),
        ScaleType.JAPANESE: (0, 1, 5, 7, 8),
        ScaleType.CHINESE: (0, 4, 6, 7, 11),
    }

    def __init__(self):
        self.detected_scales: list[Scale] = []

    def detect_scale_from_notes(self, notes: list[int]) -> list[Scale]:
        if not isinstance(notes, (list, tuple, np.ndarray)):
            raise ValueError(f"Notes must be a list, got {notes!r}")
        if len(notes) < 3:
            return []
        notes = [validate_midi_note(n, "note") for n in notes]

        unique_notes = {n % 12 for n in notes}
        histogram = self._create_histogram(notes)
        edge_pitch_classes = {notes[0] % 12, notes[-1] % 12}

        candidates = []
        seen = set()
        for root in range(12):
            for scale_type, intervals in self.SCALE_TEMPLATES.items():
                if (root, intervals) in seen:
                    continue
                seen.add((root, intervals))
                scale_notes = {(root + i) % 12 for i in intervals}
                match_score = self._calculate_match_score(
                    unique_notes, scale_notes
                )

                if match_score > 0.6:
                    scale = Scale(
                        root=root,
                        scale_type=scale_type,
                        intervals=intervals,
                        degrees=tuple(range(1, len(intervals) + 1)),
                    )
                    tonic_weight = (
                        histogram[root]
                        + 0.5 * histogram[(root + 7) % 12]
                        + (0.5 if root in edge_pitch_classes else 0.0)
                    )
                    candidates.append(
                        (round(match_score, 9), float(tonic_weight), scale)
                    )

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [scale for _, _, scale in candidates[:5]]

    def _create_histogram(self, notes: list[int]) -> np.ndarray:
        histogram = np.zeros(12)
        for note in notes:
            histogram[note % 12] += 1
        return histogram / histogram.sum() if histogram.sum() > 0 else histogram

    def _calculate_match_score(
        self, notes: set[int], scale_notes: set[int]
    ) -> float:
        if len(scale_notes) == 0:
            return 0.0

        intersection = len(notes & scale_notes)
        union = len(notes | scale_notes)

        if union == 0:
            return 0.0

        jaccard = intersection / union
        coverage = intersection / len(scale_notes)

        return (jaccard + coverage) / 2

    def suggest_scale(self, notes: list[int]) -> Scale | None:
        scales = self.detect_scale_from_notes(notes)
        return scales[0] if scales else None

    def get_relative_scales(self, scale: Scale) -> list[Scale]:
        relatives = []

        if scale.scale_type == ScaleType.MAJOR:
            relative_minor_root = (scale.root - 3) % 12
            relatives.append(
                Scale(
                    root=relative_minor_root,
                    scale_type=ScaleType.NATURAL_MINOR,
                    intervals=self.SCALE_TEMPLATES[ScaleType.NATURAL_MINOR],
                    degrees=tuple(range(1, 8)),
                )
            )
        elif scale.scale_type == ScaleType.NATURAL_MINOR:
            relative_major_root = (scale.root + 3) % 12
            relatives.append(
                Scale(
                    root=relative_major_root,
                    scale_type=ScaleType.MAJOR,
                    intervals=self.SCALE_TEMPLATES[ScaleType.MAJOR],
                    degrees=tuple(range(1, 8)),
                )
            )

        parallel_root = scale.root
        if scale.scale_type == ScaleType.MAJOR:
            relatives.append(
                Scale(
                    root=parallel_root,
                    scale_type=ScaleType.NATURAL_MINOR,
                    intervals=self.SCALE_TEMPLATES[ScaleType.NATURAL_MINOR],
                    degrees=tuple(range(1, 8)),
                )
            )
        elif scale.scale_type == ScaleType.NATURAL_MINOR:
            relatives.append(
                Scale(
                    root=parallel_root,
                    scale_type=ScaleType.MAJOR,
                    intervals=self.SCALE_TEMPLATES[ScaleType.MAJOR],
                    degrees=tuple(range(1, 8)),
                )
            )

        return relatives

    def compare_scales(self, scale1: Scale, scale2: Scale) -> dict[str, Any]:
        notes1 = {(scale1.root + i) % 12 for i in scale1.intervals}
        notes2 = {(scale2.root + i) % 12 for i in scale2.intervals}

        common = notes1 & notes2
        only_in_first = notes1 - notes2
        only_in_second = notes2 - notes1

        return {
            "common_notes": sorted(common),
            "unique_to_first": sorted(only_in_first),
            "unique_to_second": sorted(only_in_second),
            "similarity": (
                len(common) / max(len(notes1), len(notes2))
                if max(len(notes1), len(notes2)) > 0
                else 0
            ),
        }

    def detect_modal_interchange(
        self, notes: list[int], primary_scale: Scale
    ) -> list[int]:
        borrowed_notes = []
        parallel_scales = self._get_parallel_modes(primary_scale)

        for note in notes:
            if not primary_scale.contains_note(note):
                for parallel in parallel_scales:
                    if parallel.contains_note(note):
                        borrowed_notes.append(note)
                        break

        return borrowed_notes

    def _get_parallel_modes(self, scale: Scale) -> list[Scale]:
        modes = []
        for scale_type in [
            ScaleType.DORIAN,
            ScaleType.PHRYGIAN,
            ScaleType.LYDIAN,
            ScaleType.MIXOLYDIAN,
            ScaleType.AEOLIAN,
            ScaleType.LOCRIAN,
        ]:
            modes.append(
                Scale(
                    root=scale.root,
                    scale_type=scale_type,
                    intervals=self.SCALE_TEMPLATES[scale_type],
                    degrees=tuple(range(1, 8)),
                )
            )
        return modes


class MicrotonalScale:
    def __init__(self, root: float, divisions_per_octave: int = 12):
        self.root = _finite_number(root, "Root")
        self.divisions = _whole_number(
            divisions_per_octave, "Divisions per octave", 1, MAX_DIVISIONS
        )
        self.intervals: list[float] = []

    def add_interval(self, interval: float):
        self.intervals.append(_finite_number(interval, "Interval"))

    def get_frequency(self, note_index: int, base_freq: float = 440.0) -> float:
        note_index = _whole_number(note_index, "Note index", 0, 10**9)
        base_freq = _finite_number(base_freq, "Base frequency")
        if base_freq <= 0:
            raise ValueError(
                f"Base frequency must be above 0, got {base_freq!r}"
            )
        if note_index >= len(self.intervals):
            return base_freq

        semitones = self.intervals[note_index]
        return base_freq * (2 ** (semitones / 12))

    def to_equal_temperament(self, divisions: int = 24) -> list[float]:
        divisions = _whole_number(divisions, "Divisions", 1, MAX_DIVISIONS)
        step_size = 12.0 / divisions
        return [i * step_size for i in range(divisions)]


class CustomScaleBuilder:
    def __init__(self):
        self.custom_scales: dict[str, Scale] = {}

    def create_scale(
        self, name: str, root: int, intervals: tuple[int, ...]
    ) -> Scale:
        clean_name = sanitize_scale_name(name)
        if not clean_name:
            raise ValueError("scale name must not be empty")
        root = validate_midi_note(root, "root")
        if not isinstance(intervals, (list, tuple)) or not intervals:
            raise ValueError(
                f"Scale intervals must be a non-empty list, got {intervals!r}"
            )
        intervals = tuple(
            sorted(
                _whole_number(i, "Scale interval", 0, MAX_SCALE_INTERVAL)
                for i in intervals
            )
        )
        if len(set(intervals)) != len(intervals):
            raise ValueError(
                f"Scale intervals must not repeat, got {intervals!r}"
            )
        scale = Scale(
            root=root,
            scale_type=ScaleType.CUSTOM,
            intervals=intervals,
            degrees=tuple(range(1, len(intervals) + 1)),
        )
        self.custom_scales[clean_name] = scale
        return scale

    def save_scale(self, name: str, scale: Scale) -> bool:
        clean_name = sanitize_scale_name(name)
        if not clean_name:
            raise ValueError("scale name must not be empty")
        self.custom_scales[clean_name] = scale
        return True

    def load_scale(self, name: str) -> Scale | None:
        clean_name = sanitize_scale_name(name)
        return self.custom_scales.get(clean_name)

    def delete_scale(self, name: str) -> bool:
        clean_name = sanitize_scale_name(name)
        if clean_name in self.custom_scales:
            del self.custom_scales[clean_name]
            return True
        return False

    def list_scales(self) -> list[str]:
        return list(self.custom_scales.keys())


class ScaleDegreeHighlighter:
    def __init__(self, scale: Scale = None):
        self.scale = scale

    def get_degree_color(self, note: int) -> str:
        if self.scale is None or not self.scale.contains_note(note):
            return "#808080"

        note_in_scale = (note - self.scale.root) % 12

        try:
            degree_index = list(self.scale.intervals).index(note_in_scale)
        except ValueError:
            return "#808080"

        color_map = {
            0: "#FF0000",
            1: "#FF8800",
            2: "#FFFF00",
            3: "#00FF00",
            4: "#00FFFF",
            5: "#0000FF",
            6: "#FF00FF",
        }

        return color_map.get(degree_index % 7, "#FFFFFF")

    def get_degree_name(self, note: int) -> str:
        if self.scale is None or not self.scale.contains_note(note):
            return "X"

        note_in_scale = (note - self.scale.root) % 12

        try:
            degree_index = list(self.scale.intervals).index(note_in_scale)
        except ValueError:
            return "X"

        return str(degree_index + 1)
