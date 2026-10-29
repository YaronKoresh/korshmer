import math
import numbers
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .note_validation import (
    fold_into_midi_range,
    validate_chord_progression,
    validate_midi_note,
)

MAX_VOICES: int = 16
MAX_SUBJECT_LENGTH: int = 1024
MAX_CANON_OFFSET: float = 60.0


def _validate_melody(melody, name: str = "melody") -> list[tuple[float, int]]:
    if not isinstance(melody, (list, tuple)):
        raise ValueError(f"The {name} must be a list, got {melody!r}")
    checked = []
    for entry in melody:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"Every {name} entry must be (time, pitch), got {entry!r}"
            )
        time, pitch = entry
        if (
            isinstance(time, bool)
            or not isinstance(time, numbers.Real)
            or not math.isfinite(time)
        ):
            raise ValueError(
                f"The {name} time must be a finite number, got {time!r}"
            )
        checked.append((float(time), validate_midi_note(pitch)))
    return checked


def _validate_pitch_list(notes, name: str) -> list[int]:
    if not isinstance(notes, (list, tuple)):
        raise ValueError(f"The {name} must be a list, got {notes!r}")
    return [validate_midi_note(n, name) for n in notes]


class VoiceMotion(Enum):
    PARALLEL = "parallel"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"
    SIMILAR = "similar"


class Species(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5


@dataclass
class Voice:
    name: str
    notes: list[tuple[float, int]]
    range_low: int = 36
    range_high: int = 84

    def __post_init__(self):
        self.range_low = validate_midi_note(self.range_low, "range_low")
        self.range_high = validate_midi_note(self.range_high, "range_high")
        self.notes = [(t, validate_midi_note(pitch)) for t, pitch in self.notes]

    def get_note_at_time(self, time: float) -> int | None:
        for t, pitch in self.notes:
            if abs(t - time) < 0.01:
                return pitch
        return None

    def is_in_range(self, pitch: int) -> bool:
        return self.range_low <= pitch <= self.range_high


@dataclass
class VoiceLeadingViolation:
    type: str
    description: str
    time: float
    voices: list[str]
    severity: int


class VoiceLeadingAnalyzer:
    def __init__(self):
        self.violations: list[VoiceLeadingViolation] = []

    def analyze_voices(
        self, voices: list[Voice]
    ) -> list[VoiceLeadingViolation]:
        self.violations = []

        timestamps = self._get_all_timestamps(voices)

        for i, t1 in enumerate(timestamps):
            self._check_voice_crossing(voices, t1)
            self._check_voice_ranges(voices, t1)
            if i + 1 < len(timestamps):
                t2 = timestamps[i + 1]
                self._check_parallel_fifths_octaves(voices, t1, t2)
                self._check_large_leaps(voices, t1, t2)

        return self.violations

    def _get_all_timestamps(self, voices: list[Voice]) -> list[float]:
        timestamps = set()
        for voice in voices:
            for time, _ in voice.notes:
                timestamps.add(time)
        return sorted(timestamps)

    def _check_parallel_fifths_octaves(
        self, voices: list[Voice], t1: float, t2: float
    ):
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                n1_i = voices[i].get_note_at_time(t1)
                n2_i = voices[i].get_note_at_time(t2)
                n1_j = voices[j].get_note_at_time(t1)
                n2_j = voices[j].get_note_at_time(t2)

                if all(x is not None for x in [n1_i, n2_i, n1_j, n2_j]):
                    interval1 = abs(n1_i - n1_j) % 12
                    interval2 = abs(n2_i - n2_j) % 12

                    if interval1 == interval2 and interval1 in [0, 7]:
                        motion = self._get_motion_type(n1_i, n2_i, n1_j, n2_j)
                        if motion == VoiceMotion.PARALLEL:
                            interval_name = (
                                "octaves" if interval1 == 0 else "fifths"
                            )
                            self.violations.append(
                                VoiceLeadingViolation(
                                    type=f"parallel_{interval_name}",
                                    description=f"Parallel {interval_name} between {voices[i].name} and {voices[j].name}",
                                    time=t1,
                                    voices=[voices[i].name, voices[j].name],
                                    severity=8,
                                )
                            )

    def _check_voice_crossing(self, voices: list[Voice], time: float):
        notes = []
        for voice in voices:
            note = voice.get_note_at_time(time)
            if note is not None:
                notes.append((voice.name, note))

        for i in range(len(notes) - 1):
            if notes[i][1] > notes[i + 1][1]:
                self.violations.append(
                    VoiceLeadingViolation(
                        type="voice_crossing",
                        description=f"Voice crossing between {notes[i][0]} and {notes[i + 1][0]}",
                        time=time,
                        voices=[notes[i][0], notes[i + 1][0]],
                        severity=5,
                    )
                )

    def _check_voice_ranges(self, voices: list[Voice], time: float):
        for voice in voices:
            note = voice.get_note_at_time(time)
            if note is not None and not voice.is_in_range(note):
                self.violations.append(
                    VoiceLeadingViolation(
                        type="out_of_range",
                        description=f"{voice.name} note {note} is out of range ({voice.range_low}-{voice.range_high})",
                        time=time,
                        voices=[voice.name],
                        severity=9,
                    )
                )

    def _check_large_leaps(self, voices: list[Voice], t1: float, t2: float):
        for voice in voices:
            n1 = voice.get_note_at_time(t1)
            n2 = voice.get_note_at_time(t2)

            if n1 is not None and n2 is not None:
                leap = abs(n2 - n1)
                if leap > 12:
                    self.violations.append(
                        VoiceLeadingViolation(
                            type="large_leap",
                            description=f"{voice.name} has a leap of {leap} semitones",
                            time=t1,
                            voices=[voice.name],
                            severity=4,
                        )
                    )

    def _get_motion_type(
        self, n1_voice1: int, n2_voice1: int, n1_voice2: int, n2_voice2: int
    ) -> VoiceMotion:
        dir1 = np.sign(n2_voice1 - n1_voice1)
        dir2 = np.sign(n2_voice2 - n1_voice2)

        if dir1 == 0 and dir2 == 0:
            return VoiceMotion.OBLIQUE
        elif dir1 == 0 or dir2 == 0:
            return VoiceMotion.OBLIQUE
        elif dir1 == dir2:
            if abs(n2_voice1 - n1_voice1) == abs(n2_voice2 - n1_voice2):
                return VoiceMotion.PARALLEL
            else:
                return VoiceMotion.SIMILAR
        else:
            return VoiceMotion.CONTRARY

    def optimize_voice_leading(
        self, chord_progression: list[list[int]], num_voices: int = 4
    ) -> list[list[int]]:
        if not chord_progression:
            return []

        validate_chord_progression(chord_progression)
        if isinstance(num_voices, bool) or not isinstance(
            num_voices, (int, np.integer)
        ):
            raise ValueError(
                f"Number of voices must be a whole number, got {num_voices!r}"
            )
        if not 1 <= num_voices <= MAX_VOICES:
            raise ValueError(
                f"Number of voices must be from 1 to {MAX_VOICES}, "
                f"got {num_voices!r}"
            )
        chord_progression = [
            [validate_midi_note(n, "chord note") for n in chord]
            for chord in chord_progression
        ]
        if any(not chord for chord in chord_progression):
            raise ValueError("Every chord must contain at least one note")

        voiced = []
        current_voicing = self._initialize_voicing(
            chord_progression[0], num_voices
        )
        voiced.append(current_voicing)

        for i in range(1, len(chord_progression)):
            next_chord = chord_progression[i]
            next_voicing = self._find_smoothest_voicing(
                current_voicing, next_chord
            )
            voiced.append(next_voicing)
            current_voicing = next_voicing

        return voiced

    def _initialize_voicing(
        self, chord: list[int], num_voices: int
    ) -> list[int]:
        voicing = []
        octave = 4

        for i in range(num_voices):
            note_index = i % len(chord)
            pitch = fold_into_midi_range(chord[note_index] + octave * 12)
            voicing.append(pitch)

            if (i + 1) % len(chord) == 0:
                octave += 1

        return sorted(voicing)

    def _find_smoothest_voicing(
        self, current: list[int], next_chord: list[int]
    ) -> list[int]:
        classes = sorted({note % 12 for note in next_chord})
        if not current or not classes:
            return current

        options = []
        for current_note in current:
            nearest = []
            for pitch_class in classes:
                candidates = [
                    fold_into_midi_range(pitch_class + octave * 12)
                    for octave in range(3, 7)
                ]
                nearest.append(
                    min(candidates, key=lambda n: abs(n - current_note))
                )
            options.append(nearest)

        best = {0: (0, [])}
        for voice_index, voice_options in enumerate(options):
            step = {}
            for mask, (movement, pitches) in best.items():
                for class_index, pitch in enumerate(voice_options):
                    new_mask = mask | (1 << class_index)
                    new_movement = movement + abs(pitch - current[voice_index])
                    known = step.get(new_mask)
                    if known is None or new_movement < known[0]:
                        step[new_mask] = (new_movement, pitches + [pitch])
            best = step

        chosen = min(
            best.items(),
            key=lambda item: (-bin(item[0]).count("1"), item[1][0]),
        )
        return chosen[1][1]


class CounterpointChecker:
    def __init__(self, species: Species = Species.FIRST):
        self.species = species
        self.rules_violated: list[str] = []

    def check_counterpoint(
        self, cantus_firmus: list[int], counterpoint: list[int]
    ) -> list[str]:
        cantus_firmus = _validate_pitch_list(cantus_firmus, "cantus firmus")
        counterpoint = _validate_pitch_list(counterpoint, "counterpoint")
        self.rules_violated = []

        self._check_consonant_intervals(cantus_firmus, counterpoint)
        self._check_starting_ending(cantus_firmus, counterpoint)
        self._check_melodic_motion(counterpoint)

        if self.species == Species.FIRST:
            self._check_note_against_note(cantus_firmus, counterpoint)

        return self.rules_violated

    def _check_consonant_intervals(
        self, cantus_firmus: list[int], counterpoint: list[int]
    ):
        consonant = {0, 3, 4, 5, 7, 8, 9}

        for i in range(min(len(cantus_firmus), len(counterpoint))):
            interval = abs(cantus_firmus[i] - counterpoint[i]) % 12
            if interval not in consonant:
                self.rules_violated.append(
                    f"Dissonant interval at position {i}: {interval} semitones"
                )

    def _check_starting_ending(
        self, cantus_firmus: list[int], counterpoint: list[int]
    ):
        if cantus_firmus and counterpoint:
            start_interval = abs(cantus_firmus[0] - counterpoint[0]) % 12
            if start_interval not in [0, 7]:
                self.rules_violated.append(
                    f"Must start with unison or perfect fifth, got {start_interval}"
                )

            end_interval = abs(cantus_firmus[-1] - counterpoint[-1]) % 12
            if end_interval != 0:
                self.rules_violated.append(
                    f"Must end with unison, got {end_interval}"
                )

    def _check_melodic_motion(self, melody: list[int]):
        for i in range(len(melody) - 1):
            leap = abs(melody[i + 1] - melody[i])
            if leap > 7:
                self.rules_violated.append(
                    f"Large leap of {leap} semitones at position {i}"
                )

    def _check_note_against_note(
        self, cantus_firmus: list[int], counterpoint: list[int]
    ):
        if len(cantus_firmus) != len(counterpoint):
            self.rules_violated.append(
                "First species requires equal number of notes in both voices"
            )


class FugueAnalyzer:
    @dataclass
    class Subject:
        notes: list[int]
        start_time: float
        duration: float

    def __init__(self):
        self.subjects: list[FugueAnalyzer.Subject] = []

    def detect_subject(
        self, melody: list[tuple[float, int]], min_length: int = 4
    ) -> Subject | None:
        if (
            isinstance(min_length, bool)
            or not isinstance(min_length, numbers.Integral)
            or not 1 <= min_length <= MAX_SUBJECT_LENGTH
        ):
            raise ValueError(
                "Subject length must be a whole number from 1 to "
                f"{MAX_SUBJECT_LENGTH}, got {min_length!r}"
            )
        melody = _validate_melody(melody)
        if len(melody) < min_length:
            return None

        notes = [pitch for _, pitch in melody[:min_length]]
        start_time = melody[0][0]
        end_time = melody[min_length - 1][0]

        subject = self.Subject(
            notes=notes, start_time=start_time, duration=end_time - start_time
        )

        return subject

    def find_subject_entries(
        self, subject: Subject, melody: list[tuple[float, int]]
    ) -> list[float]:
        melody = _validate_melody(melody)
        entries = []
        subject_length = len(subject.notes)

        for i in range(len(melody) - subject_length + 1):
            window = [pitch for _, pitch in melody[i : i + subject_length]]

            if self._is_transposition(subject.notes, window):
                entries.append(melody[i][0])

        return entries

    def _is_transposition(
        self, original: list[int], candidate: list[int]
    ) -> bool:
        if len(original) != len(candidate):
            return False

        if not original or not candidate:
            return False

        transposition = candidate[0] - original[0]

        for i in range(len(original)):
            if candidate[i] - original[i] != transposition:
                return False

        return True


class CanonDetector:
    def __init__(self):
        pass

    def detect_canon(
        self,
        voices: list[list[tuple[float, int]]],
        max_time_offset: float = 2.0,
    ) -> list[tuple[int, int, float, int]]:
        if (
            isinstance(max_time_offset, bool)
            or not isinstance(max_time_offset, numbers.Real)
            or not math.isfinite(max_time_offset)
            or not 0 <= max_time_offset <= MAX_CANON_OFFSET
        ):
            raise ValueError(
                "Maximum time offset must be a finite number from 0 to "
                f"{MAX_CANON_OFFSET}, got {max_time_offset!r}"
            )
        if not isinstance(voices, (list, tuple)):
            raise ValueError(f"The voices must be a list, got {voices!r}")
        voices = [_validate_melody(v, "voice") for v in voices]
        canons = []

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                offset, transposition = self._find_canon_relationship(
                    voices[i], voices[j], max_time_offset
                )

                if offset is not None:
                    canons.append((i, j, offset, transposition))

        return canons

    def _find_canon_relationship(
        self,
        voice1: list[tuple[float, int]],
        voice2: list[tuple[float, int]],
        max_offset: float,
    ) -> tuple[float | None, int | None]:
        if len(voice1) < 4 or len(voice2) < 4:
            return None, None

        for time_offset in np.arange(0, max_offset, 0.25):
            for transposition in range(-12, 13):
                if self._matches_canon(
                    voice1, voice2, time_offset, transposition
                ):
                    return time_offset, transposition

        return None, None

    def _matches_canon(
        self,
        voice1: list[tuple[float, int]],
        voice2: list[tuple[float, int]],
        time_offset: float,
        transposition: int,
    ) -> bool:
        matches = 0
        required_matches = min(len(voice1), len(voice2)) // 2

        for t1, p1 in voice1[:required_matches]:
            expected_time = t1 + time_offset
            expected_pitch = p1 + transposition

            for t2, p2 in voice2:
                if abs(t2 - expected_time) < 0.1 and p2 == expected_pitch:
                    matches += 1
                    break

        return matches >= required_matches * 0.8
