from dataclasses import dataclass
from enum import Enum


class ChordQuality(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    SUS2 = "sus2"
    SUS4 = "sus4"
    ADD9 = "add9"
    DOMINANT7 = "dom7"
    MAJOR7 = "maj7"
    MINOR7 = "min7"
    DIMINISHED7 = "dim7"
    HALF_DIMINISHED7 = "half-dim7"
    AUGMENTED7 = "aug7"
    MAJOR9 = "maj9"
    MINOR9 = "min9"
    DOMINANT9 = "dom9"
    MAJOR11 = "maj11"
    MINOR11 = "min11"
    DOMINANT11 = "dom11"
    MAJOR13 = "maj13"
    MINOR13 = "min13"
    DOMINANT13 = "dom13"


class HarmonicFunction(Enum):
    TONIC = "tonic"
    SUBDOMINANT = "subdominant"
    DOMINANT = "dominant"
    MEDIANT = "mediant"
    SUBMEDIANT = "submediant"
    LEADING_TONE = "leading_tone"


class ChordVoicing(Enum):
    OPEN = "open"
    CLOSE = "close"
    DROP2 = "drop-2"
    DROP3 = "drop-3"
    ROOT_POSITION = "root"
    FIRST_INVERSION = "first"
    SECOND_INVERSION = "second"
    THIRD_INVERSION = "third"


@dataclass
class Chord:
    root: int
    quality: ChordQuality
    bass_note: int | None = None
    intervals: tuple[int, ...] = ()
    inversion: int = 0
    voicing: ChordVoicing | None = None
    complexity: float = 0.0

    def get_notes(self) -> list[int]:
        notes = [self.root]
        for interval in self.intervals:
            notes.append(self.root + interval)
        return sorted(notes)

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
        quality_str = self._get_quality_suffix()

        if self.bass_note is not None and self.bass_note != self.root:
            bass_name = note_names[self.bass_note % 12]
            return f"{root_name}{quality_str}/{bass_name}"
        return f"{root_name}{quality_str}"

    def _get_quality_suffix(self) -> str:
        quality_map = {
            ChordQuality.MAJOR: "",
            ChordQuality.MINOR: "m",
            ChordQuality.DIMINISHED: "dim",
            ChordQuality.AUGMENTED: "aug",
            ChordQuality.SUS2: "sus2",
            ChordQuality.SUS4: "sus4",
            ChordQuality.ADD9: "add9",
            ChordQuality.DOMINANT7: "7",
            ChordQuality.MAJOR7: "maj7",
            ChordQuality.MINOR7: "m7",
            ChordQuality.DIMINISHED7: "dim7",
            ChordQuality.HALF_DIMINISHED7: "m7b5",
            ChordQuality.AUGMENTED7: "aug7",
            ChordQuality.MAJOR9: "maj9",
            ChordQuality.MINOR9: "m9",
            ChordQuality.DOMINANT9: "9",
            ChordQuality.MAJOR11: "maj11",
            ChordQuality.MINOR11: "m11",
            ChordQuality.DOMINANT11: "11",
            ChordQuality.MAJOR13: "maj13",
            ChordQuality.MINOR13: "m13",
            ChordQuality.DOMINANT13: "13",
        }
        return quality_map.get(self.quality, "")


_MAJOR_SCALE_DEGREES = (0, 2, 4, 5, 7, 9, 11)


class ChordDetector:
    CHORD_TEMPLATES = {
        ChordQuality.MAJOR: (0, 4, 7),
        ChordQuality.MINOR: (0, 3, 7),
        ChordQuality.DIMINISHED: (0, 3, 6),
        ChordQuality.AUGMENTED: (0, 4, 8),
        ChordQuality.SUS2: (0, 2, 7),
        ChordQuality.SUS4: (0, 5, 7),
        ChordQuality.ADD9: (0, 4, 7, 14),
        ChordQuality.DOMINANT7: (0, 4, 7, 10),
        ChordQuality.MAJOR7: (0, 4, 7, 11),
        ChordQuality.MINOR7: (0, 3, 7, 10),
        ChordQuality.DIMINISHED7: (0, 3, 6, 9),
        ChordQuality.HALF_DIMINISHED7: (0, 3, 6, 10),
        ChordQuality.AUGMENTED7: (0, 4, 8, 10),
        ChordQuality.MAJOR9: (0, 4, 7, 11, 14),
        ChordQuality.MINOR9: (0, 3, 7, 10, 14),
        ChordQuality.DOMINANT9: (0, 4, 7, 10, 14),
        ChordQuality.MAJOR11: (0, 4, 7, 11, 14, 17),
        ChordQuality.MINOR11: (0, 3, 7, 10, 14, 17),
        ChordQuality.DOMINANT11: (0, 4, 7, 10, 14, 17),
        ChordQuality.MAJOR13: (0, 4, 7, 11, 14, 17, 21),
        ChordQuality.MINOR13: (0, 3, 7, 10, 14, 17, 21),
        ChordQuality.DOMINANT13: (0, 4, 7, 10, 14, 17, 21),
    }

    def __init__(self):
        self.detected_chords: list[Chord] = []

    def detect_chord_from_notes(self, notes: list[int]) -> Chord | None:
        if len(notes) < 2:
            return None

        unique_notes = sorted({n % 12 for n in notes})
        bass_note = min(notes) % 12

        best = None
        best_key = None
        for root in unique_notes:
            intervals = self._normalize_intervals(unique_notes, root)
            interval_set = set(intervals)

            for order, (quality, template) in enumerate(
                self.CHORD_TEMPLATES.items()
            ):
                if not self._match_intervals(intervals, template):
                    continue
                template_set = {i % 12 for i in template}

                key = (
                    template_set != interval_set,
                    -len(template_set),
                    root != bass_note,
                    order,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (root, quality, template)

        if best is not None:
            root, quality, template = best
            chord = Chord(
                root=root,
                quality=quality,
                bass_note=bass_note,
                intervals=template[1:],
                inversion=self._get_inversion(root, bass_note, template),
                complexity=self._calculate_complexity(quality, len(template)),
            )
            chord.voicing = self._analyze_voicing(notes)
            return chord

        if len(unique_notes) >= 3:
            intervals = self._normalize_intervals(unique_notes, unique_notes[0])
            return Chord(
                root=unique_notes[0],
                quality=ChordQuality.MAJOR,
                bass_note=bass_note,
                intervals=tuple(intervals[1:]),
                complexity=1.0,
            )

        return None

    def detect_chords(
        self, midi_data, window: float = 0.05
    ) -> list[tuple[float, Chord]]:
        notes = sorted(
            (note.start, note.pitch)
            for instrument in midi_data.instruments
            if not instrument.is_drum
            for note in instrument.notes
        )
        chords: list[tuple[float, Chord]] = []
        index = 0
        while index < len(notes):
            start = notes[index][0]
            group = []
            while index < len(notes) and notes[index][0] - start <= window:
                group.append(notes[index][1])
                index += 1
            chord = self.detect_chord_from_notes(group)
            if chord is not None:
                chords.append((start, chord))
        return chords

    def _normalize_intervals(self, notes: list[int], root: int) -> list[int]:
        return sorted({(n - root) % 12 for n in notes})

    def _match_intervals(
        self, intervals: list[int], template: tuple[int, ...]
    ) -> bool:
        template_set = {i % 12 for i in template}
        intervals_set = set(intervals)
        return template_set.issubset(intervals_set)

    def _get_inversion(
        self, root: int, bass: int, template: tuple[int, ...]
    ) -> int:
        if root == bass:
            return 0
        bass_interval = (bass - root) % 12
        try:
            return [i % 12 for i in template].index(bass_interval)
        except ValueError:
            return 0

    def _calculate_complexity(
        self, quality: ChordQuality, num_notes: int
    ) -> float:
        base_complexity = {
            ChordQuality.MAJOR: 0.1,
            ChordQuality.MINOR: 0.2,
            ChordQuality.DIMINISHED: 0.4,
            ChordQuality.AUGMENTED: 0.5,
            ChordQuality.SUS2: 0.3,
            ChordQuality.SUS4: 0.3,
            ChordQuality.ADD9: 0.6,
            ChordQuality.DOMINANT7: 0.5,
            ChordQuality.MAJOR7: 0.6,
            ChordQuality.MINOR7: 0.6,
            ChordQuality.DIMINISHED7: 0.8,
            ChordQuality.HALF_DIMINISHED7: 0.8,
            ChordQuality.AUGMENTED7: 0.9,
            ChordQuality.MAJOR9: 0.7,
            ChordQuality.MINOR9: 0.7,
            ChordQuality.DOMINANT9: 0.7,
            ChordQuality.MAJOR11: 0.8,
            ChordQuality.MINOR11: 0.8,
            ChordQuality.DOMINANT11: 0.8,
            ChordQuality.MAJOR13: 0.9,
            ChordQuality.MINOR13: 0.9,
            ChordQuality.DOMINANT13: 0.9,
        }
        return base_complexity.get(quality, 0.5) + (num_notes - 3) * 0.1

    def _analyze_voicing(self, notes: list[int]) -> ChordVoicing:
        if len(notes) < 3:
            return ChordVoicing.CLOSE

        sorted_notes = sorted(notes)
        intervals = [
            sorted_notes[i + 1] - sorted_notes[i]
            for i in range(len(sorted_notes) - 1)
        ]

        if all(interval <= 4 for interval in intervals):
            return ChordVoicing.CLOSE
        elif any(interval > 12 for interval in intervals):
            return ChordVoicing.OPEN

        return ChordVoicing.ROOT_POSITION

    def analyze_progression(
        self, chords: list[Chord], key: int = 0
    ) -> list[str]:
        roman_numerals = []

        for chord in chords:
            degree = (chord.root - key) % 12
            roman = self._get_roman_numeral(degree, chord.quality)
            roman_numerals.append(roman)

        return roman_numerals

    def _get_roman_numeral(self, degree: int, quality: ChordQuality) -> str:
        scale_degrees = {
            0: "I",
            1: "bII",
            2: "II",
            3: "bIII",
            4: "III",
            5: "IV",
            6: "bV",
            7: "V",
            8: "bVI",
            9: "VI",
            10: "bVII",
            11: "VII",
        }

        base_numeral = scale_degrees.get(degree, "?")

        if quality in [
            ChordQuality.MINOR,
            ChordQuality.MINOR7,
            ChordQuality.MINOR9,
            ChordQuality.MINOR11,
            ChordQuality.MINOR13,
        ]:
            base_numeral = base_numeral.lower()
        elif quality in [
            ChordQuality.DIMINISHED,
            ChordQuality.DIMINISHED7,
            ChordQuality.HALF_DIMINISHED7,
        ]:
            base_numeral = base_numeral.lower() + "°"
        elif quality == ChordQuality.AUGMENTED:
            base_numeral = base_numeral + "+"

        if (
            "7" in quality.value
            or "9" in quality.value
            or "11" in quality.value
            or "13" in quality.value
        ):
            if quality == ChordQuality.DOMINANT7:
                base_numeral += "7"
            elif quality == ChordQuality.MAJOR7:
                base_numeral += "maj7"
            elif quality == ChordQuality.MINOR7:
                base_numeral += "7"
            elif quality in [
                ChordQuality.MAJOR9,
                ChordQuality.MINOR9,
                ChordQuality.DOMINANT9,
            ]:
                base_numeral += "9"
            elif quality in [
                ChordQuality.MAJOR11,
                ChordQuality.MINOR11,
                ChordQuality.DOMINANT11,
            ]:
                base_numeral += "11"
            elif quality in [
                ChordQuality.MAJOR13,
                ChordQuality.MINOR13,
                ChordQuality.DOMINANT13,
            ]:
                base_numeral += "13"

        return base_numeral

    def get_harmonic_function(
        self, chord: Chord, key: int = 0
    ) -> HarmonicFunction:
        degree = (chord.root - key) % 12

        function_map = {
            0: HarmonicFunction.TONIC,
            2: HarmonicFunction.SUBDOMINANT,
            4: HarmonicFunction.MEDIANT,
            5: HarmonicFunction.SUBDOMINANT,
            7: HarmonicFunction.DOMINANT,
            9: HarmonicFunction.SUBMEDIANT,
            11: HarmonicFunction.LEADING_TONE,
        }

        return function_map.get(degree, HarmonicFunction.TONIC)

    def detect_modulation(self, chords: list[Chord]) -> list[tuple[int, int]]:
        modulations = []

        if len(chords) < 4:
            return modulations

        segments = []
        for i in range(len(chords) - 3):
            keys = set(self._estimate_keys(chords[i : i + 4]))
            if segments and segments[-1][1] & keys:
                segments[-1][1] &= keys
            else:
                segments.append([i, keys])

        for start, keys in segments[1:]:
            modulations.append((start, min(keys)))

        return modulations

    def _estimate_keys(self, chords: list[Chord]) -> list[int]:
        candidates = {}

        for key in range(12):
            score = 0
            for chord in chords:
                if (chord.root - key) % 12 not in _MAJOR_SCALE_DEGREES:
                    continue
                function = self.get_harmonic_function(chord, key)
                if function in [
                    HarmonicFunction.TONIC,
                    HarmonicFunction.DOMINANT,
                ]:
                    score += 2
                elif function in [
                    HarmonicFunction.SUBDOMINANT,
                    HarmonicFunction.MEDIANT,
                    HarmonicFunction.SUBMEDIANT,
                    HarmonicFunction.LEADING_TONE,
                ]:
                    score += 1
            candidates[key] = score

        max_score = max(candidates.values()) if candidates else 0
        return [k for k, v in candidates.items() if v == max_score]

    def suggest_substitutions(self, chord: Chord, key: int = 0) -> list[Chord]:
        substitutions = []

        parallel_quality = {
            ChordQuality.MAJOR: ChordQuality.MINOR,
            ChordQuality.MINOR: ChordQuality.MAJOR,
        }
        relative = {
            ChordQuality.MAJOR: (-3, ChordQuality.MINOR),
            ChordQuality.MAJOR7: (-3, ChordQuality.MINOR),
            ChordQuality.MINOR: (3, ChordQuality.MAJOR),
            ChordQuality.MINOR7: (3, ChordQuality.MAJOR),
        }

        if chord.quality in parallel_quality:
            substitutions.append(
                Chord(
                    root=chord.root,
                    quality=parallel_quality[chord.quality],
                    intervals=self.CHORD_TEMPLATES[
                        parallel_quality[chord.quality]
                    ][1:],
                )
            )

        if chord.quality in relative:
            shift, quality = relative[chord.quality]
            substitutions.append(
                Chord(
                    root=(chord.root + shift) % 12,
                    quality=quality,
                    intervals=self.CHORD_TEMPLATES[quality][1:],
                )
            )

        if chord.quality == ChordQuality.DOMINANT7:
            substitutions.append(
                Chord(
                    root=(chord.root + 6) % 12,
                    quality=ChordQuality.DOMINANT7,
                    intervals=self.CHORD_TEMPLATES[ChordQuality.DOMINANT7][1:],
                )
            )

        return substitutions


class ChordProgressionAnalyzer:
    def __init__(self):
        self.detector = ChordDetector()

    def analyze_midi_notes(
        self, notes_by_time: dict[float, list[int]], key: int = 0
    ) -> dict:
        chords = []
        timestamps = []

        for timestamp, notes in sorted(notes_by_time.items()):
            chord = self.detector.detect_chord_from_notes(notes)
            if chord:
                chords.append(chord)
                timestamps.append(timestamp)

        roman_numerals = self.detector.analyze_progression(chords, key)
        functions = [
            self.detector.get_harmonic_function(c, key) for c in chords
        ]
        modulations = self.detector.detect_modulation(chords)

        return {
            "chords": chords,
            "timestamps": timestamps,
            "roman_numerals": roman_numerals,
            "harmonic_functions": functions,
            "modulations": modulations,
            "complexity_scores": [c.complexity for c in chords],
        }
