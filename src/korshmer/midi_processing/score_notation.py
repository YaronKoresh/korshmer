from dataclasses import dataclass
from enum import Enum

from korshmer.audio_processing.validation import validate_time_signature

from .note_validation import validate_midi_note


class Clef(Enum):
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    TENOR = "tenor"
    PERCUSSION = "percussion"


class Accidental(Enum):
    DOUBLE_FLAT = "double_flat"
    FLAT = "flat"
    NATURAL = "natural"
    SHARP = "sharp"
    DOUBLE_SHARP = "double_sharp"


class Articulation(Enum):
    STACCATO = "staccato"
    ACCENT = "accent"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    FERMATA = "fermata"


class Ornament(Enum):
    TRILL = "trill"
    MORDENT = "mordent"
    TURN = "turn"
    APPOGGIATURA = "appoggiatura"
    ACCIACCATURA = "acciaccatura"


class Dynamic(Enum):
    PPP = "ppp"
    PP = "pp"
    P = "p"
    MP = "mp"
    MF = "mf"
    F = "f"
    FF = "ff"
    FFF = "fff"


@dataclass
class NoteDuration:
    whole: float = 4.0
    half: float = 2.0
    quarter: float = 1.0
    eighth: float = 0.5
    sixteenth: float = 0.25
    thirty_second: float = 0.125


@dataclass
class ScoreNote:
    pitch: int
    duration: float
    start_time: float
    velocity: int
    accidental: Accidental | None = None
    articulations: list[Articulation] = None
    ornament: Ornament | None = None
    is_tied: bool = False
    tuplet_ratio: tuple[int, int] | None = None

    def __post_init__(self):
        self.pitch = validate_midi_note(self.pitch)
        if self.articulations is None:
            self.articulations = []

    def get_staff_position(self, clef: Clef) -> int:
        if clef == Clef.TREBLE:
            return self.pitch - 59
        elif clef == Clef.BASS:
            return self.pitch - 47
        elif clef == Clef.ALTO:
            return self.pitch - 53
        elif clef == Clef.TENOR:
            return self.pitch - 50
        return 0

    def get_note_name(self) -> str:
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
        return note_names[self.pitch % 12]

    def get_octave(self) -> int:
        return (self.pitch // 12) - 1


@dataclass
class Measure:
    number: int
    notes: list[ScoreNote]
    time_signature: tuple[int, int]
    key_signature: int
    dynamics: list[tuple[float, Dynamic]]
    tempo: float | None = None

    def __post_init__(self):
        self.time_signature = validate_time_signature(self.time_signature)

    def get_duration(self) -> float:
        numerator, denominator = self.time_signature
        return numerator * (4.0 / denominator)


class ScoreRenderer:
    def __init__(self):
        self.clef = Clef.TREBLE
        self.key_signature = 0
        self.time_signature = validate_time_signature((4, 4))

    def render_note(self, note: ScoreNote) -> dict:
        staff_pos = note.get_staff_position(self.clef)

        return {
            "pitch": note.pitch,
            "staff_position": staff_pos,
            "duration": note.duration,
            "accidental": note.accidental.value if note.accidental else None,
            "articulations": [a.value for a in note.articulations],
            "ornament": note.ornament.value if note.ornament else None,
            "tied": note.is_tied,
        }

    def render_measure(self, measure: Measure) -> dict:
        rendered_notes = [self.render_note(note) for note in measure.notes]

        return {
            "number": measure.number,
            "notes": rendered_notes,
            "time_signature": measure.time_signature,
            "key_signature": measure.key_signature,
            "dynamics": [(t, d.value) for t, d in measure.dynamics],
        }


class BeamingRules:
    def __init__(self):
        self.beaming_groups = {
            (4, 4): [
                [0, 0.5],
                [0.5, 1.0],
                [1.0, 1.5],
                [1.5, 2.0],
                [2.0, 2.5],
                [2.5, 3.0],
                [3.0, 3.5],
                [3.5, 4.0],
            ],
            (3, 4): [
                [0, 0.5],
                [0.5, 1.0],
                [1.0, 1.5],
                [1.5, 2.0],
                [2.0, 2.5],
                [2.5, 3.0],
            ],
            (6, 8): [[0, 1.5], [1.5, 3.0]],
        }

    def apply_beaming(
        self, notes: list[ScoreNote], time_signature: tuple[int, int]
    ) -> list[list[ScoreNote]]:
        groups = self.beaming_groups.get(time_signature, [[0, 4.0]])

        beamed_groups = []
        for start, end in groups:
            group = [
                n
                for n in notes
                if start <= n.start_time % 4.0 < end and n.duration <= 0.5
            ]
            if group:
                beamed_groups.append(group)

        return beamed_groups


class TupletDetector:
    def __init__(self):
        pass

    def detect_tuplets(
        self, notes: list[ScoreNote], _measure_duration: float = 4.0
    ) -> list[list[ScoreNote]]:
        tuplets = []

        i = 0
        while i < len(notes):
            if self._is_triplet_start(notes[i : i + 3]):
                triplet = notes[i : i + 3]
                for note in triplet:
                    note.tuplet_ratio = (3, 2)
                tuplets.append(triplet)
                i += 3
            else:
                i += 1

        return tuplets

    def _is_triplet_start(self, notes: list[ScoreNote]) -> bool:
        if len(notes) < 3:
            return False

        triplet_notes = notes[:3]
        durations = [n.duration for n in triplet_notes]

        base_duration = durations[0]
        if not all(abs(d - base_duration) < 0.01 for d in durations):
            return False

        start_time = triplet_notes[0].start_time
        end_time = triplet_notes[-1].start_time + triplet_notes[-1].duration
        total_span = end_time - start_time

        common_base_durations = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

        duration_tolerance = 0.01
        span_tolerance = 0.02

        for larger_value in common_base_durations:
            expected_span = 2.0 * larger_value
            expected_triplet_note_duration = (2.0 * larger_value) / 3.0

            if (
                abs(total_span - expected_span) <= span_tolerance
                and abs(base_duration - expected_triplet_note_duration)
                <= duration_tolerance
            ):
                return True

        return False


class GraceNoteHandler:
    APPOGGIATURA_MAX = 0.125
    ACCIACCATURA_LENGTH = 0.03125
    MAX_SHARE = 0.25
    VELOCITY_DROP = {
        Ornament.APPOGGIATURA: 10,
        Ornament.ACCIACCATURA: 15,
    }

    def __init__(self):
        pass

    def _add_grace(
        self,
        main_note: ScoreNote,
        grace_pitch: int,
        length: float,
        ornament: Ornament,
    ) -> tuple[ScoreNote, ScoreNote]:
        grace_duration = min(length, main_note.duration * self.MAX_SHARE)
        velocity = max(1, main_note.velocity - self.VELOCITY_DROP[ornament])

        grace_note = ScoreNote(
            pitch=grace_pitch,
            duration=grace_duration,
            start_time=main_note.start_time,
            velocity=velocity,
            ornament=ornament,
        )

        main_note.start_time += grace_duration
        main_note.duration -= grace_duration

        return grace_note, main_note

    def add_appoggiatura(
        self, main_note: ScoreNote, grace_pitch: int
    ) -> tuple[ScoreNote, ScoreNote]:
        return self._add_grace(
            main_note,
            grace_pitch,
            self.APPOGGIATURA_MAX,
            Ornament.APPOGGIATURA,
        )

    def add_acciaccatura(
        self, main_note: ScoreNote, grace_pitch: int
    ) -> tuple[ScoreNote, ScoreNote]:
        return self._add_grace(
            main_note,
            grace_pitch,
            self.ACCIACCATURA_LENGTH,
            Ornament.ACCIACCATURA,
        )


class DynamicsMapper:
    VELOCITY_MAP = {
        Dynamic.PPP: 16,
        Dynamic.PP: 33,
        Dynamic.P: 49,
        Dynamic.MP: 64,
        Dynamic.MF: 80,
        Dynamic.F: 96,
        Dynamic.FF: 112,
        Dynamic.FFF: 127,
    }

    def __init__(self):
        pass

    def velocity_to_dynamic(self, velocity: int) -> Dynamic:
        for dynamic, vel in sorted(
            self.VELOCITY_MAP.items(), key=lambda x: x[1]
        ):
            if velocity <= vel:
                return dynamic
        return Dynamic.FFF

    def dynamic_to_velocity(self, dynamic: Dynamic) -> int:
        return self.VELOCITY_MAP.get(dynamic, 80)

    def apply_crescendo(
        self,
        notes: list[ScoreNote],
        start_dynamic: Dynamic,
        end_dynamic: Dynamic,
    ):
        start_vel = self.dynamic_to_velocity(start_dynamic)
        end_vel = self.dynamic_to_velocity(end_dynamic)

        for i, note in enumerate(notes):
            t = i / max(1, len(notes) - 1)
            note.velocity = int(start_vel + t * (end_vel - start_vel))


class SlurTieHandler:
    def __init__(self):
        pass

    def add_slur(self, notes: list[ScoreNote]) -> list[ScoreNote]:
        for note in notes:
            note.articulations.append(Articulation.TENUTO)
        return notes

    def add_tie(self, note1: ScoreNote, note2: ScoreNote) -> ScoreNote:
        if note1.pitch != note2.pitch:
            return note1

        note1.duration += note2.duration
        note1.is_tied = True

        return note1

    def detect_ties(self, notes: list[ScoreNote]) -> list[tuple[int, int]]:
        ties = []

        for i in range(len(notes) - 1):
            if notes[i].pitch == notes[i + 1].pitch:
                time_gap = notes[i + 1].start_time - (
                    notes[i].start_time + notes[i].duration
                )
                if abs(time_gap) < 0.01:
                    ties.append((i, i + 1))

        return ties


class ChordSymbolGenerator:
    SUFFIXES = {
        (0, 4, 7): "",
        (0, 3, 7): "m",
        (0, 3, 6): "dim",
        (0, 4, 8): "aug",
        (0, 2, 7): "sus2",
        (0, 5, 7): "sus4",
        (0, 4, 7, 10): "7",
        (0, 4, 7, 11): "maj7",
        (0, 3, 7, 10): "m7",
        (0, 3, 7, 11): "mMaj7",
        (0, 3, 6, 9): "dim7",
        (0, 3, 6, 10): "m7b5",
        (0, 4, 8, 10): "aug7",
        (0, 4, 7, 9): "6",
        (0, 3, 7, 9): "m6",
        (0, 2, 4, 7): "add9",
        (0, 2, 3, 7): "madd9",
        (0, 2, 4, 7, 10): "9",
        (0, 2, 4, 7, 11): "maj9",
        (0, 2, 3, 7, 10): "m9",
    }

    def __init__(self):
        pass

    def generate_symbol(self, notes: list[int], root: int) -> str:
        if not notes:
            return ""

        root = validate_midi_note(root, "root")
        notes = [validate_midi_note(n, "note") for n in notes]
        intervals = sorted({(n - root) % 12 for n in notes})

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
        root_name = note_names[root % 12]

        suffix = self.SUFFIXES.get(tuple(intervals))
        if suffix is None:
            return f"{root_name}?"
        return f"{root_name}{suffix}"


class GrandStaffRenderer:
    def __init__(self):
        self.treble_clef = Clef.TREBLE
        self.bass_clef = Clef.BASS
        self.split_point = 60

    def split_notes(
        self, notes: list[ScoreNote]
    ) -> tuple[list[ScoreNote], list[ScoreNote]]:
        treble_notes = [n for n in notes if n.pitch >= self.split_point]
        bass_notes = [n for n in notes if n.pitch < self.split_point]

        return treble_notes, bass_notes

    def render_grand_staff(self, measures: list[Measure]) -> dict:
        treble_measures = []
        bass_measures = []

        for measure in measures:
            treble_notes, bass_notes = self.split_notes(measure.notes)

            treble_measures.append(
                Measure(
                    number=measure.number,
                    notes=treble_notes,
                    time_signature=measure.time_signature,
                    key_signature=measure.key_signature,
                    dynamics=measure.dynamics,
                )
            )

            bass_measures.append(
                Measure(
                    number=measure.number,
                    notes=bass_notes,
                    time_signature=measure.time_signature,
                    key_signature=measure.key_signature,
                    dynamics=measure.dynamics,
                )
            )

        return {"treble": treble_measures, "bass": bass_measures}


class MusicXMLExporter:
    DIVISIONS = 480
    MIN_FIFTHS = -7
    MAX_FIFTHS = 7
    TOLERANCE = 1e-6
    SHARP_SPELLING = (
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    )
    FLAT_SPELLING = (
        ("C", 0),
        ("D", -1),
        ("D", 0),
        ("E", -1),
        ("E", 0),
        ("F", 0),
        ("G", -1),
        ("G", 0),
        ("A", -1),
        ("A", 0),
        ("B", -1),
        ("B", 0),
    )
    NOTE_TYPES = (
        (4.0, "whole"),
        (2.0, "half"),
        (1.0, "quarter"),
        (0.5, "eighth"),
        (0.25, "16th"),
        (0.125, "32nd"),
    )

    def __init__(self):
        pass

    def export_to_musicxml(self, measures: list[Measure]) -> str:
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<score-partwise version="3.1">')
        xml_parts.append("  <part-list>")
        xml_parts.append('    <score-part id="P1">')
        xml_parts.append("      <part-name>Piano</part-name>")
        xml_parts.append("    </score-part>")
        xml_parts.append("  </part-list>")
        xml_parts.append('  <part id="P1">')

        previous_time = previous_key = None
        for index, measure in enumerate(measures):
            fifths = self._fifths(measure.key_signature)
            time_signature = tuple(measure.time_signature)
            xml_parts.append(f'    <measure number="{measure.number}">')
            xml_parts.extend(
                self._attributes(
                    time_signature,
                    fifths,
                    first=index == 0,
                    new_time=time_signature != previous_time,
                    new_key=fifths != previous_key,
                )
            )
            previous_time = time_signature
            previous_key = fifths
            xml_parts.extend(self._measure_body(measure, fifths))
            xml_parts.append("    </measure>")

        xml_parts.append("  </part>")
        xml_parts.append("</score-partwise>")

        return "\n".join(xml_parts)

    def _fifths(self, key_signature) -> int:
        if (
            isinstance(key_signature, bool)
            or not isinstance(key_signature, int)
            or not self.MIN_FIFTHS <= key_signature <= self.MAX_FIFTHS
        ):
            raise ValueError(
                "Key signature must be a whole number from "
                f"{self.MIN_FIFTHS} to {self.MAX_FIFTHS}, got "
                f"{key_signature!r}"
            )
        return key_signature

    def _attributes(self, time_signature, fifths, first, new_time, new_key):
        lines = ["      <attributes>"]
        if first:
            lines.append(f"        <divisions>{self.DIVISIONS}</divisions>")
        if new_key:
            lines.append(f"        <key><fifths>{fifths}</fifths></key>")
        if new_time:
            beats, beat_type = time_signature
            lines.append(
                f"        <time><beats>{beats}</beats>"
                f"<beat-type>{beat_type}</beat-type></time>"
            )
        if first:
            lines.append("        <clef><sign>G</sign><line>2</line></clef>")
        if len(lines) == 1:
            return []
        lines.append("      </attributes>")
        return lines

    def _divisions(self, beats: float) -> int:
        return max(1, round(beats * self.DIVISIONS))

    def _voices(self, measure):
        length = measure.get_duration()
        voices = []
        ordered = sorted(measure.notes, key=lambda n: (n.start_time, n.pitch))
        for note in ordered:
            start = max(0.0, note.start_time)
            if start >= length - self.TOLERANCE:
                continue
            duration = min(note.duration, length - start)
            if duration <= self.TOLERANCE:
                continue
            for voice in voices:
                last = voice[-1]
                if (
                    abs(last["start"] - start) <= self.TOLERANCE
                    and abs(last["duration"] - duration) <= self.TOLERANCE
                ):
                    last["notes"].append(note)
                    break
                if start >= last["start"] + last["duration"] - self.TOLERANCE:
                    voice.append(
                        {"start": start, "duration": duration, "notes": [note]}
                    )
                    break
            else:
                voices.append(
                    [{"start": start, "duration": duration, "notes": [note]}]
                )
        return voices

    def _measure_body(self, measure, fifths):
        length = self._divisions(measure.get_duration())
        voices = self._voices(measure) or [[]]
        lines = []
        for number, voice in enumerate(voices, start=1):
            cursor = 0
            for group in voice:
                start = min(self._divisions(group["start"]), length)
                if group["start"] <= 0:
                    start = 0
                if start > cursor:
                    lines.extend(self._rest(start - cursor, number))
                    cursor = start
                duration = min(
                    self._divisions(group["duration"]), length - cursor
                )
                if duration <= 0:
                    continue
                for index, note in enumerate(group["notes"]):
                    lines.extend(
                        self._note(
                            note, duration, number, fifths, chord=index > 0
                        )
                    )
                cursor += duration
            if cursor < length:
                lines.extend(self._rest(length - cursor, number))
            if number < len(voices):
                lines.append(
                    f"      <backup><duration>{length}</duration></backup>"
                )
        return lines

    def _rest(self, duration, voice):
        return [
            "      <note>",
            "        <rest/>",
            f"        <duration>{duration}</duration>",
            f"        <voice>{voice}</voice>",
            "      </note>",
        ]

    def _note_type(self, duration):
        beats = duration / self.DIVISIONS
        for value, name in self.NOTE_TYPES:
            if abs(beats - value) <= self.TOLERANCE:
                return name, False
            if abs(beats - value * 1.5) <= self.TOLERANCE:
                return name, True
        return None, False

    def _note(self, note, duration, voice, fifths, chord):
        spelling = self.SHARP_SPELLING if fifths >= 0 else self.FLAT_SPELLING
        step, alter = spelling[note.pitch % 12]
        lines = ["      <note>"]
        if chord:
            lines.append("        <chord/>")
        lines.append("        <pitch>")
        lines.append(f"          <step>{step}</step>")
        if alter:
            lines.append(f"          <alter>{alter}</alter>")
        lines.append(f"          <octave>{note.get_octave()}</octave>")
        lines.append("        </pitch>")
        lines.append(f"        <duration>{duration}</duration>")
        lines.append(f"        <voice>{voice}</voice>")
        name, dotted = self._note_type(duration)
        if name:
            lines.append(f"        <type>{name}</type>")
            if dotted:
                lines.append("        <dot/>")
        lines.append("      </note>")
        return lines
