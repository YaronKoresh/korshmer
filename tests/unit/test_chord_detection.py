import pretty_midi
import pytest

from korshmer.midi_processing.chord_detection import (
    ChordDetector,
    ChordProgressionAnalyzer,
    ChordQuality,
)
from korshmer.midi_processors import process_chord_detection


def detect(notes):
    return ChordDetector().detect_chord_from_notes(notes)


class TestChordNames:
    @pytest.mark.parametrize(
        "notes, name",
        [
            ([60, 64, 67], "C"),
            ([60, 63, 67], "Cm"),
            ([59, 62, 65], "Bdim"),
            ([60, 64, 68], "Caug"),
            ([60, 65, 67], "Csus4"),
            ([60, 62, 67], "Csus2"),
            ([55, 59, 62, 65], "G7"),
            ([60, 64, 67, 71], "Cmaj7"),
            ([57, 60, 64, 67], "Am7"),
            ([60, 63, 66, 69], "Cdim7"),
            ([53, 56, 59, 63], "Fm7b5"),
            ([60, 64, 67, 71, 74], "Cmaj9"),
            ([60, 63, 67, 70, 74], "Cm9"),
            ([60, 64, 67, 70, 74], "C9"),
        ],
    )
    def test_root_position_chords_are_named_without_a_slash(self, notes, name):
        assert detect(notes).get_name() == name

    @pytest.mark.parametrize(
        "notes, name, inversion",
        [
            ([52, 60, 67], "C/E", 1),
            ([55, 60, 64], "C/G", 2),
            ([53, 62, 65, 69], "Dm/F", 1),
            ([58, 60, 64, 67], "C7/A#", 3),
        ],
    )
    def test_inversions_keep_their_bass_note(self, notes, name, inversion):
        chord = detect(notes)
        assert chord.get_name() == name
        assert chord.inversion == inversion

    def test_the_bass_note_is_a_pitch_class(self):
        chord = detect([48, 64, 67])
        assert chord.bass_note == 0
        assert chord.get_name() == "C"


class TestChordQuality:
    @pytest.mark.parametrize(
        "notes, quality",
        [
            ([60, 64, 67, 71], ChordQuality.MAJOR7),
            ([55, 59, 62, 65], ChordQuality.DOMINANT7),
            ([57, 60, 64, 67], ChordQuality.MINOR7),
            ([60, 63, 66, 69], ChordQuality.DIMINISHED7),
            ([53, 56, 59, 63], ChordQuality.HALF_DIMINISHED7),
            ([60, 64, 67, 74], ChordQuality.ADD9),
        ],
    )
    def test_extended_chords_are_not_reduced_to_triads(self, notes, quality):
        assert detect(notes).quality == quality

    def test_octave_doubling_and_note_order_do_not_matter(self):
        assert detect([72, 60, 64, 67, 60]).get_name() == "C"

    def test_too_few_notes_give_no_chord(self):
        assert detect([]) is None
        assert detect([60]) is None
        assert detect([60, 67]) is None


class TestProgression:
    def test_progression_analysis_uses_the_seventh_chords(self):
        analysis = ChordProgressionAnalyzer().analyze_midi_notes(
            {0.0: [60, 64, 67, 71], 1.0: [55, 59, 62, 65]}
        )
        names = [chord.get_name() for chord in analysis["chords"]]
        assert names == ["Cmaj7", "G7"]


class TestMidiFileDetection:
    def _midi(self, chords, drums=False):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        for start, notes in chords:
            for pitch in notes:
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=90, pitch=pitch, start=start, end=start + 0.9
                    )
                )
        midi.instruments.append(piano)
        if drums:
            kit = pretty_midi.Instrument(program=0, is_drum=True)
            kit.notes.append(
                pretty_midi.Note(velocity=100, pitch=38, start=0.0, end=0.1)
            )
            midi.instruments.append(kit)
        return midi

    def test_detect_chords_groups_simultaneous_notes(self):
        midi = self._midi(
            [
                (0.0, [60, 64, 67]),
                (1.0, [57, 60, 64, 67]),
                (2.0, [55, 59, 62, 65]),
            ],
            drums=True,
        )
        found = ChordDetector().detect_chords(midi)
        assert [(t, c.get_name()) for t, c in found] == [
            (0.0, "C"),
            (1.0, "Am7"),
            (2.0, "G7"),
        ]

    def test_single_notes_are_not_chords(self):
        midi = self._midi([(0.0, [60]), (1.0, [62])])
        assert ChordDetector().detect_chords(midi) == []

    def test_the_tool_writes_one_line_per_chord(self, tmp_path):
        midi = self._midi([(0.0, [60, 64, 67, 71]), (2.0, [55, 59, 62, 65])])
        path = tmp_path / "song.mid"
        midi.write(str(path))

        out = process_chord_detection(str(path))

        lines = open(out, encoding="utf-8").read().splitlines()
        assert lines[0] == "Chord Detection Results for: song"
        body = [line for line in lines[2:] if line.strip()]
        assert [line.split()[-1] for line in body] == ["Cmaj7", "G7"]
