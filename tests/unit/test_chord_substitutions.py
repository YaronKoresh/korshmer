import pytest

from korshmer.midi_processing.chord_detection import (
    Chord,
    ChordDetector,
    ChordQuality,
)

DETECTOR = ChordDetector()


def _chord(root, quality):
    return Chord(
        root=root,
        quality=quality,
        intervals=DETECTOR.CHORD_TEMPLATES[quality][1:],
    )


def _names(chord):
    return [c.get_name() for c in DETECTOR.suggest_substitutions(chord)]


def test_major_chord_gets_parallel_and_relative_minor():
    assert _names(_chord(0, ChordQuality.MAJOR)) == ["Cm", "Am"]


def test_minor_chord_gets_parallel_major_and_relative_major():
    assert _names(_chord(9, ChordQuality.MINOR)) == ["A", "C"]


def test_minor_chord_never_gets_a_relative_minor():
    for root in range(12):
        subs = DETECTOR.suggest_substitutions(_chord(root, ChordQuality.MINOR))
        assert subs[1].quality is ChordQuality.MAJOR
        assert subs[1].root == (root + 3) % 12


def test_major_seventh_gets_the_relative_minor():
    assert _names(_chord(0, ChordQuality.MAJOR7)) == ["Am"]


def test_minor_seventh_gets_the_relative_major():
    assert _names(_chord(9, ChordQuality.MINOR7)) == ["C"]


def test_dominant_seventh_gets_the_tritone_substitution():
    assert _names(_chord(7, ChordQuality.DOMINANT7)) == ["C#7"]


@pytest.mark.parametrize(
    "quality",
    [
        ChordQuality.DIMINISHED,
        ChordQuality.AUGMENTED,
        ChordQuality.SUS2,
        ChordQuality.SUS4,
        ChordQuality.DIMINISHED7,
    ],
)
def test_other_qualities_get_no_substitutions(quality):
    assert DETECTOR.suggest_substitutions(_chord(0, quality)) == []


def test_every_substitution_shares_notes_with_the_chord():
    chord = _chord(0, ChordQuality.MAJOR)
    for sub in DETECTOR.suggest_substitutions(chord):
        shared = {n % 12 for n in sub.get_notes()} & {
            n % 12 for n in chord.get_notes()
        }
        assert len(shared) >= 2
