import pretty_midi
import pytest

from korshmer.midi_processing.chord_detection import (
    ChordDetector,
    ChordProgressionAnalyzer,
)

C_MAJOR_PROGRESSION = [
    [60, 64, 67],
    [65, 69, 72],
    [67, 71, 74],
    [60, 64, 67],
    [57, 60, 64],
    [62, 65, 69],
    [67, 71, 74, 77],
    [60, 64, 67],
]


def _transpose(progression, semitones):
    return [[note + semitones for note in chord] for chord in progression]


def _chords(progression):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for bar, chord in enumerate(progression):
        for pitch in chord:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=80,
                    pitch=pitch,
                    start=bar * 2.0,
                    end=bar * 2.0 + 1.9,
                )
            )
    midi.instruments.append(instrument)
    return [chord for _, chord in ChordDetector().detect_chords(midi)]


def test_progression_in_one_key_has_no_modulation():
    assert ChordDetector().detect_modulation(_chords(C_MAJOR_PROGRESSION)) == []


@pytest.mark.parametrize("shift", [2, 5, 7])
def test_progression_in_one_transposed_key_has_no_modulation(shift):
    chords = _chords(_transpose(C_MAJOR_PROGRESSION, shift))
    assert ChordDetector().detect_modulation(chords) == []


def test_key_change_is_reported_with_the_new_key():
    progression = C_MAJOR_PROGRESSION + _transpose(C_MAJOR_PROGRESSION, 2)
    modulations = ChordDetector().detect_modulation(_chords(progression))
    assert modulations
    assert modulations[-1][1] == 2
    assert modulations[-1][0] > 4


def test_analyzer_reports_no_modulation_for_one_key():
    notes_by_time = {
        bar * 2.0: chord for bar, chord in enumerate(C_MAJOR_PROGRESSION)
    }
    result = ChordProgressionAnalyzer().analyze_midi_notes(notes_by_time)
    assert result["modulations"] == []
