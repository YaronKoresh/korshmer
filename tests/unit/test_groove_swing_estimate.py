import pytest

from korshmer.midi_processing.quantization import GrooveExtractor, Note


def _note(start):
    return Note(start_time=start, end_time=start + 0.1, pitch=60, velocity=100)


def test_swung_offbeats_give_swing_amount():
    extractor = GrooveExtractor(bpm=120.0)
    notes = []
    for beat in range(8):
        start = beat * 0.5
        notes.append(_note(start))
        notes.append(_note(start + 0.25 + 0.04))

    groove = extractor.extract_from_midi(notes, reference_grid=0.25)

    assert groove.swing_amount == pytest.approx(0.64)


def test_straight_eighths_give_no_swing():
    extractor = GrooveExtractor(bpm=120.0)
    notes = [_note(step * 0.25) for step in range(16)]

    groove = extractor.extract_from_midi(notes, reference_grid=0.125)

    assert groove.swing_amount == 0.0


def test_late_on_beat_notes_are_not_swing():
    extractor = GrooveExtractor(bpm=120.0)
    notes = [_note(beat * 0.5 + 0.02) for beat in range(16)]

    groove = extractor.extract_from_midi(notes, reference_grid=0.125)

    assert groove.swing_amount == 0.0


def test_swing_estimate_follows_tempo():
    extractor = GrooveExtractor(bpm=60.0)
    notes = [_note(beat * 1.0 + 0.5 + 0.08) for beat in range(8)]

    groove = extractor.extract_from_midi(notes, reference_grid=0.25)

    assert groove.swing_amount == pytest.approx(0.64)
