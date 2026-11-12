import numpy as np
import pretty_midi

from korshmer.midi.transcription_cleanup import (
    merge_sustained_splits,
    pitch_band_level,
)

SR = 22050
PITCH = 57


def _tone(seconds, envelope=None):
    t = np.arange(int(seconds * SR)) / SR
    wave = 0.4 * np.sin(2 * np.pi * pretty_midi.note_number_to_hz(PITCH) * t)
    return wave if envelope is None else wave * envelope(t)


def _pieces(bounds, pitch=PITCH, drum=False):
    inst = pretty_midi.Instrument(0, is_drum=drum)
    for start, end in zip(bounds, bounds[1:], strict=False):
        inst.notes.append(pretty_midi.Note(80, pitch, start, end))
    return inst


def test_one_held_note_cut_into_pieces_becomes_one_note_again():
    level, fps = pitch_band_level(_tone(3.0), SR)
    inst = _pieces([0.5, 1.0, 1.5, 2.0, 2.5])
    assert merge_sustained_splits(inst, level, fps) == 3
    assert [(n.start, n.end) for n in inst.notes] == [(0.5, 2.5)]


def test_notes_that_are_struck_again_stay_apart():
    struck = _tone(3.0, lambda t: np.exp(-8.0 * ((t - 0.5) % 0.5)))
    level, fps = pitch_band_level(struck, SR)
    inst = _pieces([0.5, 1.0, 1.5, 2.0, 2.5])
    assert merge_sustained_splits(inst, level, fps) == 0
    assert len(inst.notes) == 4


def test_other_pitches_gaps_and_drums_are_left_alone():
    level, fps = pitch_band_level(_tone(3.0), SR)
    apart = pretty_midi.Instrument(0)
    apart.notes = [
        pretty_midi.Note(80, PITCH, 0.5, 1.0),
        pretty_midi.Note(80, PITCH, 1.3, 1.8),
        pretty_midi.Note(80, PITCH + 4, 1.8, 2.3),
    ]
    assert merge_sustained_splits(apart, level, fps) == 0
    drums = _pieces([0.5, 1.0, 1.5], drum=True)
    assert merge_sustained_splits(drums, level, fps) == 0
    assert merge_sustained_splits(pretty_midi.Instrument(0), level, fps) == 0
