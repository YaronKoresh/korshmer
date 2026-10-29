import sys

import numpy as np
import pretty_midi

from korshmer.midi_algorithms import normalize_note_velocities


def _midi(velocities):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i, v in enumerate(velocities):
        inst.notes.append(
            pretty_midi.Note(velocity=v, pitch=60 + i, start=i, end=i + 0.5)
        )
    midi.instruments.append(inst)
    return midi


def test_quiet_notes_are_raised_into_audible_range():
    midi = normalize_note_velocities(_midi([25, 27, 31, 40, 60, 90, 108]))
    velocities = [n.velocity for n in midi.instruments[0].notes]
    assert min(velocities) >= 48
    assert max(velocities) <= 118
    assert velocities == sorted(velocities)


def test_equal_velocities_get_one_middle_value():
    midi = normalize_note_velocities(_midi([30, 30, 30]))
    assert {n.velocity for n in midi.instruments[0].notes} == {83}


def test_empty_midi_is_returned_unchanged():
    midi = pretty_midi.PrettyMIDI()
    assert normalize_note_velocities(midi) is midi


def test_transcribe_with_confidence_normalizes_velocities(monkeypatch):
    class Dummy:
        pass

    class FakeInference:
        @staticmethod
        def predict(path, **kwargs):
            midi = _midi([25, 27, 30, 100])
            events = [
                {
                    "start_time_s": float(i),
                    "end_time_s": i + 0.5,
                    "pitch_midi": 60 + i,
                }
                for i in range(4)
            ]
            return np.ones((500, 128)), midi, events

    monkeypatch.setitem(sys.modules, "basic_pitch", Dummy())
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", FakeInference)
    from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI

    midi, _ = EnhancedAudioToMIDI().transcribe_with_confidence("x.wav")
    velocities = [n.velocity for n in midi.instruments[0].notes]
    assert min(velocities) >= 48
