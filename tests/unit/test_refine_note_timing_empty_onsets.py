from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pretty_midi


def _make_midi_with_notes():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=80, pitch=60, start=0.5, end=1.0)
    )
    inst.notes.append(
        pretty_midi.Note(velocity=90, pitch=64, start=1.5, end=2.0)
    )
    pm.instruments.append(inst)
    return pm


def test_refine_note_timing_returns_unchanged_when_no_onsets():
    from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI

    with patch.object(EnhancedAudioToMIDI, "__init__", lambda self: None):
        transcriber = EnhancedAudioToMIDI.__new__(EnhancedAudioToMIDI)

    midi_data = _make_midi_with_notes()
    original_starts = [
        n.start for inst in midi_data.instruments for n in inst.notes
    ]

    silent_audio = np.zeros(44100, dtype=np.float32)
    with patch("korshmer.audio_processing.ai_audio_to_midi.librosa") as mock_lr:
        mock_lr.load.return_value = (silent_audio, 44100)
        mock_lr.onset.onset_detect.return_value = np.array([], dtype=np.int64)
        mock_lr.frames_to_time.return_value = np.array([], dtype=np.float64)

        result = transcriber.refine_note_timing("/fake/audio.wav", midi_data)

    result_starts = [n.start for inst in result.instruments for n in inst.notes]
    assert result_starts == original_starts
