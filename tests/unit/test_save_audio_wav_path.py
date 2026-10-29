from __future__ import annotations

import os
import tempfile

import numpy as np

from korshmer.audio_processing.save_audio import save_audio


def test_wav_save_uses_wav_extension():
    y = np.zeros(1000, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "output.mid")
        save_audio(path, y, sr=44100, file_format="WAV", bits=32)
        expected = os.path.join(td, "output.wav")
        assert os.path.isfile(expected), f"expected {expected} to exist"
        assert not os.path.isfile(path), f"{path} should not have been created"


def test_wav_save_replaces_existing_extension():
    y = np.zeros(1000, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "song.mp3")
        save_audio(path, y, sr=44100, file_format="WAV", bits=16)
        expected = os.path.join(td, "song.wav")
        assert os.path.isfile(expected)


def test_wav_save_already_wav_extension():
    y = np.zeros(1000, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "track.wav")
        save_audio(path, y, sr=44100, file_format="WAV", bits=32)
        assert os.path.isfile(path)
