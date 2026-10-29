import builtins
import sys

import numpy as np
import pytest

from korshmer.exceptions import AudioToMidiError


class Dummy:
    pass


def test_missing_basic_pitch_raises(monkeypatch):

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("basic_pitch"):
            raise ImportError("no module named basic_pitch")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI

    with pytest.raises(AudioToMidiError):
        EnhancedAudioToMIDI()


def test_constructor_succeeds_with_fake_predict(monkeypatch):

    class FakeInference:
        @staticmethod
        def predict(path, **kwargs):
            return None, None, []

    monkeypatch.setitem(sys.modules, "basic_pitch", Dummy())
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", FakeInference)

    from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI

    ens = EnhancedAudioToMIDI()
    assert hasattr(ens, "basic_pitch_predict")


def _fake_basic_pitch(monkeypatch):
    import pretty_midi

    class FakeInference:
        @staticmethod
        def predict(path, **kwargs):
            midi = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
            )
            midi.instruments.append(inst)
            return (
                np.zeros((10, 88)),
                midi,
                [{"start_time_s": 0.0, "end_time_s": 0.5, "pitch_midi": 60}],
            )

    monkeypatch.setitem(sys.modules, "basic_pitch", Dummy())
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", FakeInference)


def test_transcribe_with_confidence_reports_rising_progress(monkeypatch):
    _fake_basic_pitch(monkeypatch)
    from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI

    calls = []
    EnhancedAudioToMIDI().transcribe_with_confidence(
        "x.wav", progress=lambda f, m: calls.append(f)
    )
    assert len(calls) >= 3
    assert calls == sorted(calls)
    assert all(0.0 <= c <= 1.0 for c in calls)


def test_transcribe_ensemble_passes_progress_through(monkeypatch):
    _fake_basic_pitch(monkeypatch)
    from korshmer.audio_processing.ai_audio_to_midi import (
        EnsembleTranscription,
    )

    calls = []
    EnsembleTranscription().transcribe_ensemble(
        "x.wav", progress=lambda f, m: calls.append(f)
    )
    assert len(calls) >= 3


def test_transcribe_without_progress_and_failing_callback(monkeypatch):
    _fake_basic_pitch(monkeypatch)
    from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI

    def boom(f, m):
        raise RuntimeError("ui gone")

    t = EnhancedAudioToMIDI()
    assert t.transcribe_with_confidence("x.wav")[0] is not None
    assert t.transcribe_with_confidence("x.wav", progress=boom)[0] is not None


def test_polyphonic_separation_reports_progress(monkeypatch):
    _fake_basic_pitch(monkeypatch)
    from korshmer.audio_processing import ai_audio_to_midi

    monkeypatch.setattr(
        ai_audio_to_midi.librosa,
        "load",
        lambda p, sr=None: (np.zeros(10), 22050),
    )
    calls = []
    ai_audio_to_midi.EnhancedAudioToMIDI().polyphonic_separation(
        "x.wav", progress=lambda f, m: calls.append(f)
    )
    assert len(calls) >= 3 and calls == sorted(calls)
