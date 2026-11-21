import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest

from korshmer.audio_processing.ai_audio_to_midi import EnhancedAudioToMIDI
from korshmer.core import audio_setup
from korshmer.exceptions import AudioToMidiError

LOAD_FAILURES = [
    TypeError("Unable to convert function return value to a Python type"),
    AttributeError("_ARRAY_API not found"),
    OSError("cannot load shared library"),
    ImportError("no module named tensorflow"),
]


@pytest.fixture
def library_imports(monkeypatch):
    seen = []
    state = {"error": None}
    real_import = builtins.__import__

    def watched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("basic_pitch"):
            seen.append(name)
            if state["error"] is not None:
                raise state["error"]
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", watched_import)
    return SimpleNamespace(seen=seen, state=state)


def test_creating_the_transcriber_does_not_load_the_transcription_library(
    library_imports,
):
    EnhancedAudioToMIDI()

    assert library_imports.seen == []


@pytest.mark.parametrize("error", LOAD_FAILURES, ids=lambda e: type(e).__name__)
def test_a_library_that_breaks_while_loading_is_reported_when_transcribing(
    library_imports, tmp_path, error
):
    library_imports.state["error"] = error
    transcriber = EnhancedAudioToMIDI()

    with pytest.raises(AudioToMidiError) as raised:
        transcriber.transcribe_with_confidence(str(tmp_path / "song.wav"))

    assert str(error) in str(raised.value)


def test_a_missing_transcription_library_is_reported_when_creating_the_transcriber(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "basic_pitch", None)

    with pytest.raises(AudioToMidiError) as raised:
        EnhancedAudioToMIDI()

    assert "pip install basic-pitch" in str(raised.value)


def test_all_processors_are_created_while_the_transcription_library_is_broken(
    library_imports,
):
    library_imports.state["error"] = TypeError("broken numeric library")
    window = SimpleNamespace()

    audio_setup.init_audio_and_midi_processors(window)

    assert window.enhanced_audio_to_midi is not None
    assert window.beat_detector is not None
    assert window.chord_detector is not None
    assert library_imports.seen == []


def test_the_model_download_script_loads_while_the_transcription_library_is_broken(
    library_imports, monkeypatch, tmp_path
):
    library_imports.state["error"] = TypeError("broken numeric library")
    monkeypatch.delitem(sys.modules, "korshmer.download_models", raising=False)

    download_models = importlib.import_module("korshmer.download_models")

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    copied = download_models.copy_transcription_model(model, tmp_path / "out")
    assert copied.read_bytes() == b"model"
