import importlib.util
import sys
import types

import pytest

from korshmer import lazy_imports


def _fake_basic_pitch(monkeypatch, tmp_path, onnx_exists=True):
    calls = []
    onnx = tmp_path / "nmp.onnx"
    if onnx_exists:
        onnx.write_bytes(b"model")

    def predict(audio_path, model_or_model_path="nmp.tflite", **kwargs):
        calls.append((audio_path, model_or_model_path, kwargs))
        return None, None, []

    package = types.ModuleType("basic_pitch")
    package.FilenameSuffix = types.SimpleNamespace(onnx="onnx")
    package.build_icassp_2022_model_path = lambda suffix: onnx
    inference = types.ModuleType("basic_pitch.inference")
    inference.predict = predict
    monkeypatch.setitem(sys.modules, "basic_pitch", package)
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", inference)
    return calls, onnx


def _onnxruntime(monkeypatch, installed):
    found = importlib.util.find_spec

    def find_spec(name, *args):
        if name == "onnxruntime":
            return object() if installed else None
        return found(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


@pytest.fixture
def onnxruntime_installed(monkeypatch):
    _onnxruntime(monkeypatch, True)


def test_the_onnx_model_is_used_when_onnxruntime_is_installed(
    monkeypatch, tmp_path, onnxruntime_installed
):
    calls, onnx = _fake_basic_pitch(monkeypatch, tmp_path)
    predict = lazy_imports.load_basic_pitch_predict()
    predict("song.wav", midi_tempo=100)
    assert calls == [("song.wav", onnx, {"midi_tempo": 100})]


def test_a_model_chosen_by_the_caller_is_kept(
    monkeypatch, tmp_path, onnxruntime_installed
):
    calls, _ = _fake_basic_pitch(monkeypatch, tmp_path)
    lazy_imports.load_basic_pitch_predict()(
        "song.wav", model_or_model_path="mine.tflite"
    )
    assert calls[0][1] == "mine.tflite"


def test_the_default_model_is_used_without_an_onnx_file(
    monkeypatch, tmp_path, onnxruntime_installed
):
    calls, _ = _fake_basic_pitch(monkeypatch, tmp_path, onnx_exists=False)
    lazy_imports.load_basic_pitch_predict()("song.wav")
    assert calls[0][1] == "nmp.tflite"


def test_the_default_model_is_used_without_onnxruntime(monkeypatch, tmp_path):
    calls, _ = _fake_basic_pitch(monkeypatch, tmp_path)
    _onnxruntime(monkeypatch, False)
    lazy_imports.load_basic_pitch_predict()("song.wav")
    assert calls[0][1] == "nmp.tflite"


def test_a_basic_pitch_without_model_paths_still_loads(monkeypatch):
    inference = types.ModuleType("basic_pitch.inference")
    inference.predict = lambda *a, **k: "called"
    monkeypatch.setitem(sys.modules, "basic_pitch", types.ModuleType("x"))
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", inference)
    assert lazy_imports.load_basic_pitch_predict()("song.wav") == "called"
