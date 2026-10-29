import numpy as np
import soundfile as sf

from korshmer import audio_processors


class _Recorder:
    def __init__(self, calls):
        self.calls = calls

    def process(self, audio, *args):
        self.calls.append(args)
        return audio


def _write_tone(path, sr=44100):
    t = np.arange(sr) / sr
    tone = np.where(
        t > 0.1,
        0.9 * np.sin(2 * np.pi * 440 * t),
        0.01 * np.sin(2 * np.pi * 440 * t),
    )
    sf.write(path, tone, sr)


def test_compressor_processor_converts_seconds_to_milliseconds(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(audio_processors, "_PROC_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio_processors, "Compressor", lambda sr: _Recorder(calls)
    )
    src = str(tmp_path / "in.wav")
    _write_tone(src)
    audio_processors.process_compressor(src)
    assert calls == [(-20, 4.0, 5.0, 100.0)]


def test_limiter_processor_converts_seconds_to_milliseconds(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(audio_processors, "_PROC_DIR", str(tmp_path))
    monkeypatch.setattr(
        audio_processors, "Limiter", lambda sr: _Recorder(calls)
    )
    src = str(tmp_path / "in.wav")
    _write_tone(src)
    audio_processors.process_limiter(src)
    assert calls == [(-1.0, 50.0)]


def test_compressor_processor_keeps_transient_with_default_attack(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audio_processors, "_PROC_DIR", str(tmp_path))
    src = str(tmp_path / "in.wav")
    _write_tone(src)
    out_path = audio_processors.process_compressor(src)
    out, sr = sf.read(out_path)
    onset = np.max(np.abs(out[int(0.1 * sr) : int(0.105 * sr)]))
    assert onset > 0.8
