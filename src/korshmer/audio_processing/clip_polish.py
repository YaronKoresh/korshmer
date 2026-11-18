import os

import librosa
import numpy as np
import soundfile as sf

from korshmer.constants import DATA_DIR

POLISHED_FOLDER = "clip_edits"
PULSE_HOP = 256
MIN_PULSE_CLARITY = 0.1
MIN_PULSE_SECONDS = 1.0
NOISE_SUFFIX = "clean"
QUALITY_SUFFIX = "improved"


def _polished_path(path, suffix, folder=None):
    folder = folder or os.path.join(str(DATA_DIR), POLISHED_FOLDER)
    os.makedirs(folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(folder, f"{base}_{suffix}.wav")


def _read(path):
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return data, int(sample_rate)


def _write(path, data, sample_rate):
    partial = f"{path}.part"
    sf.write(partial, data, sample_rate, subtype="FLOAT", format="WAV")
    os.replace(partial, path)
    return path


def pulse_shift_beats(
    data, sample_rate, beats_at, start=0.0, offset=0.0, duration=None
):
    audio = np.asarray(data, dtype=np.float32)
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    first = max(0, int(round(float(offset) * sample_rate)))
    last = mono.shape[0]
    if duration is not None:
        last = min(last, first + int(round(float(duration) * sample_rate)))
    mono = mono[first:last]
    if mono.shape[0] < MIN_PULSE_SECONDS * sample_rate:
        return None
    strength = librosa.onset.onset_strength(
        y=mono, sr=sample_rate, hop_length=PULSE_HOP
    )
    strength = np.maximum(strength - np.median(strength), 0.0)
    total = float(strength.sum())
    if total <= 0.0:
        return None
    seconds = (
        float(start) + np.arange(strength.shape[0]) * PULSE_HOP / sample_rate
    )
    beats = np.array([beats_at(second) for second in seconds], dtype=np.float64)
    pulse = np.sum(strength * np.exp(2j * np.pi * beats))
    if abs(pulse) / total < MIN_PULSE_CLARITY:
        return None
    return float(-np.angle(pulse) / (2.0 * np.pi))


def file_pulse_shift_beats(
    path, beats_at, start=0.0, offset=0.0, duration=None
):
    data, sample_rate = _read(path)
    return pulse_shift_beats(
        data, sample_rate, beats_at, start, offset, duration
    )


def remove_noise_file(path, folder=None):
    from korshmer.audio_processing.vocal_cleanup import clean_vocal

    data, sample_rate = _read(path)
    return _write(
        _polished_path(path, NOISE_SUFFIX, folder),
        clean_vocal(data, sample_rate),
        sample_rate,
    )


def improve_quality_file(path, folder=None):
    from korshmer.audio_processing.smart_mastering import (
        SmartMastering,
        SmartMasteringConfig,
    )

    data, sample_rate = _read(path)
    improved, sample_rate = SmartMastering(SmartMasteringConfig()).process(
        data.T, sample_rate
    )
    improved = np.asarray(improved, dtype=np.float32)
    if improved.ndim == 2:
        improved = improved.T
    return _write(
        _polished_path(path, QUALITY_SUFFIX, folder), improved, int(sample_rate)
    )
