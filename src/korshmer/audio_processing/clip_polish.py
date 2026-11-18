import os

import librosa
import numpy as np
import soundfile as sf

from korshmer.constants import DATA_DIR

POLISHED_FOLDER = "clip_edits"
PULSE_HOP = 256
PULSE_STEPS_PER_BEAT = (1, 2, 4)
PULSE_CERTAINTY = 2.6
COARSER_PULSE_SHARE = 0.7
CUT_EDGE_SECONDS = 0.05
NOISE_SPREADS = 2.0
FIRST_SOUND_SHARE = 0.1
SILENCE_LEVEL = 1e-4
ON_PULSE = "pulse"
ON_FIRST_SOUND = "first sound"
SILENT = "silent"
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


def _kept_part(data, sample_rate, offset, duration):
    audio = np.asarray(data, dtype=np.float32)
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    first = max(0, int(round(float(offset) * sample_rate)))
    last = mono.shape[0]
    if duration is not None:
        last = min(last, first + int(round(float(duration) * sample_rate)))
    return np.nan_to_num(mono[first:last], nan=0.0, posinf=0.0, neginf=0.0)


def _beats_of(frames, sample_rate, beats_at, start):
    seconds = float(start) + frames * PULSE_HOP / sample_rate
    return np.array([beats_at(second) for second in seconds], dtype=np.float64)


def _pulse_shift(mono, sample_rate, beats_at, start):
    if mono.shape[0] < MIN_PULSE_SECONDS * sample_rate:
        return None
    strength = librosa.onset.onset_strength(
        y=mono, sr=sample_rate, hop_length=PULSE_HOP
    )
    hits = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sample_rate, hop_length=PULSE_HOP
    )
    hits = hits[hits >= CUT_EDGE_SECONDS * sample_rate / PULSE_HOP]
    middle = float(np.median(strength))
    floor = middle + NOISE_SPREADS * float(np.median(np.abs(strength - middle)))
    weights = np.maximum(strength[hits] - floor, 0.0).astype(np.float64)
    spread = float(np.sqrt(np.sum(weights**2)))
    if hits.size == 0 or spread <= 0.0:
        return None
    beats = _beats_of(hits, sample_rate, beats_at, start)
    pulses = [
        (np.sum(weights * np.exp(2j * np.pi * steps * beats)), steps)
        for steps in PULSE_STEPS_PER_BEAT
    ]
    strongest = max(abs(pulse) for pulse, _steps in pulses)
    if strongest / spread < PULSE_CERTAINTY:
        return None
    pulse, steps = next(
        found
        for found in pulses
        if abs(found[0]) >= COARSER_PULSE_SHARE * strongest
    )
    return float(-np.angle(pulse) / (2.0 * np.pi * steps))


def _first_sound_shift(mono, sample_rate, beats_at, start):
    if mono.shape[0] < PULSE_HOP:
        return None
    levels = librosa.feature.rms(
        y=mono, frame_length=2 * PULSE_HOP, hop_length=PULSE_HOP, center=False
    )[0]
    peak = float(levels.max()) if levels.size else 0.0
    if peak <= SILENCE_LEVEL:
        return None
    heard = np.flatnonzero(levels >= peak * FIRST_SOUND_SHARE)
    beat = _beats_of(heard[:1], sample_rate, beats_at, start)[0]
    return float(np.round(beat) - beat)


def clip_beat_shift(
    data, sample_rate, beats_at, start=0.0, offset=0.0, duration=None
):
    mono = _kept_part(data, sample_rate, offset, duration)
    shift = _pulse_shift(mono, sample_rate, beats_at, start)
    if shift is not None:
        return shift, ON_PULSE
    shift = _first_sound_shift(mono, sample_rate, beats_at, start)
    if shift is not None:
        return shift, ON_FIRST_SOUND
    return None, SILENT


def file_beat_shift(path, beats_at, start=0.0, offset=0.0, duration=None):
    data, sample_rate = _read(path)
    return clip_beat_shift(data, sample_rate, beats_at, start, offset, duration)


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
