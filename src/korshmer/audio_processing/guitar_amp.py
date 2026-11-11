
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

OVERSAMPLE = 4


@dataclass(frozen=True)
class AmpSettings:
    drive: float
    bias: float = 0.2
    mid_boost_db: float = 6.0
    cab_low_hz: float = 90.0
    cab_high_hz: float = 4800.0
    double_ms: float = 14.0
    mix: float = 1.0


CLEAN = AmpSettings(drive=1.5, mid_boost_db=2.0, cab_high_hz=6500.0, mix=0.6)
CRUNCH = AmpSettings(drive=8.0)
HIGH_GAIN = AmpSettings(drive=25.0, mid_boost_db=4.0, cab_high_hz=4500.0)

                                                                      
AMP_FOR_PROGRAM = {27: CLEAN, 28: CLEAN, 29: CRUNCH, 30: HIGH_GAIN}


def _peaking(sr, freq, gain_db, q=0.8):
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * freq / sr
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])[None, :]


def _rms(x):
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def amp(audio, sample_rate, settings=CRUNCH):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0:
        return np.zeros((0, 2), np.float32)
    dry_rms = _rms(x)
    if dry_rms < 1e-9:
        return np.zeros((len(x), 2), np.float32)
    nyq = sample_rate / 2
    y = sosfilt(butter(2, 100 / nyq, "highpass", output="sos"), x)
    y = sosfilt(_peaking(sample_rate, 800.0, settings.mid_boost_db), y)
    y = y / (np.abs(y).max() + 1e-12)
    up = resample_poly(y, OVERSAMPLE, 1)
    b = settings.bias
    up = np.tanh(settings.drive * up + b) - np.tanh(b)
    y = resample_poly(up, 1, OVERSAMPLE)[: len(x)]
    sos_cab = np.vstack(
        [
            butter(2, settings.cab_low_hz / nyq, "highpass", output="sos"),
            butter(4, settings.cab_high_hz / nyq, "lowpass", output="sos"),
            _peaking(sample_rate, 2500.0, 3.0, q=1.2),
        ]
    )
    y = sosfilt(sos_cab, y)
    y = y * (dry_rms / (_rms(y) + 1e-12))
    y = settings.mix * y + (1 - settings.mix) * x
    delay = int(settings.double_ms / 1000 * sample_rate)
    right = np.concatenate([np.zeros(delay), y[: len(y) - delay]])
                                                                     
                                  
    right = sosfilt(butter(1, 3500 / nyq, "lowpass", output="sos"), right)
    right = right * (_rms(y) / (_rms(right) + 1e-12))
    return np.stack([y, right], axis=1).astype(np.float32)
