from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOMENTARY_S = 0.4
SHORT_TERM_S = 3.0
SILENCE_DB = -70.0
FLOOR = 1e-10


@dataclass(frozen=True)
class Levels:
    peak_db: float = SILENCE_DB
    rms_db: float = SILENCE_DB
    momentary_lufs: float = SILENCE_DB
    short_term_lufs: float = SILENCE_DB
    crest_db: float = 0.0
    correlation: float = 1.0

    @property
    def clipping(self) -> bool:
        return self.peak_db >= -0.1


def _db(value):
    return max(SILENCE_DB, float(20 * np.log10(value + FLOOR)))


def _k_weighted(audio, rate):
    from scipy.signal import lfilter

    gain = 10 ** (3.999843853973347 / 20)
    k = np.tan(np.pi * 1681.974450955533 / rate)
    q = 0.7071752369554196
    vb = gain**0.4996667741545416
    norm = 1 + k / q + k * k
    shelf_b = np.array(
        [
            (gain + vb * k / q + k * k) / norm,
            2 * (k * k - gain) / norm,
            (gain - vb * k / q + k * k) / norm,
        ]
    )
    shelf_a = np.array(
        [1.0, 2 * (k * k - 1) / norm, (1 - k / q + k * k) / norm]
    )
    k = np.tan(np.pi * 38.13547087602444 / rate)
    q = 0.5003270373238773
    norm = 1 + k / q + k * k
    pass_b = np.array([1.0, -2.0, 1.0])
    pass_a = np.array([1.0, 2 * (k * k - 1) / norm, (1 - k / q + k * k) / norm])
    return lfilter(
        pass_b, pass_a, lfilter(shelf_b, shelf_a, audio, axis=0), axis=0
    )


def loudness(audio, rate) -> float:
    block = np.asarray(audio, dtype=np.float64)
    if block.ndim == 1:
        block = block[:, None]
    if len(block) == 0:
        return SILENCE_DB
    power = float(np.mean(_k_weighted(block, int(rate)) ** 2, axis=0).sum())
    if power <= FLOOR:
        return SILENCE_DB
    return max(SILENCE_DB, float(-0.691 + 10 * np.log10(power)))


def measure_levels(audio, rate, position) -> Levels:
    if audio is None or rate <= 0:
        return Levels()
    position = int(max(0, min(len(audio), position)))
    short = np.asarray(
        audio[max(0, position - int(SHORT_TERM_S * rate)) : position],
        dtype=np.float64,
    )
    if short.ndim == 1:
        short = short[:, None]
    recent = short[-int(MOMENTARY_S * rate) :]
    if len(recent) == 0:
        return Levels()
    peak = float(np.abs(recent).max())
    rms = float(np.sqrt(np.mean(recent**2)))
    correlation = 1.0
    if recent.shape[1] >= 2:
        left, right = recent[:, 0], recent[:, 1]
        if np.ptp(left) > 0 and np.ptp(right) > 0:
            correlation = float(np.corrcoef(left, right)[0, 1])
    return Levels(
        _db(peak),
        _db(rms),
        loudness(recent, rate),
        loudness(short, rate),
        _db(peak) - _db(rms) if rms > FLOOR else 0.0,
        correlation,
    )
