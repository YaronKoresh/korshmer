from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from korshmer.audio_processing.validation import validate_sample_rate
from korshmer.audio_processing.vocal_cleanup import (
    HOP_LENGTH,
    N_FFT,
    VOCAL_ABOVE_MIDI_DB,
    active_frames,
    active_level_db,
    frame_levels_db,
)

LEVEL_TOLERANCE_DB = 0.5
PEAK_LIMIT_DB = -1.0
PEAK_TOLERANCE_DB = 0.05
MAX_PEAK_TAMING_DB = 6.0
USUAL_PEAK_PERCENTILE = 99.9
PHRASE_SECONDS = 3.0
PHRASE_HOP_SECONDS = 1.0
MIN_PHRASE_ACTIVE_SHARE = 0.25
PHRASE_SPREAD_GOAL_DB = 6.0
PHRASE_SPREAD_TOLERANCE_DB = 0.5
MAX_PHRASE_RIDE_DB = 6.0
OFFSET_GOAL = 2e-3
LIMITER_LOOKAHEAD_S = 0.005
LIMITER_RELEASE_S = 0.08
MAX_PASSES = 4
SILENT_DB = -120.0

LEVEL = "level"
PEAK = "peak"
PHRASES = "phrase spread"
OFFSET = "offset"


@dataclass(frozen=True)
class Reading:
    name: str
    before: float
    goal: float
    after: float
    reached: bool
    was_reached: bool


def level_goal_db():
    from korshmer.audio_processors import RENDER_TARGET_RMS_DB

    return float(RENDER_TARGET_RMS_DB) + float(VOCAL_ABOVE_MIDI_DB)


def _decibels(value):
    return 20.0 * np.log10(max(float(value), 10.0 ** (SILENT_DB / 20.0)))


def _mono(audio):
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def _phrase_levels(mono, sample_rate):
    levels = frame_levels_db(mono)
    active = active_frames(levels)
    size = max(1, int(PHRASE_SECONDS * sample_rate / HOP_LENGTH))
    hop = max(1, int(PHRASE_HOP_SECONDS * sample_rate / HOP_LENGTH))
    power = np.where(active, 10.0 ** (levels / 10.0), 0.0)
    centres, found = [], []
    for first in range(0, max(1, levels.shape[0] - size + 1), hop):
        part = active[first : first + size]
        if part.mean() < MIN_PHRASE_ACTIVE_SHARE:
            continue
        mean_power = power[first : first + size].sum() / part.sum()
        centres.append((first + size / 2.0) * HOP_LENGTH)
        found.append(10.0 * np.log10(mean_power))
    return np.array(centres), np.array(found)


def _phrase_spread(levels):
    if levels.shape[0] < 2:
        return 0.0
    return float(np.percentile(levels, 90) - np.percentile(levels, 10))


def measure(data, sample_rate):
    audio = np.asarray(data, dtype=np.float32)
    mono = _mono(audio)
    level = active_level_db(audio)
    _centres, phrases = (
        _phrase_levels(mono, sample_rate)
        if mono.shape[0] >= N_FFT
        else (np.array([]), np.array([]))
    )
    return {
        LEVEL: SILENT_DB if level is None else level,
        PEAK: _decibels(np.max(np.abs(audio))) if audio.size else SILENT_DB,
        PHRASES: _phrase_spread(phrases),
        OFFSET: float(np.max(np.abs(np.mean(audio, axis=0))))
        if audio.size
        else 0.0,
    }


def goals():
    return {
        LEVEL: level_goal_db(),
        PEAK: PEAK_LIMIT_DB,
        PHRASES: PHRASE_SPREAD_GOAL_DB,
        OFFSET: OFFSET_GOAL,
    }


def meets(name, value, goal):
    if name == LEVEL:
        return abs(value - goal) <= LEVEL_TOLERANCE_DB
    if name == PEAK:
        return value <= goal + PEAK_TOLERANCE_DB
    if name == PHRASES:
        return value <= goal + PHRASE_SPREAD_TOLERANCE_DB
    return value <= goal


def _without_offset(audio):
    return audio - np.mean(audio, axis=0, keepdims=True)


def _evened_phrases(audio, sample_rate):
    mono = _mono(audio)
    centres, levels = _phrase_levels(mono, sample_rate)
    spread = _phrase_spread(levels)
    if spread <= PHRASE_SPREAD_GOAL_DB or centres.shape[0] < 2:
        return audio
    keep = PHRASE_SPREAD_GOAL_DB / spread
    ride = (np.median(levels) - levels) * (1.0 - keep)
    ride = np.clip(ride, -MAX_PHRASE_RIDE_DB, MAX_PHRASE_RIDE_DB)
    curve = np.interp(np.arange(mono.shape[0]), centres, ride)
    smooth = max(1, int(PHRASE_HOP_SECONDS * sample_rate))
    curve = uniform_filter1d(curve, size=smooth, mode="nearest")
    gain = (10.0 ** (curve / 20.0)).astype(np.float32)
    return audio * (gain if audio.ndim == 1 else gain[:, np.newaxis])


def _tamed_peaks(audio, sample_rate, ceiling):
    loudest = np.abs(audio) if audio.ndim == 1 else np.abs(audio).max(axis=1)
    if float(loudest.max()) <= ceiling:
        return audio
    wanted = np.minimum(1.0, ceiling / np.maximum(loudest, 1e-9))
    ahead = max(1, int(LIMITER_LOOKAHEAD_S * sample_rate))
    held = minimum_filter1d(wanted, size=2 * ahead + 1, mode="nearest")
    release = max(1, int(LIMITER_RELEASE_S * sample_rate))
    slow = -maximum_filter1d(-held, size=release, mode="nearest", origin=0)
    gain = uniform_filter1d(np.minimum(held, slow), size=ahead, mode="nearest")
    gain = np.minimum(gain, held).astype(np.float32)
    return audio * (gain if audio.ndim == 1 else gain[:, np.newaxis])


def _at_level(audio, sample_rate, goal):
    ceiling = 10.0 ** (PEAK_LIMIT_DB / 20.0)
    for _ in range(MAX_PASSES):
        level = active_level_db(audio)
        if level is None:
            return audio
        lift = goal - level
        if abs(lift) <= LEVEL_TOLERANCE_DB / 5.0:
            break
        usual_peak = np.percentile(np.abs(audio), USUAL_PEAK_PERCENTILE)
        over = _decibels(usual_peak) + lift - PEAK_LIMIT_DB
        lift -= max(0.0, over - MAX_PEAK_TAMING_DB)
        audio = audio * np.float32(10.0 ** (lift / 20.0))
        audio = _tamed_peaks(audio, sample_rate, ceiling)
        if over > MAX_PEAK_TAMING_DB:
            break
    return audio


def improve(data, sample_rate):
    sample_rate = validate_sample_rate(sample_rate)
    audio = np.nan_to_num(
        np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    wanted = goals()
    before = measure(audio, sample_rate)
    if before[LEVEL] <= SILENT_DB:
        return audio, []
    met = {name: meets(name, before[name], wanted[name]) for name in wanted}
    if all(met.values()):
        return audio, _readings(before, wanted, before, met)
    if not met[OFFSET]:
        audio = _without_offset(audio)
    for _ in range(MAX_PASSES):
        spread = measure(audio, sample_rate)[PHRASES]
        if meets(PHRASES, spread, wanted[PHRASES]):
            break
        audio = _evened_phrases(audio, sample_rate)
    audio = _at_level(audio, sample_rate, wanted[LEVEL])
    audio = np.asarray(audio, dtype=np.float32)
    return audio, _readings(before, wanted, measure(audio, sample_rate), met)


def _readings(before, wanted, after, met):
    return [
        Reading(
            name,
            float(before[name]),
            float(wanted[name]),
            float(after[name]),
            bool(meets(name, after[name], wanted[name])),
            bool(met[name]),
        )
        for name in wanted
    ]


def unchanged(readings):
    return all(reading.was_reached for reading in readings)


def summary(readings):
    if not readings:
        return "is silent, so nothing was changed"
    if unchanged(readings):
        return "already meets every goal, so nothing was changed"
    parts = []
    for reading in readings:
        if reading.name == OFFSET:
            if not reading.was_reached:
                parts.append("offset removed")
            continue
        if reading.name == PHRASES and reading.was_reached:
            continue
        note = "reached" if reading.reached else "not reached"
        parts.append(
            f"{reading.name} {reading.before:.1f} to {reading.after:.1f} dB "
            f"(goal {reading.goal:.1f}, {note})"
        )
    return ", ".join(parts)
