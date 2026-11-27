import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

from korshmer.audio_processing.vocal_cleanup import active_level_db

VOCALS = "vocals"
MUSIC = "music"
KINDS = (VOCALS, MUSIC)
ABSENT_UNDER_DB = 30.0
SILENT_DB = -70.0
STEM_SUBTYPE = "FLOAT"
CLIP_QUALITY = "balanced"
NOISE_FRAME = 2048
NOISE_ACTIVE_DB = 30.0
NOISE_SMOOTH_BINS = 33
NOISE_MIN_FLATNESS = 0.9
NOISE_MAX_SPREAD_DB = 5.0
FRAMES_PER_BLOCK = 256
LEAK_LOWEST_HZ = 80.0
LEAK_HIGHEST_HZ = 12000.0
LEAK_BANDS = 12
LEAK_ACTIVE_DB = 40.0
LEAK_FLOOR_DB = 80.0
LEAK_TREND_S = 2.0
LEAK_MIN_FRAMES = 16
LEAK_MIN_MOVE_DB = 1.0
LEAK_MIN_CORRELATION = 0.6
RESIDUAL_UNDER_VOCALS_DB = 20.0
RESIDUAL_PAUSE_DROP_DB = 20.0


class ClipSeparationUnavailable(RuntimeError):
    pass


def vocals_of(mix, progress=None, should_stop=None):
    from korshmer.audio_processing.separation_engine import (
        SeparationOptions,
        vocals_of as split,
    )

    return split(
        mix,
        SeparationOptions(quality=CLIP_QUALITY),
        progress=progress,
        should_stop=should_stop,
    )


def _level_db(part):
    level = active_level_db(part.T)
    return -np.inf if level is None else float(level)


def _mono(part):
    mono = np.asarray(part, dtype=np.float32).mean(axis=0)
    return np.nan_to_num(mono, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _power_blocks(mono):
    frames = sliding_window_view(mono, NOISE_FRAME)[:: NOISE_FRAME // 2]
    window = np.hanning(NOISE_FRAME).astype(np.float32)
    for first in range(0, frames.shape[0], FRAMES_PER_BLOCK):
        block = frames[first : first + FRAMES_PER_BLOCK] * window
        yield first, np.abs(np.fft.rfft(block, axis=1)) ** 2


def _sounds_like_noise(part):
    from scipy.ndimage import median_filter

    mono = _mono(part)
    if mono.shape[0] < NOISE_FRAME:
        return False
    levels = np.concatenate(
        [
            10.0 * np.log10(power.sum(axis=1) + 1e-20)
            for _first, power in _power_blocks(mono)
        ]
    )
    active = levels > levels.max() - NOISE_ACTIVE_DB
    if float(np.std(levels[active])) > NOISE_MAX_SPREAD_DB:
        return False
    spectrum = np.zeros(NOISE_FRAME // 2 + 1)
    for first, power in _power_blocks(mono):
        kept = active[first : first + power.shape[0]]
        spectrum += power[kept].sum(axis=0)
    spectrum = spectrum[1:] / int(active.sum()) + 1e-20
    ratio = spectrum / median_filter(
        spectrum, size=NOISE_SMOOTH_BINS, mode="nearest"
    )
    flatness = np.exp(np.mean(np.log(ratio))) / np.mean(ratio)
    return bool(flatness >= NOISE_MIN_FLATNESS)


def _band_levels_db(mono, sample_rate):
    bins = np.fft.rfftfreq(NOISE_FRAME, 1.0 / sample_rate)
    edges = np.geomspace(LEAK_LOWEST_HZ, LEAK_HIGHEST_HZ, LEAK_BANDS + 1)
    band = np.digitize(bins, edges) - 1
    members = (band[:, np.newaxis] == np.arange(LEAK_BANDS)).astype(np.float32)
    bands = np.concatenate(
        [power @ members for _first, power in _power_blocks(mono)]
    )
    levels = 10.0 * np.log10(bands + 1e-20)
    return np.maximum(levels, levels.max() - LEAK_FLOOR_DB)


def _follows_vocals(music, vocals, sample_rate):
    from scipy.ndimage import uniform_filter1d

    mono = _mono(vocals)
    if mono.shape[0] < NOISE_FRAME:
        return False
    sung = _band_levels_db(mono, sample_rate)
    left = _band_levels_db(_mono(music), sample_rate)
    loudness = 10.0 * np.log10(np.sum(10.0 ** (sung / 10.0), axis=1))
    active = loudness > loudness.max() - LEAK_ACTIVE_DB
    if int(active.sum()) < LEAK_MIN_FRAMES:
        return False
    trend = max(1, int(LEAK_TREND_S * sample_rate / (NOISE_FRAME // 2)))
    changes = []
    for levels in (sung, left):
        moves = levels - uniform_filter1d(levels, trend, axis=0, mode="nearest")
        moves = moves[active]
        changes.append(moves - moves.mean(axis=0))
    together = np.sum(changes[0] * changes[1], axis=0)
    spread = np.sqrt(
        np.sum(changes[0] ** 2, axis=0) * np.sum(changes[1] ** 2, axis=0)
    )
    moving = np.std(changes[0], axis=0) >= LEAK_MIN_MOVE_DB
    heard = moving & (spread > 0)
    if not heard.any():
        return False
    correlation = np.median(together[heard] / spread[heard])
    return bool(correlation >= LEAK_MIN_CORRELATION)


def _frame_loudness_db(part, sample_rate):
    bands = _band_levels_db(_mono(part), sample_rate)
    return 10.0 * np.log10(np.sum(10.0 ** (bands / 10.0), axis=1))


def _only_while_singing(music, vocals, sample_rate):
    if _mono(vocals).shape[0] < NOISE_FRAME:
        return False
    sung = _frame_loudness_db(vocals, sample_rate)
    left = _frame_loudness_db(music, sample_rate)
    singing = sung > sung.max() - LEAK_ACTIVE_DB
    pauses = ~singing
    if int(pauses.sum()) < LEAK_MIN_FRAMES or int(singing.sum()) < 1:
        return False
    under_singing = 10.0 * np.log10(np.mean(10.0 ** (left[singing] / 10.0)))
    in_pauses = 10.0 * np.log10(np.mean(10.0 ** (left[pauses] / 10.0)))
    return bool(in_pauses < under_singing - RESIDUAL_PAUSE_DROP_DB)


def _residual_of_vocals(music, vocals, levels, sample_rate):
    return levels[MUSIC] < levels[VOCALS] - RESIDUAL_UNDER_VOCALS_DB and (
        _only_while_singing(music, vocals, sample_rate)
    )


def _fitted(vocals, frames):
    vocals = np.asarray(vocals, dtype=np.float32)[:, :frames]
    if vocals.shape[1] < frames:
        vocals = np.pad(vocals, ((0, 0), (0, frames - vocals.shape[1])))
    return vocals


def separate_clip_file(path, folder=None, progress=None, should_stop=None):
    from korshmer.audio_processing.clip_polish import fixed_clip_path
    from korshmer.audio_processing.separation_engine import (
        SAMPLE_RATE,
        read_stereo,
    )

    mix = read_stereo(path).astype(np.float32)
    vocals = vocals_of(mix, progress=progress, should_stop=should_stop)
    if vocals is None:
        raise ClipSeparationUnavailable(
            "No vocal separation model could be loaded, so the clip was "
            "not split into vocals and music"
        )
    vocals = _fitted(vocals, mix.shape[1])
    music = mix - vocals
    levels = {VOCALS: _level_db(vocals), MUSIC: _level_db(music)}
    loudest = max(levels.values())
    if not np.isfinite(loudest) or loudest < SILENT_DB:
        return []
    parts = []
    for kind, data in ((VOCALS, vocals), (MUSIC, music)):
        if levels[kind] < loudest - ABSENT_UNDER_DB:
            continue
        if kind == MUSIC and (
            _sounds_like_noise(data)
            or _follows_vocals(data, vocals, SAMPLE_RATE)
            or _residual_of_vocals(data, vocals, levels, SAMPLE_RATE)
        ):
            continue
        target = fixed_clip_path(path, kind, folder)
        sf.write(target, data.T, SAMPLE_RATE, subtype=STEM_SUBTYPE)
        parts.append((kind, target))
    return parts
