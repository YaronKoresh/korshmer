import librosa
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import (
    label,
    maximum_filter1d,
    median_filter,
    percentile_filter,
    uniform_filter,
    uniform_filter1d,
)

from korshmer.audio_processing.validation import validate_sample_rate
from korshmer.ui.viewmodels.editor_vm import MAX_CLIP_GAIN

HIGHPASS_HZ = 70.0
N_FFT = 2048
HOP_LENGTH = 512
ACTIVE_RANGE_DB = 30.0
PEAK_CEILING = 0.99
EPSILON = 1e-10
NOISE_PERCENTILE = 10.0
NOISE_SPREAD_HZ = 600.0
NOISE_SPREAD_PERCENTILE = 20.0
NOISE_OVER_SUBTRACTION = 3.0
NOISE_MAX_CUT_DB = 12.0
NOISE_GAIN_SMOOTHING = (3, 5)
PAUSE_SNR_DB = 8.0
PAUSE_RATIO = 4.0
PAUSE_MAX_CUT_DB = 12.0
PAUSE_HOLD_S = 0.15
PAUSE_LOOKAHEAD_S = 0.05
PAUSE_SMOOTHING_S = 0.08
BALANCE_CENTRES_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
BALANCE_TARGET_DB = (-6.0, 0.0, 1.0, 0.0, -5.0, -11.0, -17.0)
BALANCE_CUT_ONLY = (True, True, True, False, False, False, False)
BALANCE_TOLERANCE_DB = (6.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0)
BALANCE_BODY = 3
BALANCE_MAX_DB = 9.0
BALANCE_MIN_SNR_DB = 10.0
BALANCE_BODY_RANGE_DB = 30.0
BALANCE_PASSES = 3
BALANCE_BLOCK_S = 2.0
BALANCE_BLOCK_STEPS = 4
BALANCE_BLOCK_TOLERANCE_DB = 1.0
BREATH_FLAT_BAND_HZ = (500.0, 3000.0)
BREATH_CENTROID_BAND_HZ = (300.0, 10000.0)
BREATH_FLATNESS = 0.3
BREATH_CENTROID_HZ = 4000.0
BREATH_UNDER_SINGING_DB = 12.0
BREATH_NEARBY_S = 1.0
BREATH_MIN_S = 0.15
BREATH_CUT_DB = 12.0
POP_BAND_HZ = (20.0, 250.0)
POP_FADE_TO_HZ = 350.0
POP_REFERENCE_HZ = (300.0, 3000.0)
POP_TYPICAL_S = 1.0
POP_JUMP_DB = 12.0
POP_LOW_OVER_MIDDLE_DB = 0.0
POP_MAX_CUT_DB = 30.0
LEVEL_SHARE = 0.8
LEVEL_DEAD_BAND_DB = 1.5
LEVEL_SUNG_NEARBY_S = 0.1
LEVEL_MAX_DB = 9.0
LEVEL_SMOOTHING_S = 1.0
LEVEL_SUNG_RANGE_DB = 40.0
LEVEL_MIN_SUNG_SHARE = 0.05
BALANCE_FADE_FROM_HZ = 11000.0
BALANCE_FADE_TO_HZ = 16000.0
SIBILANCE_HZ = (4500.0, 10000.0)
SIBILANCE_REFERENCE_HZ = (1000.0, 4000.0)
SIBILANCE_SPAN_DB = 10.0
SIBILANCE_NATURAL_RISE_DB = -12.0
SIBILANCE_SPREAD_FRAMES = 5
LOUDNESS_MAX_DB = 6.0
VOCAL_ABOVE_MIDI_DB = -1.5
MIN_VOCAL_GAIN = 0.1
MAX_VOCAL_GAIN = MAX_CLIP_GAIN
NOISE_POWER_PER_FLOOR = 2.0 / (-2.0 * np.log(1.0 - NOISE_PERCENTILE / 100.0))


def frame_levels_db(channel, frame_length=N_FFT, hop_length=HOP_LENGTH):
    frames = librosa.feature.rms(
        y=channel, frame_length=frame_length, hop_length=hop_length
    )[0]
    return 20.0 * np.log10(np.maximum(frames, EPSILON))


def active_frames(levels_db, range_db=ACTIVE_RANGE_DB):
    if levels_db.size == 0:
        return np.zeros(0, dtype=bool)
    top = float(np.percentile(levels_db, 95))
    return levels_db > top - range_db


def highpass(channel, sample_rate, cutoff_hz=HIGHPASS_HZ):
    if sample_rate <= 2 * cutoff_hz * 2:
        return channel
    sos = signal.butter(
        4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos"
    )
    if channel.shape[0] <= 3 * (2 * sos.shape[0] + 1):
        return signal.sosfilt(sos, channel).astype(np.float32)
    return signal.sosfiltfilt(sos, channel).astype(np.float32)


def _power_db(power):
    return 10.0 * np.log10(np.maximum(power, EPSILON**2))


def noise_floor_power(magnitude, sample_rate):
    steady = np.percentile(magnitude, NOISE_PERCENTILE, axis=1)
    width = max(3, int(round(NOISE_SPREAD_HZ * N_FFT / sample_rate)) | 1)
    floor = percentile_filter(
        steady, NOISE_SPREAD_PERCENTILE, size=width, mode="nearest"
    )
    return (floor**2 * NOISE_POWER_PER_FLOOR).astype(np.float32)


def noise_gains(
    magnitude,
    noise_power,
    over_subtraction=NOISE_OVER_SUBTRACTION,
    max_cut_db=NOISE_MAX_CUT_DB,
):
    lowest = 10.0 ** (-max_cut_db / 10.0)
    share = over_subtraction * noise_power[:, np.newaxis]
    kept = 1.0 - share / np.maximum(magnitude**2, EPSILON**2)
    gains = np.sqrt(np.clip(kept, lowest, 1.0))
    return uniform_filter(gains, size=NOISE_GAIN_SMOOTHING, mode="nearest")


def _frames(seconds, sample_rate):
    return max(0, int(round(seconds * sample_rate / HOP_LENGTH)))


def pause_gains(
    magnitude,
    noise_power,
    sample_rate,
    snr_db=PAUSE_SNR_DB,
    ratio=PAUSE_RATIO,
    max_cut_db=PAUSE_MAX_CUT_DB,
):
    heard = _power_db(np.sum(magnitude**2, axis=0))
    snr = heard - _power_db(np.sum(noise_power))
    behind = _frames(PAUSE_HOLD_S, sample_rate)
    ahead = _frames(PAUSE_LOOKAHEAD_S, sample_rate)
    padded = np.pad(snr, (behind, ahead), mode="edge")
    held = sliding_window_view(padded, behind + ahead + 1).max(axis=1)
    cut = np.clip((snr_db - held) * (ratio - 1.0), 0.0, max_cut_db)
    smoothing = max(1, _frames(PAUSE_SMOOTHING_S, sample_rate)) | 1
    cut = uniform_filter1d(cut, size=smoothing, mode="nearest")
    return (10.0 ** (-cut / 20.0)).astype(np.float32)


def _band(freqs, low, high):
    return (freqs >= low) & (freqs < high)


def band_levels(power, noise_power, freqs):
    levels = _power_db(power.sum(axis=0))
    talking = levels > np.percentile(levels, 95) - ACTIVE_RANGE_DB
    voice = power[:, talking].mean(axis=1)
    measured, snr = [], []
    for centre in BALANCE_CENTRES_HZ:
        bins = _band(freqs, centre / np.sqrt(2.0), centre * np.sqrt(2.0))
        if not np.any(bins):
            measured.append(None)
            snr.append(None)
            continue
        level = float(_power_db(voice[bins].mean()))
        measured.append(level)
        snr.append(level - float(_power_db(noise_power[bins].mean())))
    body = measured[BALANCE_BODY]
    present = [level for level in measured if level is not None]
    if (
        body is None
        or snr[BALANCE_BODY] < BALANCE_MIN_SNR_DB
        or body < max(present) - BALANCE_BODY_RANGE_DB
    ):
        return None
    relative = [None if level is None else level - body for level in measured]
    return relative, snr


def _clip_targets(power, noise_power, freqs):
    reading = band_levels(power, noise_power, freqs)
    if reading is None:
        return list(BALANCE_TARGET_DB)
    relative, _snr = reading
    return [
        target
        if level is None
        else float(np.clip(level, target - tolerance, target + tolerance))
        for level, target, tolerance in zip(
            relative, BALANCE_TARGET_DB, BALANCE_TOLERANCE_DB, strict=True
        )
    ]


def _balance_db(power, noise_power, freqs, targets):
    reading = band_levels(power, noise_power, freqs)
    if reading is None:
        return None
    relative, snr = reading
    centres = []
    for index, level in enumerate(relative):
        if level is None:
            continue
        deviation = level - targets[index]
        excess = max(abs(deviation) - BALANCE_BLOCK_TOLERANCE_DB, 0.0)
        change = -np.sign(deviation) * min(excess, BALANCE_MAX_DB)
        if change > 0 and (
            BALANCE_CUT_ONLY[index] or snr[index] < BALANCE_MIN_SNR_DB
        ):
            change = 0.0
        centres.append((np.log2(BALANCE_CENTRES_HZ[index]), change))
    places, changes = zip(*centres, strict=True)
    curve = np.interp(np.log2(np.maximum(freqs, 1.0)), places, changes)
    fade = np.clip(
        (BALANCE_FADE_TO_HZ - freqs)
        / (BALANCE_FADE_TO_HZ - BALANCE_FADE_FROM_HZ),
        0.0,
        1.0,
    )
    return np.where(curve > 0.0, curve * fade, curve)


def _settled_balance_db(power, noise_power, freqs, targets):
    total = np.zeros_like(freqs)
    found = False
    for _ in range(BALANCE_PASSES):
        shaped = power * 10.0 ** (total / 10.0)[:, np.newaxis]
        step = _balance_db(shaped, noise_power, freqs, targets)
        if step is None:
            break
        found = True
        if not np.any(step):
            break
        total = np.clip(total + step, -BALANCE_MAX_DB, BALANCE_MAX_DB)
    return total if found else None


def _section_curves(power, noise_power, freqs, sample_rate):
    targets = _clip_targets(power, noise_power, freqs)
    frames = power.shape[1]
    block = max(1, _frames(BALANCE_BLOCK_S, sample_rate))
    if frames <= block:
        starts = [0]
    else:
        step = max(1, block // BALANCE_BLOCK_STEPS)
        starts = list(range(0, frames - block // 2, step))
    centres, curves = [], []
    for start in starts:
        part = power[:, start : start + block]
        curve = _settled_balance_db(part, noise_power, freqs, targets)
        if curve is not None:
            centres.append(start + part.shape[1] / 2.0)
            curves.append(curve)
    if not curves:
        return None
    table = np.asarray(curves, dtype=np.float32).T
    if table.shape[1] == 1:
        return np.repeat(table, frames, axis=1)
    where = np.arange(frames, dtype=np.float64)
    after = np.clip(np.searchsorted(centres, where), 1, len(centres) - 1)
    left = np.asarray(centres)[after - 1]
    right = np.asarray(centres)[after]
    share = np.clip((where - left) / np.maximum(right - left, 1e-9), 0.0, 1.0)
    share = share.astype(np.float32)
    return table[:, after - 1] * (1.0 - share) + table[:, after] * share


def sibilance(power, freqs):
    hiss = power[_band(freqs, *SIBILANCE_HZ)].mean(axis=0)
    body = power[_band(freqs, *SIBILANCE_REFERENCE_HZ)].mean(axis=0)
    rise = _power_db(hiss) - _power_db(body) - SIBILANCE_NATURAL_RISE_DB
    share = np.clip(rise / SIBILANCE_SPAN_DB, 0.0, 1.0)
    return maximum_filter1d(share, size=SIBILANCE_SPREAD_FRAMES, mode="nearest")


def _tone_gains(power, noise_power, sample_rate):
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    curves = _section_curves(power, noise_power, freqs, sample_rate)
    if curves is None or not np.any(curves):
        return np.ones_like(power)
    boost = np.maximum(curves, np.float32(0.0))
    cut = np.minimum(curves, np.float32(0.0))
    withheld = sibilance(power, freqs).astype(np.float32)[np.newaxis, :]
    return np.float32(10.0) ** ((cut + boost * (1.0 - withheld)) / 20.0)


def _breath_gains(power, sample_rate):
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    flat_band = power[_band(freqs, *BREATH_FLAT_BAND_HZ)] + EPSILON**2
    flatness = np.exp(np.mean(np.log(flat_band), axis=0)) / np.mean(
        flat_band, axis=0
    )
    wide = _band(freqs, *BREATH_CENTROID_BAND_HZ)
    spread = power[wide] + EPSILON**2
    centroid = (freqs[wide, np.newaxis] * spread).sum(axis=0) / spread.sum(
        axis=0
    )
    level = _power_db(power.sum(axis=0))
    nearby = maximum_filter1d(
        level, size=max(1, _frames(BREATH_NEARBY_S, sample_rate)) | 1
    )
    breathy = (
        (flatness >= BREATH_FLATNESS)
        & (centroid < BREATH_CENTROID_HZ)
        & (level <= nearby - BREATH_UNDER_SINGING_DB)
    )
    runs, count = label(breathy)
    shortest = max(1, _frames(BREATH_MIN_S, sample_rate))
    lasting = np.zeros_like(breathy)
    for run in range(1, count + 1):
        members = runs == run
        if members.sum() >= shortest:
            lasting |= members
    cut = uniform_filter1d(
        np.where(lasting, BREATH_CUT_DB, 0.0), size=3, mode="nearest"
    )
    return (10.0 ** (-cut / 20.0)).astype(np.float32)


def _pop_gains(power, sample_rate):
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    low = _power_db(power[_band(freqs, *POP_BAND_HZ)].sum(axis=0))
    middle = _power_db(power[_band(freqs, *POP_REFERENCE_HZ)].sum(axis=0))
    usual = max(1, _frames(POP_TYPICAL_S, sample_rate)) | 1
    low_rise = low - median_filter(low, size=usual, mode="nearest")
    middle_rise = middle - median_filter(middle, size=usual, mode="nearest")
    jump = low_rise - np.maximum(middle_rise, 0.0)
    popped = (jump >= POP_JUMP_DB) & (low >= middle + POP_LOW_OVER_MIDDLE_DB)
    cut = np.where(popped, np.minimum(jump, POP_MAX_CUT_DB), 0.0)
    cut = maximum_filter1d(cut, size=3, mode="nearest")
    reach = np.clip(
        (POP_FADE_TO_HZ - freqs) / (POP_FADE_TO_HZ - POP_BAND_HZ[1]), 0.0, 1.0
    )
    shaped = reach[:, np.newaxis] * cut[np.newaxis, :]
    return (10.0 ** (-shaped / 20.0)).astype(np.float32)


def _level_gains(power, pause, sample_rate):
    frame_power = power.sum(axis=0)
    level = _power_db(frame_power)
    sung = (pause > 0.7) & (level > level.max() - LEVEL_SUNG_RANGE_DB)
    if not np.any(sung):
        return np.ones(level.shape, dtype=np.float32)
    reference = float(_power_db(frame_power[sung].mean()))
    size = max(1, _frames(LEVEL_SMOOTHING_S, sample_rate)) | 1
    weight = uniform_filter1d(sung.astype(np.float64), size, mode="nearest")
    total = uniform_filter1d(
        np.where(sung, frame_power, 0.0), size, mode="nearest"
    )
    known = weight > LEVEL_MIN_SUNG_SHARE
    local = np.interp(
        np.arange(level.size),
        np.flatnonzero(known),
        _power_db(total[known] / weight[known]),
    )
    shortfall = reference - local - LEVEL_DEAD_BAND_DB
    lift = np.clip(shortfall * LEVEL_SHARE, 0.0, LEVEL_MAX_DB)
    near = max(1, _frames(LEVEL_SUNG_NEARBY_S, sample_rate)) | 1
    lift *= uniform_filter1d(sung.astype(np.float64), near, mode="nearest")
    return (10.0 ** (lift / 20.0)).astype(np.float32)


def loudness_gain(cleaned, reference):
    before = frame_levels_db(reference)
    after = frame_levels_db(cleaned)
    count = min(before.size, after.size)
    before, after = before[:count], after[:count]
    sung = active_frames(before)
    if not np.any(sung):
        return 1.0
    wanted = 10.0 * np.log10(np.mean(10.0 ** (before[sung] / 10.0)))
    got = 10.0 * np.log10(np.mean(10.0 ** (after[sung] / 10.0)))
    change = float(np.clip(wanted - got, -LOUDNESS_MAX_DB, LOUDNESS_MAX_DB))
    return 10.0 ** (change / 20.0)


def istft(spectrum, gains, length):
    return librosa.istft(spectrum * gains, hop_length=HOP_LENGTH, length=length)


def _clean_channels(channels, sample_rate):
    filtered = [highpass(channel, sample_rate) for channel in channels]
    length = filtered[0].shape[0]
    if length < N_FFT:
        return filtered
    mono = np.mean(filtered, axis=0, dtype=np.float32)
    magnitude = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH))
    noise_power = noise_floor_power(magnitude, sample_rate)
    raw_power = magnitude**2
    denoise = noise_gains(magnitude, noise_power)
    pause = pause_gains(magnitude, noise_power, sample_rate)
    repaired = (
        denoise
        * _pop_gains(raw_power, sample_rate)
        * _breath_gains(raw_power, sample_rate)[np.newaxis, :]
    )
    tone = _tone_gains(raw_power * repaired**2, noise_power, sample_rate)
    gains = repaired * tone * pause[np.newaxis, :]
    spectra = [
        librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for channel in filtered
    ]
    outputs = [istft(spectrum, gains, length) for spectrum in spectra]
    level = np.float32(loudness_gain(np.mean(outputs, axis=0), mono))
    shaped = raw_power * (gains * level) ** 2
    lift = _level_gains(shaped, pause, sample_rate)[np.newaxis, :]
    if np.all(lift == 1.0):
        return [output * level for output in outputs]
    return [
        istft(spectrum, gains * (level * lift), length) for spectrum in spectra
    ]


def clean_vocal(data, sample_rate):
    sample_rate = validate_sample_rate(sample_rate)
    audio = np.asarray(data, dtype=np.float32)
    if audio.size == 0:
        return audio.copy()
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    wide = audio[:, np.newaxis] if audio.ndim == 1 else audio
    cleaned = np.stack(
        _clean_channels(
            [np.ascontiguousarray(wide[:, i]) for i in range(wide.shape[1])],
            sample_rate,
        ),
        axis=1,
    )
    if audio.ndim == 1:
        cleaned = cleaned[:, 0]
    cleaned = cleaned.astype(np.float32)
    peak = float(np.max(np.abs(cleaned)))
    if peak > PEAK_CEILING:
        cleaned *= np.float32(PEAK_CEILING / peak)
    return cleaned


def active_level_db(data):
    audio = np.asarray(data, dtype=np.float32)
    if audio.size == 0:
        return None
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    if mono.shape[0] < N_FFT:
        return None
    levels = frame_levels_db(mono)
    active = levels[active_frames(levels)]
    if active.size == 0:
        return None
    level = 10.0 * np.log10(np.mean(10.0 ** (active / 10.0)))
    return float(level) if level > 20.0 * np.log10(EPSILON) + 1.0 else None


def vocal_clip_gain(
    data, midi_level_db=None, above_midi_db=VOCAL_ABOVE_MIDI_DB
):
    level = active_level_db(data)
    if level is None:
        return 1.0
    if midi_level_db is None:
        from korshmer.audio_processors import RENDER_TARGET_RMS_DB

        midi_level_db = RENDER_TARGET_RMS_DB
    gain = 10.0 ** (
        (float(midi_level_db) + float(above_midi_db) - level) / 20.0
    )
    return float(min(MAX_VOCAL_GAIN, max(MIN_VOCAL_GAIN, gain)))
