from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import (
    maximum_filter1d,
    minimum_filter1d,
    uniform_filter,
    uniform_filter1d,
)
from scipy.signal import resample_poly

from korshmer.audio_processing.clip_stems import MUSIC, VOCALS
from korshmer.audio_processing.validation import validate_sample_rate
from korshmer.audio_processing.vocal_cleanup import (
    BALANCE_CENTRES_HZ,
    BALANCE_CUT_ONLY,
    BALANCE_FADE_FROM_HZ,
    BALANCE_FADE_TO_HZ,
    BALANCE_MIN_SNR_DB,
    BALANCE_TARGET_DB,
    HOP_LENGTH,
    N_FFT,
    VOCAL_ABOVE_MIDI_DB,
    active_frames,
    active_level_db,
    band_levels,
    frame_levels_db,
    noise_floor_power,
    sibilance,
)

LEVEL_TOLERANCE_DB = 0.5
PEAK_LIMIT_DB = -1.0
PEAK_TOLERANCE_DB = 0.05
MAX_PEAK_TAMING_DB = 6.0
TRUE_PEAK_OVERSAMPLING = 4
USUAL_PEAK_PERCENTILE = 99.9
PHRASE_SECONDS = 3.0
PHRASE_HOP_SECONDS = 1.0
MIN_PHRASE_ACTIVE_SHARE = 0.25
PHRASE_SPREAD_GOAL_DB = 6.0
PHRASE_SPREAD_TOLERANCE_DB = 0.5
MAX_PHRASE_RIDE_DB = 6.0
OFFSET_GOAL = 2e-3
NOISE_GOAL_DB = -40.0
NOISE_TOLERANCE_DB = 0.5
PAUSE_GAP_DB = 15.0
PAUSE_PERCENTILE = 2.0
STEADY_NOISE_RANGE_DB = 4.0
MAX_NOISE_CUT_DB = 18.0
NOISE_MARGIN = 2.0
NOISE_SMOOTHING = (3, 7)
OVER_NOTES_GOAL_DB = 3.0
OVER_NOTES_TOLERANCE_DB = 0.5
MAX_LIFT_OVER_NOTES_DB = 6.0
NOTES_SILENT_DB = -70.0
MAIN_BAND_SHARES = (0.1, 0.9)
LIMITER_LOOKAHEAD_S = 0.005
LIMITER_RELEASE_S = 0.08
TONE_GOAL_DB = 3.0
TONE_TOLERANCE_DB = 0.5
MAX_TONE_CHANGE_DB = 15.0
VOICE_BANDS_HZ = (250.0, 4000.0)
VOICE_BAND_RANGE_DB = 40.0
UPPER_VOICE_BANDS_FROM_HZ = 4000.0
UPPER_VOICE_BAND_RANGE_DB = 60.0
MAX_PASSES = 4
SILENT_DB = -120.0

LEVEL = "level"
PEAK = "peak"
PHRASES = "phrase spread"
OFFSET = "offset"
NOISE = "noise in pauses"
OVER_NOTES = "over the notes"
TONE = "tone"


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


def _true_peak(audio):
    if not audio.size:
        return 0.0
    samples = np.asarray(audio, dtype=np.float64)
    finer = resample_poly(samples, TRUE_PEAK_OVERSAMPLING, 1, axis=0)
    return float(np.abs(finer).max())


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


def _pause_frames(levels):
    return levels <= np.percentile(levels, PAUSE_PERCENTILE)


def _noise_db(mono, level):
    if mono.shape[0] < N_FFT or level <= SILENT_DB:
        return SILENT_DB
    levels = frame_levels_db(mono)
    quiet = levels[_pause_frames(levels)]
    if quiet.max() - quiet.min() > STEADY_NOISE_RANGE_DB:
        return SILENT_DB
    noise = 10.0 * np.log10(np.mean(10.0 ** (quiet / 10.0))) - level
    if noise > -PAUSE_GAP_DB:
        return SILENT_DB
    return max(SILENT_DB, float(noise))


def _spectrum(mono, sample_rate):
    magnitude = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    return magnitude**2, noise_floor_power(magnitude, sample_rate), freqs


def _voice_band_range_db(centre):
    if centre >= UPPER_VOICE_BANDS_FROM_HZ:
        return UPPER_VOICE_BAND_RANGE_DB
    return VOICE_BAND_RANGE_DB


def _fills_voice_bands(relative, snr):
    low, high = VOICE_BANDS_HZ
    return all(
        level is not None
        and level >= -_voice_band_range_db(centre)
        and band_snr >= BALANCE_MIN_SNR_DB
        for centre, level, band_snr in zip(
            BALANCE_CENTRES_HZ, relative, snr, strict=True
        )
        if low <= centre <= high
    )


def _tone_changes(power, noise_power, freqs):
    reading = band_levels(power, noise_power, freqs)
    if reading is None:
        return None
    relative, snr = reading
    if not _fills_voice_bands(relative, snr):
        return None
    changes = {}
    for index, level in enumerate(relative):
        if level is None:
            continue
        change = BALANCE_TARGET_DB[index] - level
        if change > 0.0 and (
            BALANCE_CUT_ONLY[index] or snr[index] < BALANCE_MIN_SNR_DB
        ):
            change = 0.0
        changes[BALANCE_CENTRES_HZ[index]] = change
    return changes or None


def _tone_db(mono, sample_rate):
    if mono.shape[0] < N_FFT:
        return None
    changes = _tone_changes(*_spectrum(mono, sample_rate))
    if changes is None:
        return None
    return float(max(abs(change) for change in changes.values()))


def _fitted(music, length):
    notes = np.zeros(length, dtype=np.float32)
    if music is not None:
        heard = np.asarray(music, dtype=np.float32)[:length]
        notes[: heard.shape[0]] = heard
    return notes


def _spectra(mono, notes, sounding):
    clip = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    song = np.abs(librosa.stft(notes, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    frames = min(clip.shape[1], sounding.shape[0])
    chosen = sounding[:frames]
    return clip[:, :frames][:, chosen].mean(axis=1), song[:, :frames][
        :, chosen
    ].mean(axis=1)


def notes_levels(mono, music):
    if music is None or mono.shape[0] < N_FFT:
        return None
    notes = _fitted(music, mono.shape[0])
    levels = frame_levels_db(mono)
    sounding = active_frames(levels)
    if not np.any(sounding):
        return None
    heard = frame_levels_db(notes)[sounding]
    level = 10.0 * np.log10(np.mean(10.0 ** (heard / 10.0)))
    if level <= NOTES_SILENT_DB:
        return None
    clip, song = _spectra(mono, notes, sounding)
    share = np.cumsum(clip) / max(float(clip.sum()), 1e-30)
    low, high = np.searchsorted(share, MAIN_BAND_SHARES)
    band = slice(int(low), int(max(high, low + 1)) + 1)
    over = 10.0 * np.log10(
        max(float(clip[band].sum()), 1e-30)
        / max(float(song[band].sum()), 1e-30)
    )
    return float(level), float(over)


def measure(data, sample_rate, music=None):
    audio = np.asarray(data, dtype=np.float32)
    mono = _mono(audio)
    level = active_level_db(audio)
    level = SILENT_DB if level is None else level
    _centres, phrases = (
        _phrase_levels(mono, sample_rate)
        if mono.shape[0] >= N_FFT
        else (np.array([]), np.array([]))
    )
    found = {
        LEVEL: level,
        PEAK: _decibels(_true_peak(audio)) if audio.size else SILENT_DB,
        PHRASES: _phrase_spread(phrases),
        OFFSET: float(np.max(np.abs(np.mean(audio, axis=0))))
        if audio.size
        else 0.0,
        NOISE: _noise_db(mono, level),
    }
    tone = _tone_db(mono, sample_rate)
    if tone is not None:
        found[TONE] = tone
    beside_notes = notes_levels(mono, music)
    if beside_notes is not None:
        found[OVER_NOTES] = beside_notes[1]
    return found


def music_level_goal_db():
    from korshmer.audio_processors import RENDER_TARGET_RMS_DB

    return float(RENDER_TARGET_RMS_DB)


def music_goals():
    return {
        LEVEL: music_level_goal_db(),
        PEAK: PEAK_LIMIT_DB,
        OFFSET: OFFSET_GOAL,
        NOISE: NOISE_GOAL_DB,
    }


def goals(before=None, music_level=None):
    level = level_goal_db()
    wanted = {
        LEVEL: level,
        PEAK: PEAK_LIMIT_DB,
        PHRASES: PHRASE_SPREAD_GOAL_DB,
        OFFSET: OFFSET_GOAL,
        NOISE: NOISE_GOAL_DB,
    }
    if before is not None and TONE in before:
        wanted[TONE] = TONE_GOAL_DB
    if before is not None and music_level is not None and OVER_NOTES in before:
        level = max(level, music_level + float(VOCAL_ABOVE_MIDI_DB))
        at_level = before[OVER_NOTES] + level - before[LEVEL]
        lift = min(
            MAX_LIFT_OVER_NOTES_DB, max(0.0, OVER_NOTES_GOAL_DB - at_level)
        )
        wanted[LEVEL] = level + lift
        wanted[OVER_NOTES] = OVER_NOTES_GOAL_DB
    return wanted


def meets(name, value, goal):
    if name == LEVEL:
        return abs(value - goal) <= LEVEL_TOLERANCE_DB
    if name == PEAK:
        return value <= goal + PEAK_TOLERANCE_DB
    if name == PHRASES:
        return value <= goal + PHRASE_SPREAD_TOLERANCE_DB
    if name == OVER_NOTES:
        return value >= goal - OVER_NOTES_TOLERANCE_DB
    if name == NOISE:
        return value <= goal + NOISE_TOLERANCE_DB
    if name == TONE:
        return value <= goal + TONE_TOLERANCE_DB
    return value <= goal


def _quieter_pauses(audio, cut_db):
    mono = _mono(audio)
    spectrum = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH))
    quiet = _pause_frames(frame_levels_db(mono))[: spectrum.shape[1]]
    noise = spectrum[:, : quiet.shape[0]][:, quiet].mean(axis=1, keepdims=True)
    floor = 10.0 ** (-min(float(cut_db), MAX_NOISE_CUT_DB) / 20.0)
    steady = uniform_filter(spectrum, size=NOISE_SMOOTHING, mode="nearest")
    kept = 1.0 - (NOISE_MARGIN * noise / np.maximum(steady, 1e-12)) ** 2
    gain = np.maximum(np.sqrt(np.maximum(kept, 0.0)), floor)
    return _with_gains(audio, gain)


def _with_gains(audio, gains):
    wide = audio[:, np.newaxis] if audio.ndim == 1 else audio
    shaped = np.stack(
        [
            librosa.istft(
                librosa.stft(
                    np.ascontiguousarray(wide[:, channel]),
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                )
                * gains,
                hop_length=HOP_LENGTH,
                length=wide.shape[0],
            )
            for channel in range(wide.shape[1])
        ],
        axis=1,
    ).astype(np.float32)
    return shaped[:, 0] if audio.ndim == 1 else shaped


def _tone_gains(curve, withheld):
    boost = np.maximum(curve, 0.0)[:, np.newaxis]
    cut = np.minimum(curve, 0.0)[:, np.newaxis]
    return 10.0 ** ((cut + boost * (1.0 - withheld[np.newaxis, :])) / 20.0)


def _balanced_tone(audio, sample_rate):
    power, noise_power, freqs = _spectrum(_mono(audio), sample_rate)
    withheld = sibilance(power, freqs)
    fade = np.clip(
        (BALANCE_FADE_TO_HZ - freqs)
        / (BALANCE_FADE_TO_HZ - BALANCE_FADE_FROM_HZ),
        0.0,
        1.0,
    )
    places = np.log2(np.maximum(freqs, 1.0))
    curve = np.zeros_like(freqs)
    gains = np.ones_like(power)
    for _ in range(MAX_PASSES):
        shaping = gains**2
        changes = _tone_changes(
            power * shaping, noise_power * shaping.mean(axis=1), freqs
        )
        if changes is None or all(
            abs(change) <= TONE_TOLERANCE_DB for change in changes.values()
        ):
            break
        step = np.interp(places, np.log2(list(changes)), list(changes.values()))
        curve = np.clip(curve + step, -MAX_TONE_CHANGE_DB, MAX_TONE_CHANGE_DB)
        gains = _tone_gains(
            np.where(curve > 0.0, curve * fade, curve), withheld
        )
    return _with_gains(audio, gains.astype(np.float32))


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
    return _below_true_peak(audio, sample_rate, ceiling)


def _below_true_peak(audio, sample_rate, ceiling):
    limit = ceiling
    for _ in range(MAX_PASSES):
        peak = _true_peak(audio)
        if _decibels(peak) <= PEAK_LIMIT_DB + PEAK_TOLERANCE_DB:
            break
        limit *= ceiling / peak
        audio = _tamed_peaks(audio, sample_rate, limit)
    return audio


def improve(data, sample_rate, music=None, kind=VOCALS):
    sample_rate = validate_sample_rate(sample_rate)
    audio = np.nan_to_num(
        np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    before = measure(audio, sample_rate, music)
    if before[LEVEL] <= SILENT_DB:
        return audio, []
    if kind == MUSIC:
        wanted = music_goals()
    else:
        beside_notes = notes_levels(_mono(audio), music)
        wanted = goals(
            before, None if beside_notes is None else beside_notes[0]
        )
    met = {name: meets(name, before[name], wanted[name]) for name in wanted}
    if all(met.values()):
        return audio, _readings(before, wanted, before, met)
    if not met[OFFSET]:
        audio = _without_offset(audio)
    for _ in range(MAX_PASSES):
        noise = measure(audio, sample_rate)[NOISE]
        if meets(NOISE, noise, wanted[NOISE]):
            break
        audio = _quieter_pauses(
            audio, noise - wanted[NOISE] + NOISE_TOLERANCE_DB
        )
    if TONE in wanted and not met[TONE]:
        audio = _balanced_tone(audio, sample_rate)
    for _ in range(MAX_PASSES if PHRASES in wanted else 0):
        spread = measure(audio, sample_rate)[PHRASES]
        if meets(PHRASES, spread, wanted[PHRASES]):
            break
        audio = _evened_phrases(audio, sample_rate)
    audio = _at_level(audio, sample_rate, wanted[LEVEL])
    audio = np.asarray(audio, dtype=np.float32)
    after = measure(audio, sample_rate, music)
    return audio, _readings(before, wanted, after, met)


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
        if reading.name in (PHRASES, NOISE, TONE) and reading.was_reached:
            continue
        note = "reached" if reading.reached else "not reached"
        parts.append(
            f"{reading.name} {reading.before:.1f} to {reading.after:.1f} dB "
            f"(goal {reading.goal:.1f}, {note})"
        )
    return ", ".join(parts)
