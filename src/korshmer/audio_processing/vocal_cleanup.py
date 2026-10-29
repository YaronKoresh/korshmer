import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter, uniform_filter1d

from korshmer.audio_processing.validation import validate_sample_rate

HIGHPASS_HZ = 70.0
N_FFT = 2048
HOP_LENGTH = 512
NOISE_OVER_SUBTRACTION = 1.0
MASK_FLOOR = 0.55
MASK_SMOOTHING = (3, 5)
ACTIVE_RANGE_DB = 30.0
GATE_RANGE_DB = 45.0
MIN_GATE_RANGE_DB = 20.0
GATE_FLOOR_SHARE = 0.15
FLOOR_PERCENTILE = 10.0
PAUSE_ATTENUATION = 0.35
GATE_HOLD_FRAMES = 6
GATE_SMOOTHING_FRAMES = 5
PEAK_CEILING = 0.99
EPSILON = 1e-10
VOCAL_ABOVE_MIDI_DB = -4.0
MIN_VOCAL_GAIN = 0.1
MAX_VOCAL_GAIN = 4.0


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


def _highpass(channel, sample_rate):
    if sample_rate <= 2 * HIGHPASS_HZ * 2:
        return channel
    sos = signal.butter(
        4, HIGHPASS_HZ, btype="highpass", fs=sample_rate, output="sos"
    )
    if channel.shape[0] <= 3 * (2 * sos.shape[0] + 1):
        return signal.sosfilt(sos, channel).astype(np.float32)
    return signal.sosfiltfilt(sos, channel).astype(np.float32)


def _spectral_mask(magnitude, speaking):
    mask = np.ones_like(magnitude)
    quiet = ~speaking
    if not np.any(quiet):
        return mask
    noise = np.mean(magnitude[:, quiet], axis=1, keepdims=True)
    cleaned = np.maximum(magnitude - NOISE_OVER_SUBTRACTION * noise, 0.0)
    reduced = cleaned / np.maximum(magnitude, EPSILON)
    mask[:, quiet] = np.clip(reduced[:, quiet], MASK_FLOOR, 1.0)
    return uniform_filter(mask, size=MASK_SMOOTHING, mode="nearest")


def _speaking_frames(levels_db, frames):
    everything = np.ones(frames, dtype=bool)
    if levels_db.size == 0:
        return everything
    top = float(np.percentile(levels_db, 95))
    floor = float(np.percentile(levels_db, FLOOR_PERCENTILE))
    if top - floor < MIN_GATE_RANGE_DB:
        return everything
    threshold = max(
        top - GATE_RANGE_DB, floor + GATE_FLOOR_SHARE * (top - floor)
    )
    active = levels_db > threshold
    if active.size < frames:
        active = np.pad(active, (0, frames - active.size), constant_values=True)
    return active[:frames]


def _pause_envelope(active):
    held = maximum_filter1d(
        active.astype(np.float32), size=GATE_HOLD_FRAMES, mode="nearest"
    )
    smooth = uniform_filter1d(held, size=GATE_SMOOTHING_FRAMES, mode="nearest")
    return PAUSE_ATTENUATION + (1.0 - PAUSE_ATTENUATION) * smooth


def _gate_gains(mono):
    magnitude = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH))
    speaking = _speaking_frames(frame_levels_db(mono), magnitude.shape[1])
    mask = _spectral_mask(magnitude, speaking)
    return mask * _pause_envelope(speaking)[np.newaxis, :]


def _clean_channels(channels, sample_rate):
    filtered = [_highpass(channel, sample_rate) for channel in channels]
    length = filtered[0].shape[0]
    if length < N_FFT:
        return filtered
    gains = _gate_gains(np.mean(filtered, axis=0, dtype=np.float32))
    return [
        librosa.istft(
            librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH) * gains,
            hop_length=HOP_LENGTH,
            length=length,
        )
        for channel in filtered
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
