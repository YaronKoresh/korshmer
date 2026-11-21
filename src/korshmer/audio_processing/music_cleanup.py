import librosa
import numpy as np

from korshmer.audio_processing.validation import validate_sample_rate
from korshmer.audio_processing.vocal_cleanup import (
    HOP_LENGTH,
    N_FFT,
    PEAK_CEILING,
    highpass,
    istft,
    loudness_gain,
    noise_floor_power,
    noise_gains,
    pause_gains,
)

RUMBLE_HZ = 25.0
NOISE_OVER_SUBTRACTION = 3.0
NOISE_MAX_CUT_DB = 12.0
PAUSE_SNR_DB = 4.0
PAUSE_RATIO = 4.0
PAUSE_MAX_CUT_DB = 15.0


def _clean_channels(channels, sample_rate):
    filtered = [
        highpass(channel, sample_rate, RUMBLE_HZ) for channel in channels
    ]
    length = filtered[0].shape[0]
    if length < N_FFT:
        return filtered
    mono = np.mean(filtered, axis=0, dtype=np.float32)
    magnitude = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH))
    noise = noise_floor_power(magnitude, sample_rate)
    gains = (
        noise_gains(magnitude, noise, NOISE_OVER_SUBTRACTION, NOISE_MAX_CUT_DB)
        * pause_gains(
            magnitude,
            noise,
            sample_rate,
            PAUSE_SNR_DB,
            PAUSE_RATIO,
            PAUSE_MAX_CUT_DB,
        )[np.newaxis, :]
    )
    outputs = [
        istft(
            librosa.stft(channel, n_fft=N_FFT, hop_length=HOP_LENGTH),
            gains,
            length,
        )
        for channel in filtered
    ]
    level = np.float32(loudness_gain(np.mean(outputs, axis=0), mono))
    return [output * level for output in outputs]


def clean_music(data, sample_rate):
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
    ).astype(np.float32)
    if audio.ndim == 1:
        cleaned = cleaned[:, 0]
    peak = float(np.max(np.abs(cleaned)))
    if peak > PEAK_CEILING:
        cleaned *= np.float32(PEAK_CEILING / peak)
    return cleaned
