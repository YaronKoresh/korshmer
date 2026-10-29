import math

import librosa
import numpy as np
import scipy.fft

from .validation import DEFAULT_CONFIDENCE_THRESHOLD, validate_sample_rate


def _validate_pitch_range(sr, fmin, fmax) -> None:
    validate_sample_rate(sr)
    for name, value in (("fmin", fmin), ("fmax", fmax)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive finite number, got {value!r}"
            )
    if fmin >= fmax:
        raise ValueError(f"fmin must be below fmax, got {fmin} and {fmax}")


def _autocorrelation_lags(signal: np.ndarray, lag_count: int) -> np.ndarray:
    samples = np.asarray(signal, dtype=np.float64)
    size = scipy.fft.next_fast_len(len(samples) + lag_count)
    spectrum = scipy.fft.rfft(samples, size)
    power = spectrum.real**2 + spectrum.imag**2
    return scipy.fft.irfft(power, size)[:lag_count]


def autocorrelation_pitch(
    audio_segment: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[float, float]:
    _validate_pitch_range(sr, fmin, fmax)
    if len(audio_segment) == 0:
        return 0, 0.0

    min_period = max(1, int(sr / fmax))
    max_period = int(sr / fmin)

    if max_period >= len(audio_segment):
        return 0, 0.0

    autocorr = _autocorrelation_lags(audio_segment, max_period + 1)

    if autocorr[0] == 0:
        return 0, 0.0

    autocorr = autocorr / autocorr[0]

    search_range = autocorr[min_period:max_period]
    if len(search_range) == 0:
        return 0, 0.0

    peak_idx = np.argmax(search_range) + min_period
    if peak_idx == 0:
        return 0, 0.0

    confidence = autocorr[peak_idx]

    if confidence > confidence_threshold:
        pitch_hz = sr / peak_idx
        return pitch_hz, confidence

    return 0, 0.0


def piptrack_pitch(
    audio_segment: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> tuple[float, float]:
    _validate_pitch_range(sr, fmin, fmax)
    pitches, magnitudes = librosa.piptrack(
        y=audio_segment, sr=sr, fmin=fmin, fmax=fmax
    )

    max_idx = np.unravel_index(np.argmax(magnitudes), magnitudes.shape)
    return pitches[max_idx], magnitudes[max_idx]
