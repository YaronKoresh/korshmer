from __future__ import annotations

import functools
import inspect
import math
from pathlib import Path

import numpy as np

from korshmer.constants import MAX_AUDIO_CHANNELS

Number = int | float


MIN_SAMPLE_RATE: int = 8000


MAX_SAMPLE_RATE: int = 192000


def validate_sample_rate(sample_rate: Number) -> Number:
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
        sample_rate, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Sample rate must be a positive number, got {sample_rate!r}"
        )
    if not math.isfinite(sample_rate):
        raise ValueError(
            f"Sample rate must be a positive number, got {sample_rate!r}"
        )
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, "
            f"got {sample_rate}"
        )
    return sample_rate


MIN_HOP_LENGTH: int = 1


MAX_HOP_LENGTH: int = 16384


MIN_N_FFT: int = 256


MAX_N_FFT: int = 8192


def validate_n_fft(n_fft: int) -> int:
    if isinstance(n_fft, bool) or not isinstance(n_fft, (int, np.integer)):
        raise ValueError(f"n_fft must be an integer, got {n_fft!r}")
    n_fft = int(n_fft)
    if n_fft < MIN_N_FFT or n_fft > MAX_N_FFT:
        raise ValueError(
            f"n_fft must be between {MIN_N_FFT} and {MAX_N_FFT}, got {n_fft}"
        )
    if n_fft & (n_fft - 1) != 0:
        raise ValueError(f"n_fft must be a power of 2, got {n_fft}")
    return n_fft


def validate_hop_length(hop_length: int, n_fft: int | None = None) -> int:
    if isinstance(hop_length, (bool, np.bool_)) or not isinstance(
        hop_length, (int, np.integer)
    ):
        raise ValueError(f"hop_length must be an integer, got {hop_length!r}")
    if hop_length < MIN_HOP_LENGTH:
        raise ValueError(
            f"hop_length must be >= {MIN_HOP_LENGTH}, got {hop_length}"
        )
    if n_fft is not None and hop_length > n_fft:
        raise ValueError(
            f"hop_length ({hop_length}) must be <= n_fft ({n_fft})"
        )
    if hop_length > MAX_HOP_LENGTH:
        raise ValueError(
            f"hop_length must be <= {MAX_HOP_LENGTH}, got {hop_length}"
        )
    return hop_length


MIN_Q_FACTOR: float = 0.1


MAX_Q_FACTOR: float = 20.0


def validate_q_factor(q_factor: Number) -> Number:
    if isinstance(q_factor, (bool, np.bool_)) or not isinstance(
        q_factor, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Q factor must be a finite number, got {q_factor!r}")
    if not math.isfinite(q_factor):
        raise ValueError(f"Q factor must be a finite number, got {q_factor!r}")
    if q_factor < MIN_Q_FACTOR or q_factor > MAX_Q_FACTOR:
        raise ValueError(
            f"Q factor must be between {MIN_Q_FACTOR:g} and {MAX_Q_FACTOR:g}, "
            f"got {q_factor!r}"
        )
    return q_factor


MAX_EQ_BANDS: int = 31


def validate_eq_band_count(band_count: int) -> int:
    if isinstance(band_count, (bool, np.bool_)) or not isinstance(
        band_count, (int, np.integer)
    ):
        raise ValueError(
            f"EQ band count must be an integer, got {band_count!r}"
        )
    if band_count < 1 or band_count > MAX_EQ_BANDS:
        raise ValueError(
            f"EQ band count must be between 1 and {MAX_EQ_BANDS}, "
            f"got {band_count!r}"
        )
    return band_count


MIN_COMPRESSION_RATIO: float = 1.0


MAX_COMPRESSION_RATIO: float = 20.0


def validate_compression_ratio(ratio: Number) -> Number:
    if isinstance(ratio, (bool, np.bool_)) or not isinstance(
        ratio, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Compression ratio must be a finite number, got {ratio!r}"
        )
    if not math.isfinite(ratio):
        raise ValueError(
            f"Compression ratio must be a finite number, got {ratio!r}"
        )
    if ratio < MIN_COMPRESSION_RATIO or ratio > MAX_COMPRESSION_RATIO:
        raise ValueError(
            f"Compression ratio must be between {MIN_COMPRESSION_RATIO:g} and "
            f"{MAX_COMPRESSION_RATIO:g}, got {ratio!r}"
        )
    return ratio


MIN_REVERB_DECAY: float = 0.0


MAX_REVERB_DECAY: float = 1.0


def validate_reverb_decay(decay: Number) -> Number:
    if isinstance(decay, (bool, np.bool_)) or not isinstance(
        decay, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Reverb decay must be a finite number, got {decay!r}")
    if not math.isfinite(decay):
        raise ValueError(f"Reverb decay must be a finite number, got {decay!r}")
    if decay < MIN_REVERB_DECAY or decay > MAX_REVERB_DECAY:
        raise ValueError(
            f"Reverb decay must be between {MIN_REVERB_DECAY:g} and "
            f"{MAX_REVERB_DECAY:g}, got {decay!r}"
        )
    return decay


MIN_REVERB_DECAY_TIME_S: float = 0.1


MAX_REVERB_DECAY_TIME_S: float = 30.0


def validate_reverb_decay_time(decay_time_s: Number) -> Number:
    if isinstance(decay_time_s, (bool, np.bool_)) or not isinstance(
        decay_time_s, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Reverb decay time must be a finite number, got {decay_time_s!r}"
        )
    if not math.isfinite(decay_time_s):
        raise ValueError(
            f"Reverb decay time must be a finite number, got {decay_time_s!r}"
        )
    if (
        decay_time_s < MIN_REVERB_DECAY_TIME_S
        or decay_time_s > MAX_REVERB_DECAY_TIME_S
    ):
        raise ValueError(
            f"Reverb decay time must be between {MIN_REVERB_DECAY_TIME_S:g} and "
            f"{MAX_REVERB_DECAY_TIME_S:g} seconds, got {decay_time_s!r}"
        )
    return decay_time_s


MIN_PITCH_SHIFT_SEMITONES: float = -24.0

MAX_PITCH_SHIFT_SEMITONES: float = 24.0


def validate_pitch_shift_semitones(semitones: Number) -> Number:
    if isinstance(semitones, (bool, np.bool_)) or not isinstance(
        semitones, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Pitch shift must be a finite number, got {semitones!r}"
        )
    if not math.isfinite(semitones):
        raise ValueError(
            f"Pitch shift must be a finite number, got {semitones!r}"
        )
    if (
        semitones < MIN_PITCH_SHIFT_SEMITONES
        or semitones > MAX_PITCH_SHIFT_SEMITONES
    ):
        raise ValueError(
            f"Pitch shift must be between {MIN_PITCH_SHIFT_SEMITONES:g} and "
            f"{MAX_PITCH_SHIFT_SEMITONES:g} semitones, got {semitones!r}"
        )
    return semitones


MIN_DELAY_TIME_MS: float = 1.0


MAX_DELAY_TIME_MS: float = 10000.0


def validate_delay_time(delay_ms: Number) -> Number:
    if isinstance(delay_ms, (bool, np.bool_)) or not isinstance(
        delay_ms, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Delay time must be a finite number, got {delay_ms!r}"
        )
    if not math.isfinite(delay_ms):
        raise ValueError(
            f"Delay time must be a finite number, got {delay_ms!r}"
        )
    if delay_ms < MIN_DELAY_TIME_MS or delay_ms > MAX_DELAY_TIME_MS:
        raise ValueError(
            f"Delay time must be between {MIN_DELAY_TIME_MS:g} and "
            f"{MAX_DELAY_TIME_MS:g} ms, got {delay_ms!r}"
        )
    return delay_ms


def validate_effect_number(
    value: Number, name: str, minimum: Number, maximum: Number
) -> Number:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}, got {value!r}"
        )
    return value


def validate_effect_integer(
    value: int, name: str, minimum: int, maximum: int
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value!r}"
        )
    return int(value)


MIN_NORMALIZE_TARGET_DB: float = -60.0


MAX_NORMALIZE_TARGET_DB: float = 0.0


def validate_normalize_target_db(target_db: Number) -> Number:
    if isinstance(target_db, (bool, np.bool_)) or not isinstance(
        target_db, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Normalize target must be a finite number in dB, got {target_db!r}"
        )
    if not math.isfinite(target_db):
        raise ValueError(
            f"Normalize target must be a finite number in dB, got {target_db!r}"
        )
    if (
        target_db < MIN_NORMALIZE_TARGET_DB
        or target_db > MAX_NORMALIZE_TARGET_DB
    ):
        raise ValueError(
            f"Normalize target must be between {MIN_NORMALIZE_TARGET_DB:g} and "
            f"{MAX_NORMALIZE_TARGET_DB:g} dB, got {target_db!r}"
        )
    return target_db


MIN_AUDIO_CHANNELS: int = 1


def validate_channel_count(
    channels: int, max_channels: int = MAX_AUDIO_CHANNELS
) -> int:
    if isinstance(channels, (bool, np.bool_)) or not isinstance(
        channels, (int, np.integer)
    ):
        raise ValueError(
            f"Audio channel count must be an integer, got {channels!r}"
        )
    channels = int(channels)
    if channels < MIN_AUDIO_CHANNELS or channels > max_channels:
        raise ValueError(
            f"Audio has {channels} channels; supported range is "
            f"{MIN_AUDIO_CHANNELS} to {max_channels}"
        )
    return channels


def audio_channel_count(audio, channels_first: bool = True) -> int:
    array = np.asarray(audio)
    if array.ndim <= 1:
        return 1
    if array.ndim != 2:
        raise ValueError(
            f"Audio array must be 1-D or 2-D, got {array.ndim} dimensions"
        )
    return int(array.shape[0] if channels_first else array.shape[1])


EMPTY_AUDIO_MESSAGE: str = "Audio array is empty"


def is_empty_audio(audio) -> bool:
    if audio is None:
        return True
    size = getattr(audio, "size", None)
    if size is None:
        size = np.asarray(audio).size
    return size == 0


def require_output_target(value, what: str):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(
            f"{what} is required when processing an audio file, got {value!r}"
        )
    return value


def validate_audio_not_empty(audio):
    if is_empty_audio(audio):
        raise ValueError(EMPTY_AUDIO_MESSAGE)
    return audio


def validate_audio_finite(audio):
    if not np.isfinite(np.asarray(audio)).all():
        raise ValueError("Audio contains NaN or infinite values")
    return audio


ALLOWED_EXPORT_BIT_DEPTHS: tuple[int, ...] = (16, 24, 32)


def validate_bit_depth(bit_depth: int) -> int:
    if isinstance(bit_depth, (bool, np.bool_)) or not isinstance(
        bit_depth, (int, np.integer)
    ):
        raise ValueError(f"Bit depth must be an integer, got {bit_depth!r}")
    bit_depth = int(bit_depth)
    if bit_depth not in ALLOWED_EXPORT_BIT_DEPTHS:
        allowed = ", ".join(str(d) for d in ALLOWED_EXPORT_BIT_DEPTHS)
        raise ValueError(f"Bit depth must be one of {allowed}, got {bit_depth}")
    return bit_depth


MIN_SPECTRAL_GATE_THRESHOLD_DB: float = -120.0
MAX_SPECTRAL_GATE_THRESHOLD_DB: float = 0.0


def validate_spectral_gate_threshold(threshold_db: Number) -> Number:
    if isinstance(threshold_db, (bool, np.bool_)) or not isinstance(
        threshold_db, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Spectral gate threshold must be a finite number, got "
            f"{threshold_db!r}"
        )
    if not math.isfinite(threshold_db):
        raise ValueError(
            f"Spectral gate threshold must be a finite number, got "
            f"{threshold_db!r}"
        )
    if (
        threshold_db < MIN_SPECTRAL_GATE_THRESHOLD_DB
        or threshold_db > MAX_SPECTRAL_GATE_THRESHOLD_DB
    ):
        raise ValueError(
            f"Spectral gate threshold must be between "
            f"{MIN_SPECTRAL_GATE_THRESHOLD_DB:g} and "
            f"{MAX_SPECTRAL_GATE_THRESHOLD_DB:g} dB, got {threshold_db!r}"
        )
    return threshold_db


VALID_TIME_SIGNATURE_DENOMINATORS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
MIN_TIME_SIGNATURE_NUMERATOR: int = 1
MAX_TIME_SIGNATURE_NUMERATOR: int = 64


def validate_time_signature(time_signature):
    if (
        not isinstance(time_signature, (tuple, list))
        or len(time_signature) != 2
    ):
        raise ValueError(
            "Time signature must be a (numerator, denominator) pair, got "
            f"{time_signature!r}"
        )
    numerator, denominator = time_signature
    for label, value in (
        ("numerator", numerator),
        ("denominator", denominator),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(
                f"Time signature {label} must be an integer, got {value!r}"
            )
    numerator = int(numerator)
    denominator = int(denominator)
    if not (
        MIN_TIME_SIGNATURE_NUMERATOR
        <= numerator
        <= MAX_TIME_SIGNATURE_NUMERATOR
    ):
        raise ValueError(
            f"Time signature numerator must be between "
            f"{MIN_TIME_SIGNATURE_NUMERATOR} and "
            f"{MAX_TIME_SIGNATURE_NUMERATOR}, got {numerator}"
        )
    if denominator not in VALID_TIME_SIGNATURE_DENOMINATORS:
        allowed = ", ".join(str(d) for d in VALID_TIME_SIGNATURE_DENOMINATORS)
        raise ValueError(
            f"Time signature denominator must be one of {allowed}, got "
            f"{denominator}"
        )
    return (numerator, denominator)


MIN_PAN_POSITION: float = -100.0
MAX_PAN_POSITION: float = 100.0


def validate_pan_position(pan: Number) -> Number:
    if isinstance(pan, (bool, np.bool_)) or not isinstance(
        pan, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Pan position must be a finite number, got {pan!r}")
    if not math.isfinite(pan):
        raise ValueError(f"Pan position must be a finite number, got {pan!r}")
    if pan < MIN_PAN_POSITION or pan > MAX_PAN_POSITION:
        raise ValueError(
            f"Pan position must be between {MIN_PAN_POSITION:g} and "
            f"{MAX_PAN_POSITION:g}, got {pan!r}"
        )
    return pan


def pan_to_unit(pan: Number) -> float:
    return float(pan) / 100.0


def validate_fade_duration(
    fade_duration_s: Number, region_duration_s: Number
) -> Number:
    if isinstance(fade_duration_s, (bool, np.bool_)) or not isinstance(
        fade_duration_s, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Fade duration must be a finite number, got {fade_duration_s!r}"
        )
    if not math.isfinite(fade_duration_s):
        raise ValueError(
            f"Fade duration must be a finite number, got {fade_duration_s!r}"
        )
    if fade_duration_s <= 0 or fade_duration_s > region_duration_s:
        raise ValueError(
            f"Fade duration must be between 0 (exclusive) and the audio "
            f"length of {region_duration_s:g} s, got {fade_duration_s!r}"
        )
    return fade_duration_s


MIN_CONFIDENCE_THRESHOLD: float = 0.0
MAX_CONFIDENCE_THRESHOLD: float = 1.0


DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3


def validate_confidence_threshold(threshold: Number) -> Number:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Confidence threshold must be a finite number, got {threshold!r}"
        )
    if not math.isfinite(threshold):
        raise ValueError(
            f"Confidence threshold must be a finite number, got {threshold!r}"
        )
    if not MIN_CONFIDENCE_THRESHOLD <= threshold <= MAX_CONFIDENCE_THRESHOLD:
        raise ValueError(
            f"Confidence threshold must be between "
            f"{MIN_CONFIDENCE_THRESHOLD:g} and {MAX_CONFIDENCE_THRESHOLD:g}, "
            f"got {threshold!r}"
        )
    return threshold


def _audio_argument(func):
    params = [
        name
        for name in inspect.signature(func).parameters
        if name not in ("self", "cls")
    ]
    if not params:
        raise TypeError(f"{func.__qualname__} takes no audio argument")
    name = params[0]
    is_method = "self" in inspect.signature(func).parameters
    position = 1 if is_method else 0

    def get(args, kwargs):
        if len(args) > position:
            return args[position]
        return kwargs.get(name)

    return get


def passthrough_empty_audio(func):
    get_audio = _audio_argument(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        audio = get_audio(args, kwargs)
        if audio is None:
            raise ValueError(EMPTY_AUDIO_MESSAGE)
        if is_empty_audio(audio):
            return np.array(audio, copy=True)
        return func(*args, **kwargs)

    return wrapper


def reject_empty_audio(func):
    get_audio = _audio_argument(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        validate_audio_not_empty(get_audio(args, kwargs))
        return func(*args, **kwargs)

    return wrapper


def check_range(**ranges):
    if not ranges:
        raise TypeError("check_range needs at least one name=(min, max) range")
    for name, bounds in ranges.items():
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise TypeError(
                f"check_range bounds for {name!r} must be a (min, max) tuple"
            )
        if bounds[0] > bounds[1]:
            raise ValueError(
                f"check_range minimum for {name!r} is above its maximum"
            )

    def decorator(func):
        signature = inspect.signature(func)
        missing = [n for n in ranges if n not in signature.parameters]
        if missing:
            raise TypeError(
                f"{func.__qualname__} has no argument named {missing[0]!r}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            for name, (low, high) in ranges.items():
                if name not in bound.arguments:
                    continue
                value = bound.arguments[name]
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(
                        value, (int, float, np.integer, np.floating)
                    )
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"{name} must be a finite number between "
                        f"{low:g} and {high:g}, got {value!r}"
                    )
                if value < low or value > high:
                    raise ValueError(
                        f"{name} must be between {low:g} and {high:g}, "
                        f"got {value!r}"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_audio_params(
    func=None,
    *,
    max_channels: int = MAX_AUDIO_CHANNELS,
    channels_first: bool = True,
    empty: str = "raise",
):
    if empty not in ("raise", "passthrough"):
        raise ValueError("empty must be 'raise' or 'passthrough'")

    def decorator(f):
        get_audio = _audio_argument(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            audio = get_audio(args, kwargs)
            if audio is None:
                raise ValueError(EMPTY_AUDIO_MESSAGE)
            if is_empty_audio(audio):
                if empty == "passthrough":
                    return np.array(audio, copy=True)
                raise ValueError(EMPTY_AUDIO_MESSAGE)
            validate_channel_count(
                audio_channel_count(audio, channels_first=channels_first),
                max_channels,
            )
            return f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
