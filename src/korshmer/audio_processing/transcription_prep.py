import os
from dataclasses import asdict, dataclass, replace

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .validation import validate_effect_number

TARGET_RATE = 22050

BAND_LIMITS = {
    "bass": (35.0, 1200.0),
    "piano": (25.0, 5000.0),
    "guitar": (75.0, 4000.0),
    "vocals": (70.0, 1400.0),
    "general": (30.0, 5000.0),
}


BASS_MODEL_SETTINGS = {
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length": 80,
    "minimum_frequency": 30,
    "maximum_frequency": 500,
}

OTHER_MODEL_SETTINGS = {"onset_threshold": 0.5, "frame_threshold": 0.3}


@dataclass(frozen=True)
class PrepOptions:
    mono: bool = True
    highpass_hz: float = 30.0
    band_limit: bool = False
    instrument: str = "general"
    harmonic_only: bool = False
    harmonic_margin: float = 2.0
    transient_amount: float = 0.0
    octave_mix: float = 0.0
    normalize_db: float = -3.0
    compress: bool = False
    enabled: bool = True
    ensemble: bool = True

    def __post_init__(self):
        validate_effect_number(self.highpass_hz, "Highpass frequency", 0, 20000)
        validate_effect_number(self.harmonic_margin, "Harmonic margin", 1, 10)
        validate_effect_number(self.transient_amount, "Transient amount", 0, 1)
        validate_effect_number(self.octave_mix, "Octave mix", 0, 1)
        validate_effect_number(self.normalize_db, "Normalize level", -60, 0)


@dataclass(frozen=True)
class ModelOptions:
    onset_threshold: float = 0.55
    frame_threshold: float = 0.375
    minimum_note_length: float = 80
    minimum_frequency: float = 30
    maximum_frequency: float = 4000

    def as_kwargs(self):
        return asdict(self)

    @classmethod
    def for_stem(cls, stem_name):
        lower = stem_name.lower()
        if "bass" in lower:
            return cls(**BASS_MODEL_SETTINGS)
        if "other" in lower:
            return cls(**OTHER_MODEL_SETTINGS)
        return cls()


ONSET_DELTA = 0.1


def onset_times(path, delta=ONSET_DELTA):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        return librosa.onset.onset_detect(
            y=y, sr=sr, units="time", delta=delta, backtrack=False
        )
    except Exception:
        return np.array([])


def to_mono(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y
    if y.shape[0] <= 8 and y.shape[0] < y.shape[-1]:
        return y.mean(axis=0)
    return y.mean(axis=1)


def highpass(y, sr, cutoff_hz):
    if cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return y
    sos = signal.butter(2, cutoff_hz, "hp", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y - np.mean(y))


def band_limit(y, sr, low_hz, high_hz):
    high_hz = min(high_hz, sr / 2 * 0.95)
    if low_hz >= high_hz:
        return y
    sos = signal.butter(4, [low_hz, high_hz], "bp", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y)


def harmonic_part(y, margin=2.0):
    if len(y) < 2048:
        return y
    return librosa.effects.harmonic(y, margin=float(margin))


def soften_transients(y, sr, amount):
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount == 0.0 or len(y) == 0:
        return y
    rect = np.abs(y)

    def follow(attack_s, release_s):
        a = np.exp(-1.0 / (attack_s * sr))
        r = np.exp(-1.0 / (release_s * sr))
        env = np.empty_like(rect)
        state = 0.0
        for i, v in enumerate(rect):
            coef = a if v > state else r
            state = coef * state + (1 - coef) * v
            env[i] = state
        return env

    fast = follow(0.001, 0.02)
    slow = follow(0.03, 0.1)
    ratio = slow / np.maximum(fast, 1e-9)
    gain = np.clip(ratio, 0.0, 1.0) ** amount
    return y * gain


def octave_layer(y, sr, mix):
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix == 0.0 or len(y) < 4096:
        return y
    up = librosa.effects.pitch_shift(y, sr=sr, n_steps=12)
    return y + mix * up[: len(y)]


def compress(y, threshold=0.5, ratio=3.0):
    mag = np.abs(y)
    over = mag > threshold
    out = y.copy()
    out[over] = np.sign(y[over]) * (threshold + (mag[over] - threshold) / ratio)
    return out


def normalize(y, target_db=-3.0):
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-9:
        return y
    return y * (10 ** (target_db / 20.0) / peak)


def process_audio(y, sr, options):
    y = to_mono(y) if options.mono else np.asarray(y, dtype=np.float64)
    if sr != TARGET_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_RATE)
        sr = TARGET_RATE
    y = highpass(y, sr, options.highpass_hz)
    if options.band_limit:
        low, high = BAND_LIMITS.get(options.instrument, BAND_LIMITS["general"])
        y = band_limit(y, sr, low, high)
    if options.harmonic_only:
        y = harmonic_part(y, options.harmonic_margin)
    if options.transient_amount > 0:
        y = soften_transients(y, sr, options.transient_amount)
    if options.octave_mix > 0:
        y = octave_layer(y, sr, options.octave_mix)
    y = normalize(y, -3.0)
    if options.compress:
        y = compress(y, 0.5)
    return normalize(y, options.normalize_db), sr


def options_for_stem(stem_name, base=None):
    base = base or PrepOptions()
    lower = stem_name.lower()
    if "drum" in lower:
        return replace(base, enabled=False)
    if "bass" in lower:
        return replace(base, instrument="bass", band_limit=True, ensemble=False)
    if "vocal" in lower:
        return replace(base, instrument="vocals", band_limit=True)
    if "guitar" in lower:
        return replace(base, instrument="guitar", band_limit=True)
    if "piano" in lower:
        return replace(base, instrument="piano", band_limit=True)
    return base


def variants_for_stem(stem_name, base=None):
    main = options_for_stem(stem_name, base)
    if not main.enabled or not main.ensemble:
        return [main]
    return [main, replace(main, harmonic_only=True, harmonic_margin=1.0)]


def prepare_for_transcription(audio_path, out_dir, options=None, tag=""):
    options = options or PrepOptions()
    if not options.enabled:
        return audio_path
    try:
        y, sr = sf.read(audio_path, always_2d=False)
        if len(y) == 0:
            return audio_path
        cleaned, rate = process_audio(y, sr, options)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(out_dir, f"{stem}_prep{tag}.wav")
        sf.write(out_path, cleaned.astype(np.float32), rate)
        return out_path
    except Exception:
        return audio_path
