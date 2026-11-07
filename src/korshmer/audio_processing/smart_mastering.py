import logging
from dataclasses import dataclass, field, replace
from itertools import pairwise

import librosa
import numpy as np
from numba import njit
from scipy import signal
from scipy.ndimage import (
    gaussian_filter1d,
    minimum_filter1d,
    uniform_filter1d,
)
from scipy.signal import (
    butter,
    firwin,
    firwin2,
    lfilter,
    resample_poly,
    sosfilt,
    sosfiltfilt,
    welch,
)

from ..exceptions import InvalidAudioInputError
from .generate_bands import generate_bands
from .validation import (
    EMPTY_AUDIO_MESSAGE,
    is_empty_audio,
    passthrough_empty_audio,
    validate_audio_not_empty,
    validate_fade_duration,
    validate_sample_rate,
)

logger = logging.getLogger(__name__)


EXCITER_LOW_HZ = 1400.0
EXCITER_HIGH_HZ = 22000.0
EXCITER_BAND_COUNT = 16
EXCITER_FILTER_ORDER = 4
EXCITER_MAX_HARMONIC_ORDER = 5
EXCITER_PEAK_DRIVE = 1.2
EXCITER_NYQUIST_FRACTION = 0.95
EXCITER_SILENCE_LEVEL = 1e-9
EXCITER_KAISER_BETA = 12.0
EXCITER_GUARD_RATIO = 10.0
EXCITER_NPERSEG = 8192
EXCITER_FIR_TAPS = 4097
EXCITER_MIN_SAMPLES = 1024
EXCITER_BANDS_PER_OCTAVE = 6
EXCITER_FILL_SLOPE_DB_PER_OCTAVE = -0.5
EXCITER_MAX_FILL_GAIN_DB = 60.0
EXCITER_POWER_FLOOR = 1e-24
OVERSAMPLING_TAPS_PER_FACTOR = 40
EXCITER_HARMONIC_CAP_DB_LOW = -12.0
EXCITER_HARMONIC_CAP_DB_HIGH = -8.0
EXCITER_CAP_SPLIT_HZ = 5000.0
EXCITER_DUCK_ATTACK_MS = 20.0
EXCITER_DUCK_RELEASE_MS = 150.0
EXCITER_ORDER_SPLIT_HZ = 2000.0
EXCITER_MAX_HARMONIC_ORDER_LOW = 3
EXCITER_MAX_DRIVE = 2.0
TRUE_PEAK_GUARD_DB = 0.15
EXPANSION_MIN_SAMPLES = 2048


@dataclass
class MixParams:
    weight: float = 0.3
    fade_duration: float = 2.0
    duck_depth: float = 0.3
    cutoff_hz: float = None


@dataclass
class MultibandBand:
    fc: float
    base_threshold: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_db: float
    knee_db: float

    upward_boost_db: float = 2.0
    upward_low_thr_db: float = -42.0
    upward_high_thr_db: float = -18.0


@dataclass
class MultibandParams:
    """Multiband compressor with combined downward control + upward detail retrieval."""

    num_bands: int = 5
    intensity: float = 1.0

    bass_ratio: float = 2.8
    bass_attack_ms: float = 35.0
    bass_release_ms: float = 180.0
    bass_threshold_db: float = -18.0

    treb_ratio: float = 1.8
    treb_attack_ms: float = 15.0
    treb_release_ms: float = 80.0
    treb_threshold_db: float = -14.0

    knee_db: float = 8.0
    relative_threshold: bool = True
    auto_makeup: bool = True
    enable_upward_detail: bool = True

    def make_bands(
        self, fcs: list[float], freq_min: float, freq_max: float
    ) -> list[MultibandBand]:
        if not len(fcs):
            return []

        fcs_arr = np.array(fcs, dtype=float)
        ref_min = np.log2(max(freq_min, 1.0))
        ref_max = np.log2(max(freq_max, freq_min + 1.0))
        span = max(ref_max - ref_min, 1e-9)

        fcs_safe = np.clip(fcs_arr, freq_min, freq_max)
        positions = (np.log2(fcs_safe) - ref_min) / span

        base_thr = (
            self.bass_threshold_db
            + (self.treb_threshold_db - self.bass_threshold_db) * positions
        )
        ratios = (
            self.bass_ratio + (self.treb_ratio - self.bass_ratio) * positions
        )
        attacks = (
            self.bass_attack_ms
            + (self.treb_attack_ms - self.bass_attack_ms) * positions
        )
        releases = (
            self.bass_release_ms
            + (self.treb_release_ms - self.bass_release_ms) * positions
        )

        thr = (
            np.mean(base_thr) + (base_thr - np.mean(base_thr)) * self.intensity
        )
        ratios = 1.0 + (ratios - 1.0) * self.intensity
        attacks = np.maximum(attacks, 15.0)
        releases = np.maximum(releases, 40.0)

        makeups = np.zeros_like(fcs_arr, dtype=float)
        knees = np.full_like(fcs_arr, self.knee_db, dtype=float)

        bands = []
        for i in range(len(fcs_arr)):
            fc = float(fcs_arr[i])
            if 300.0 <= fc <= 6000.0:
                up_boost = 2.0 * self.intensity
            elif fc < 300.0:
                up_boost = 1.0 * self.intensity
            else:
                up_boost = 1.5 * self.intensity

            bands.append(
                MultibandBand(
                    fc=fc,
                    base_threshold=float(thr[i]),
                    ratio=float(ratios[i]),
                    attack_ms=float(max(attacks[i], 15.0)),
                    release_ms=float(max(releases[i], 40.0)),
                    makeup_db=float(makeups[i]),
                    knee_db=float(knees[i]),
                    upward_boost_db=(
                        float(up_boost) if self.enable_upward_detail else 0.0
                    ),
                )
            )
        return bands


@dataclass(frozen=True)
class ExciterSettings:
    drive: float | None
    oversample: int
    ceiling_hz: float


@dataclass
class LimiterParams:
    drive_db: float | None = None
    ceil_db: float = -0.3
    max_drive_db: float = 10.0

    lookahead_ms: float = 5.0
    release_fast_ms: float = 30.0
    release_slow_ms: float = 240.0
    slow_weight: float = 0.3

    clip_ceiling_db: float = 0.0
    clip_knee: float = 8.0

    os_factor: int = 4
    kaiser_beta: float = 12.0
    margin_db: float = 0.3

    split_hz: float | None = 250.0
    band_ceiling_lo: float = 0.88
    bass_clip_db: float = 1.2
    bass_clip_knee: float = 0.4
    lo_lookahead_ms: float = 5.0
    lo_release_fast_ms: float = 45.0
    lo_release_slow_ms: float = 300.0
    lo_slow_weight: float = 0.5

    loudness_tol_lu: float = 0.1
    min_efficiency_lu_per_db: float = 0.1
    max_iterations: int = 7


@dataclass
class SmartMasteringConfig:
    sample_rate: int | None = None
    slope_db: float = 3.3
    slope_hz: float = 1000.0
    smoothing_fraction: float = 0.0
    mix: MixParams = field(default_factory=MixParams)
    multiband: MultibandParams = field(default_factory=MultibandParams)
    limiter: LimiterParams = field(default_factory=LimiterParams)
    target_lufs: float = -8.5
    work_lufs: float = -18.0
    correction_strength: float = 1.0
    max_correction_db: float = 60.0

    bass_db: float = 2.0
    bass_hz: float = 60.0
    bass_octaves: float = 1.5
    warmth_db: float = 0.8
    warmth_hz: float = 180.0
    warmth_octaves: float = 1.3
    presence_db: float = 6.0
    presence_hz: float = 3800.0
    presence_octaves: float = 1.5
    air_db: float = 2.0
    air_hz: float = 12000.0
    air_octaves: float = 1.8

    bass_enhance_mix: float = 0.2
    bass_mono_hz: float | None = 160.0
    exciter_mix: float = 8.0
    stereo_width: float = 1.15

    low_cut: int | None = 30
    high_cut: int | None = 20000
    phase_type: str = "minimal"

    expansion_db: float = 6.0
    expansion_low_hz: float = 800.0
    expansion_high_hz: float = 6000.0
    expansion_bands: int = 4
    expansion_window_ms: float = 3000.0
    expansion_attack_ms: float = 20.0
    expansion_release_ms: float = 200.0
    expansion_ratio: float = 0.5
    pre_limiter_rise_lu: float = 1.5


@passthrough_empty_audio
def resample(y, sr, target_sr=44100):
    validate_sample_rate(sr)
    validate_sample_rate(target_sr)
    if np.issubdtype(y.dtype, np.integer):
        y = librosa.util.buf_to_float(y)
    else:
        y = y.astype(np.float64)
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)


@njit(cache=True, fastmath=True)
def _decoupled_envelope(env_db, attack_coef, release_coef):
    out = np.empty_like(env_db)
    prev = env_db[0]
    for i in range(len(env_db)):
        coef = attack_coef if env_db[i] > prev else release_coef
        prev = coef * prev + (1.0 - coef) * env_db[i]
        out[i] = prev
    return out


@njit(cache=True, fastmath=True)
def _limiter_release(gs_db, rc_fast, rc_slow, w_slow):
    n = gs_db.shape[0]
    out = np.empty(n)
    f = 0.0
    s = 0.0
    for i in range(n):
        g = gs_db[i]
        f = min(g, f * rc_fast)
        s = min(g, s * rc_slow)
        out[i] = w_slow * s + (1.0 - w_slow) * f
    return out


def _build_oversampling_kernel(
    sr: int, os_factor: int, kaiser_beta: float
) -> np.ndarray:
    if os_factor <= 1:
        return np.ones(1)
    return firwin(
        OVERSAMPLING_TAPS_PER_FACTOR * os_factor + 1,
        sr / 2.0,
        window=("kaiser", kaiser_beta),
        fs=sr * os_factor,
    )


def _saturate_analog_warmth(
    band: np.ndarray, sr: int, drive: float, oversample: int
) -> np.ndarray:
    """Asymmetric tube/console saturation:

    Combines 3rd harmonic bite with warm 2nd harmonic richness. Eliminates
    tractor buzz while adding expensive aggression.
    """

    def curve(x):

        xd = x * drive
        return np.tanh(xd) + 0.14 * xd * np.abs(xd) / (1.0 + np.abs(xd))

    if oversample <= 1:
        return curve(band)

    kernel = _build_oversampling_kernel(sr, oversample, EXCITER_KAISER_BETA)
    rows = []
    for row in band:
        up = resample_poly(row, up=oversample, down=1, window=kernel)
        distorted = curve(up.astype(np.float32))
        down = resample_poly(distorted, up=1, down=oversample, window=kernel)[
            : row.shape[-1]
        ]
        rows.append(down)
    return np.stack(rows).astype(np.float64)


def _excite_band(
    y: np.ndarray,
    sr: int,
    band_hz: tuple[float, float],
    settings: ExciterSettings,
) -> np.ndarray:
    nyquist = sr / 2.0
    low_hz, high_hz = band_hz

    f_low = max(low_hz, 20.0)
    f_high = min(high_hz, settings.ceiling_hz * 0.98)
    if f_low >= f_high - 10.0:
        return np.zeros_like(y)

    band_sos = butter(
        EXCITER_FILTER_ORDER,
        [f_low / nyquist, f_high / nyquist],
        btype="band",
        output="sos",
    )
    band = sosfilt(band_sos, y)
    peak_band = np.max(np.abs(band))
    if peak_band <= EXCITER_SILENCE_LEVEL:
        return np.zeros_like(y)

    drive = (
        min(EXCITER_PEAK_DRIVE / (peak_band + 1e-6), EXCITER_MAX_DRIVE)
        if settings.drive is None
        else min(settings.drive, EXCITER_MAX_DRIVE)
    )
    order = (
        EXCITER_MAX_HARMONIC_ORDER_LOW
        if f_high <= EXCITER_ORDER_SPLIT_HZ
        else EXCITER_MAX_HARMONIC_ORDER
    )
    is_oversampling_needed = f_high * order > nyquist
    saturated = _saturate_analog_warmth(
        band, sr, drive, settings.oversample if is_oversampling_needed else 2
    )

    guard_low = f_high * EXCITER_GUARD_RATIO
    guard_high = min(settings.ceiling_hz, f_high * order)

    if guard_low >= guard_high - 50.0:
        return np.zeros_like(y)

    wn0 = np.clip(guard_low / nyquist, 0.01, 0.98)
    wn1 = np.clip(guard_high / nyquist, wn0 + 0.01, 0.995)

    harmonics_sos = butter(
        EXCITER_FILTER_ORDER,
        [wn0, wn1],
        btype="band",
        output="sos",
    )
    harmonics = sosfilt(harmonics_sos, saturated)
    rms_band = np.sqrt(np.mean(band**2))
    rms_harmonics = np.sqrt(np.mean(harmonics**2))
    if rms_harmonics <= EXCITER_SILENCE_LEVEL:
        return np.zeros_like(y)

    cap_db = _harmonic_cap_db(f_high)
    harmonics = harmonics * (
        rms_band * 10.0 ** (cap_db / 20.0) / (rms_harmonics + 1e-12)
    )
    return harmonics * _duck_envelope(band, sr)[None, :]


def _harmonic_cap_db(freq_hz: float) -> float:
    if freq_hz <= EXCITER_CAP_SPLIT_HZ:
        return EXCITER_HARMONIC_CAP_DB_LOW
    return EXCITER_HARMONIC_CAP_DB_HIGH


def _duck_envelope(band: np.ndarray, sr: int) -> np.ndarray:
    level = np.max(np.abs(band), axis=0)
    ac = float(np.exp(-1.0 / (sr * EXCITER_DUCK_ATTACK_MS / 1000.0)))
    rc = float(np.exp(-1.0 / (sr * EXCITER_DUCK_RELEASE_MS / 1000.0)))
    env = _decoupled_envelope(np.ascontiguousarray(level), ac, rc)
    peak = float(np.max(env))
    if peak <= EXCITER_SILENCE_LEVEL:
        return np.zeros_like(env)
    return np.clip(env / peak, 0.0, 1.0)


def _generate_harmonics(
    y: np.ndarray,
    sr: int,
    edges_hz: np.ndarray,
    settings: ExciterSettings,
) -> np.ndarray:
    harmonics = np.zeros_like(y)
    for band_hz in pairwise(edges_hz):
        harmonics += _excite_band(y, sr, band_hz, settings)
    return harmonics


def _compute_band_centers_hz(low_hz: float, ceiling_hz: float) -> np.ndarray:
    count = int(
        np.ceil(np.log2(ceiling_hz / low_hz) * EXCITER_BANDS_PER_OCTAVE)
    )
    return low_hz * 2.0 ** (np.arange(count + 1) / EXCITER_BANDS_PER_OCTAVE)


def _measure_band_powers_db(
    y: np.ndarray, sr: int, centers_hz: np.ndarray
) -> np.ndarray:
    mono = np.mean(y, axis=0)
    freqs, psd = welch(
        mono, fs=sr, nperseg=min(EXCITER_NPERSEG, mono.shape[-1])
    )
    half_band = 2.0 ** (0.5 / EXCITER_BANDS_PER_OCTAVE)
    powers = np.array(
        [
            np.sum(
                psd[
                    (freqs >= center / half_band) & (freqs < center * half_band)
                ]
            )
            for center in centers_hz
        ]
    )
    return 10.0 * np.log10(np.maximum(powers, EXCITER_POWER_FLOOR))


def _project_levels_db(
    dry_db: np.ndarray, centers_hz: np.ndarray
) -> np.ndarray:
    octaves_between = np.log2(centers_hz[None, :] / centers_hz[:, None])
    projections = (
        dry_db[:, None] - EXCITER_FILL_SLOPE_DB_PER_OCTAVE * octaves_between
    )
    is_lower_band = octaves_between > 0.0
    return np.max(np.where(is_lower_band, projections, -np.inf), axis=0)


def _compute_fill_gains(
    dry_db: np.ndarray,
    harmonics_db: np.ndarray,
    centers_hz: np.ndarray,
    mix: float,
) -> np.ndarray:
    target_db = np.maximum(_project_levels_db(dry_db, centers_hz), dry_db)
    deficit_power = 10.0 ** (target_db / 10.0) - 10.0 ** (dry_db / 10.0)
    gains = np.sqrt(
        mix
        * np.maximum(deficit_power, 0.0)
        / (10.0 ** (harmonics_db / 10.0) + 1e-12)
    )
    cap_db = np.array([_harmonic_cap_db(c) for c in centers_hz])
    budget = 10.0 ** ((dry_db + cap_db - harmonics_db) / 20.0)
    gains = np.minimum(gains, budget)
    return np.minimum(gains, 10.0 ** (EXCITER_MAX_FILL_GAIN_DB / 20.0))


def _build_fill_filter(
    centers_hz: np.ndarray, gains: np.ndarray, sr: int
) -> np.ndarray:
    grid_hz = np.linspace(0.0, sr / 2.0, EXCITER_NPERSEG // 2 + 1)
    curve = np.interp(
        np.log2(np.maximum(grid_hz, 1.0)),
        np.log2(centers_hz),
        gains,
        left=0.0,
        right=0.0,
    )
    return firwin2(
        EXCITER_FIR_TAPS, np.clip(grid_hz / (sr / 2.0), 0.0, 1.0), curve
    )


@passthrough_empty_audio
def apply_exciter(
    y: np.ndarray,
    sr: int,
    low_hz: float = EXCITER_LOW_HZ,
    high_hz: float = EXCITER_HIGH_HZ,
    drive: float = None,
    oversample: int = 4,
    mix: float = 0.25,
) -> np.ndarray:
    validate_sample_rate(sr)
    y = stereo(y, sr=sr)
    if mix <= 0.0 or y.shape[-1] < EXCITER_MIN_SAMPLES:
        return y

    ceiling_hz = min(high_hz, sr / 2.0 * EXCITER_NYQUIST_FRACTION)
    if not (0.0 < low_hz < ceiling_hz - 100.0):
        low_hz = max(1000.0, ceiling_hz * 0.1)

    edges_hz = np.geomspace(low_hz, ceiling_hz, EXCITER_BAND_COUNT + 1)
    settings = ExciterSettings(
        drive=drive, oversample=oversample, ceiling_hz=ceiling_hz
    )
    harmonics = _generate_harmonics(y, sr, edges_hz, settings)
    centers_hz = _compute_band_centers_hz(low_hz, ceiling_hz)
    gains = _compute_fill_gains(
        _measure_band_powers_db(y, sr, centers_hz),
        _measure_band_powers_db(harmonics, sr, centers_hz),
        centers_hz,
        mix,
    )
    gains = np.where(centers_hz >= edges_hz[1], gains, 0.0)
    air = signal.oaconvolve(
        harmonics,
        _build_fill_filter(centers_hz, gains, sr)[None, :],
        mode="same",
        axes=-1,
    )
    return y + air * mix


@passthrough_empty_audio
def apply_bass_enhancer(
    y: np.ndarray,
    sr: int,
    f_lo: float = 35.0,
    f_hi: float = 140.0,
    f_hp: float = 75.0,
    f_lp: float = 600.0,
    mix: float = 0.28,
    drive: float = 2.0,
) -> np.ndarray:
    """Tight psychoacoustic bass with zero intermodulation rumble."""
    validate_sample_rate(sr)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if mix <= 0.0 or y.shape[-1] < 256:
        return y

    ny = sr / 2.0
    wn_lo = np.clip(f_lo / ny, 0.001, 0.45)
    wn_hi = np.clip(f_hi / ny, wn_lo + 0.01, 0.48)
    bass = sosfiltfilt(
        butter(2, [wn_lo, wn_hi], btype="band", output="sos"), y, axis=-1
    )
    rms_b = float(np.sqrt(np.mean(bass**2)))
    if rms_b < 1e-9:
        return y

    u = np.mean(bass, axis=0) / (3.0 * rms_b)

    harm = np.tanh(drive * u) + 0.25 * (u * np.abs(u))
    harm = sosfiltfilt(
        butter(2, np.clip(f_hp / ny, 0.001, 0.45), btype="high", output="sos"),
        harm,
    )
    harm = sosfiltfilt(
        butter(2, np.clip(f_lp / ny, 0.01, 0.85), btype="low", output="sos"),
        harm,
    )

    rms_h = float(np.sqrt(np.mean(harm**2)))
    harm = harm * (mix * rms_b / (rms_h + 1e-12))
    return y + harm[None, :]


def stereo(*y_s, lr=True, sr=44100):
    validate_sample_rate(sr)
    for y in y_s:
        validate_audio_not_empty(y)
    stereo_result = None

    if lr and len(y_s) == 2:
        a, b = y_s
        a = stereo(a)
        b = stereo(b)
        _l, _r = pad_audio(a, b)
        y_s = [np.vstack([_l[0], _r[1]])]

    for y in y_s:
        was_mono = y.ndim == 1
        if was_mono:
            y = y[np.newaxis, :]
            y = np.vstack([y, y])
        if y.ndim > 2:
            y = np.vstack([y[0], y[1]])
        if not was_mono and y.shape[0] > 2 and y.shape[1] < y.shape[0]:
            y = y.T

        y = np.asarray(y, dtype=np.float64)

        if stereo_result is None:
            stereo_result = y
        else:
            region_s = max(stereo_result.shape[-1], y.shape[-1]) / sr
            stereo_result = mix_audio(
                stereo_result,
                y,
                sr,
                weight=0.5,
                fade_duration=min(2.0, region_s),
            )
    return stereo_result


def pad_audio(y1: np.ndarray, y2: np.ndarray):
    y1 = stereo(y1)
    y2 = stereo(y2)
    max_len = max(y1.shape[1], y2.shape[1])
    y1_padded = np.zeros((y1.shape[0], max_len), dtype=np.float64)
    y2_padded = np.zeros((y2.shape[0], max_len), dtype=np.float64)
    y1_padded[:, : y1.shape[1]] = y1
    y2_padded[:, : y2.shape[1]] = y2
    return y1_padded, y2_padded


def mix_audio(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    weight: float = 0.3,
    fade_duration: float = 2.0,
    duck_depth: float = 0.3,
    cutoff_hz: int | None = None,
):
    validate_sample_rate(sr)
    validate_audio_not_empty(y1)
    validate_audio_not_empty(y2)

    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    y1, y2 = stereo(y1), stereo(y2)
    y1_padded, y2_padded = pad_audio(y1, y2)

    y1_padded = y1_padded.T
    y2_padded = y2_padded.T

    if cutoff_hz is None:
        y1_filtered = y1_padded
    else:
        nyquist = 0.5 * sr
        normal_cutoff = cutoff_hz / nyquist
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        y1_filtered = lfilter(b, a, y1_padded, axis=0)

    window_size = int(sr * 0.1)
    envelope = np.abs(y2_padded.mean(axis=1))
    envelope = np.convolve(
        envelope, np.ones(window_size) / window_size, mode="same"
    )

    if np.max(envelope) > 0:
        envelope = envelope / np.max(envelope)
    envelope = envelope[:, np.newaxis]

    y1_processed = (y1_padded * (1 - envelope)) + (y1_filtered * envelope)
    duck_mask = 1.0 - (envelope * (1.0 - duck_depth))
    y_length = len(y1_padded)

    validate_fade_duration(fade_duration, y_length / sr)
    fade_samples = max(1, min(int(fade_duration * sr), y_length))
    t = np.linspace(0, np.pi / 2, fade_samples).reshape(-1, 1)
    fade_in, fade_out = np.sin(t), np.cos(t)

    mask1 = np.ones((y_length, 1)) * weight
    mask1[-fade_samples:] *= fade_out

    mask2 = np.zeros((y_length, 1))
    mask2[:fade_samples] = (1.0 - weight) * fade_in
    mask2[fade_samples:] = 1.0 - weight

    y1_weighted = y1_processed * mask1 * duck_mask
    y2_weighted = y2_padded * mask2
    y_mixed = y1_weighted + y2_weighted

    max_val = np.max(np.abs(y_mixed))
    if max_val > 1.0:
        y_mixed /= max_val

    if y_mixed.shape[0] > y_mixed.shape[1]:
        y_mixed = y_mixed.T

    return y_mixed


@passthrough_empty_audio
def freq_cut(y: np.ndarray, sr: int, low_cut: int | None, high_cut: int | None):
    validate_sample_rate(sr)
    y = np.array(y, dtype=np.float64)
    if y.size == 0:
        return y

    y -= np.mean(y, axis=-1, keepdims=True)
    nyquist = sr / 2.0
    min_len = 64

    if y.shape[-1] > min_len:
        if high_cut and 0 < high_cut < nyquist:
            sos = butter(4, high_cut / nyquist, btype="low", output="sos")
            y = sosfiltfilt(sos, y, axis=-1)

        if low_cut and 0 < low_cut < nyquist:
            sos = butter(4, low_cut / nyquist, btype="high", output="sos")
            y = sosfiltfilt(sos, y, axis=-1)

    return y


class SmartMastering:
    _profiling_data: dict[str, float] = {}

    def update_bands(self) -> None:
        count = self.multiband_params.num_bands
        freq_min = self.low_cut or 30.0
        freq_max = self.high_cut or min(
            self.resampling_target / 2 - 100.0, 18000.0
        )
        fcs = generate_bands(freq_min, freq_max, count)
        self.bands = self.multiband_params.make_bands(fcs, freq_min, freq_max)

    @property
    def slope_db(self) -> float:
        return self._slope_db

    @slope_db.setter
    def slope_db(self, value: float) -> None:
        self._slope_db = value
        self.update_bands()

    @classmethod
    def enable_profiling(cls, enabled: bool = True) -> None:
        cls._profiling_data.clear()
        cls._profiling = enabled

    @classmethod
    def get_profile(cls) -> dict[str, float]:
        return dict(cls._profiling_data)

    @classmethod
    def _profile(cls, name: str, duration: float) -> None:
        if not getattr(cls, "_profiling", False):
            return
        cls._profiling_data[name] = (
            cls._profiling_data.get(name, 0.0) + duration
        )

    def __init__(
        self,
        config: SmartMasteringConfig | None = None,
    ) -> None:
        cfg = config or SmartMasteringConfig()
        self.resampling_target = 44100
        self.sr = (
            validate_sample_rate(cfg.sample_rate)
            if cfg.sample_rate is not None
            else self.resampling_target
        )
        self._slope_db = (
            cfg.slope_db
            if cfg.slope_db is not None
            else SmartMasteringConfig().slope_db
        )
        self.slope_hz = cfg.slope_hz or SmartMasteringConfig().slope_hz
        self.smoothing_fraction = (
            cfg.smoothing_fraction or SmartMasteringConfig().smoothing_fraction
        )
        self.config = cfg

        self.mix_params = cfg.mix or SmartMasteringConfig().mix
        self.multiband_params = (
            cfg.multiband or SmartMasteringConfig().multiband
        )
        self.limiter_defaults = cfg.limiter or SmartMasteringConfig().limiter
        self.target_lufs = cfg.target_lufs or SmartMasteringConfig().target_lufs
        self.low_cut = cfg.low_cut or SmartMasteringConfig().low_cut
        self.high_cut = cfg.high_cut or SmartMasteringConfig().high_cut
        self.correction_strength = (
            cfg.correction_strength
            if cfg.correction_strength is not None
            else SmartMasteringConfig().correction_strength
        )
        self.phase_type = cfg.phase_type or SmartMasteringConfig().phase_type
        self.work_lufs = cfg.work_lufs
        self.max_correction_db = cfg.max_correction_db
        self.presence_db = cfg.presence_db
        self.presence_hz = cfg.presence_hz
        self.presence_octaves = cfg.presence_octaves
        self.exciter_mix = cfg.exciter_mix
        self.stereo_width = cfg.stereo_width
        self.bass_db, self.bass_hz, self.bass_octaves = (
            cfg.bass_db,
            cfg.bass_hz,
            cfg.bass_octaves,
        )
        self.warmth_db, self.warmth_hz, self.warmth_octaves = (
            cfg.warmth_db,
            cfg.warmth_hz,
            cfg.warmth_octaves,
        )
        self.air_db, self.air_hz, self.air_octaves = (
            cfg.air_db,
            cfg.air_hz,
            cfg.air_octaves,
        )
        self.bass_enhance_mix = cfg.bass_enhance_mix
        self.bass_mono_hz = cfg.bass_mono_hz
        self.expansion_db = cfg.expansion_db
        self.expansion_low_hz = cfg.expansion_low_hz
        self.expansion_high_hz = cfg.expansion_high_hz
        self.expansion_bands = cfg.expansion_bands
        self.expansion_window_ms = cfg.expansion_window_ms
        self.expansion_attack_ms = cfg.expansion_attack_ms
        self.expansion_release_ms = cfg.expansion_release_ms
        self.expansion_ratio = cfg.expansion_ratio
        self.pre_limiter_rise_lu = cfg.pre_limiter_rise_lu

        self.update_bands()
        self.nperseg = int(2 ** np.ceil(np.log2(self.sr / 5)))

    def _measure_spectrum(
        self, y_mono: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        NPERSEG = self.nperseg
        FLOOR_DB = -120.0
        CEILING_DB = 20.0

        if len(y_mono) < NPERSEG:
            y_mono_padded = np.pad(y_mono, (0, NPERSEG - len(y_mono)))
        else:
            y_mono_padded = y_mono

        f_axis, psd = signal.welch(
            y_mono_padded,
            fs=self.sr,
            nperseg=NPERSEG,
            noverlap=int(NPERSEG * 0.75),
            window=("kaiser", 18),
            scaling="density",
            average="median",
            detrend="constant",
        )
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
        psd_db = np.clip(psd_db, FLOOR_DB, CEILING_DB)
        return psd_db, f_axis

    def _build_target_spectrum(self, f_axis: np.ndarray) -> np.ndarray:
        min_freq = self.low_cut if self.low_cut else 25.0
        max_freq = self.high_cut if self.high_cut else f_axis[-1]
        f_axis_safe = np.clip(f_axis, min_freq, max_freq)
        octaves = np.log2(f_axis_safe / self.slope_hz)
        return -self.slope_db * octaves

    def _smooth_target_curve(
        self,
        target_db: np.ndarray,
        f_axis: np.ndarray,
        smoothing_fraction: float | None = None,
    ) -> np.ndarray:
        if smoothing_fraction is None:
            smoothing_fraction = self.smoothing_fraction

        target_db = np.asarray(target_db, dtype=float)
        if not smoothing_fraction or smoothing_fraction <= 0 or len(f_axis) < 8:
            return np.copy(target_db)

        f = np.maximum(np.asarray(f_axis, dtype=float), 1.0)
        log_f = np.log2(f)
        pts_per_oct = 48
        grid = np.arange(log_f[1], log_f[-1], 1.0 / pts_per_oct)
        if len(grid) < 4:
            return np.copy(target_db)

        on_grid = np.interp(grid, log_f[1:], target_db[1:])
        sigma = max(smoothing_fraction * pts_per_oct / 2.355, 0.5)
        smoothed_grid = gaussian_filter1d(on_grid, sigma, mode="nearest")
        smoothed = np.interp(log_f, grid, smoothed_grid)
        smoothed[0] = smoothed[1]
        return smoothed

    def _tone_curve(self, f_axis: np.ndarray) -> np.ndarray:
        f = np.maximum(np.asarray(f_axis, dtype=float), 1.0)
        total = np.zeros_like(f)
        for db, hz, octaves in (
            (self.bass_db, self.bass_hz, self.bass_octaves),
            (self.warmth_db, self.warmth_hz, self.warmth_octaves),
            (self.presence_db, self.presence_hz, self.presence_octaves),
        ):
            if db:
                sigma = max(octaves, 0.1) / 2.355
                total += db * np.exp(-0.5 * (np.log2(f / hz) / sigma) ** 2)
        if self.air_db:
            width = max(self.air_octaves, 0.2) / 4.0
            total += self.air_db / (
                1.0 + np.exp(-np.log2(f / self.air_hz) / width)
            )
        return total

    def _phase(self, y: np.ndarray, tp: str) -> np.ndarray:
        if y.ndim == 2 and y.shape[1] < y.shape[0]:
            return self._phase(y.T, tp).T

        orig_len = y.shape[-1]
        y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

        input_db, f_axis = self._measure_spectrum(y_mono)
        target_db = self._build_target_spectrum(f_axis)

        if not np.all(np.isfinite(target_db)):
            target_db = np.nan_to_num(
                target_db, nan=0.0, posinf=0.0, neginf=0.0
            )

        input_db = self._smooth_target_curve(
            input_db, f_axis, self.smoothing_fraction
        )

        min_len = min(len(target_db), len(input_db))
        f_axis = f_axis[:min_len]
        target_db = target_db[:min_len]
        input_db = input_db[:min_len]

        lo = float(self.low_cut or 28.0)
        hi = float(self.high_cut or f_axis[-1])
        band = (f_axis >= lo) & (f_axis <= hi)

        if np.any(band):
            w = 1.0 / np.maximum(f_axis[band], 1.0)
            mean_target = np.average(target_db[band], weights=w)
            mean_input = np.average(input_db[band], weights=w)
        else:
            mean_target = mean_input = 0.0

        target_db = target_db - mean_target + mean_input
        correction_db = (target_db - input_db) * self.correction_strength
        correction_db[~band] = 0.0

        lim = max(float(self.max_correction_db), 1e-3)
        correction_db = lim * np.tanh(correction_db / lim)
        correction_db = correction_db + self._tone_curve(f_axis)
        correction_db = self._smooth_target_curve(
            correction_db, f_axis, 1.0 / 6.0
        )

        audible_band = (f_axis >= 80.0) & (f_axis <= 8000.0)
        center_gain = (
            np.median(correction_db[audible_band])
            if np.any(audible_band)
            else 0.0
        )
        correction_db_norm = correction_db - center_gain
        H_lin = 10.0 ** (correction_db_norm / 20.0)

        if tp == "minimal":
            log_mag = np.log(np.maximum(H_lin, 1e-12))
            cepstrum = np.fft.irfft(log_mag)
            n = len(cepstrum)
            lifter = np.zeros(n)
            lifter[0] = 1.0
            lifter[1 : n // 2] = 2.0
            if n % 2 == 0:
                lifter[n // 2] = 1.0

            causal_cepstrum = cepstrum * lifter
            min_phase_spectrum = np.exp(np.fft.rfft(causal_cepstrum))
            h_ir = np.fft.irfft(min_phase_spectrum)

            FIR_LEN = min(len(h_ir), 4096)
            h_ir_cut = h_ir[:FIR_LEN]
            window = np.ones(FIR_LEN)
            fade_size = FIR_LEN // 2
            if fade_size > 0:
                window[-fade_size:] = np.hamming(fade_size * 2)[fade_size:]

            h_ir_final = (h_ir_cut * window).reshape(
                (1,) * (y.ndim - 1) + (-1,)
            )
            out = signal.oaconvolve(y, h_ir_final, mode="full")

        elif tp == "linear":
            H_full = np.concatenate([H_lin, H_lin[-2:0:-1]])
            h = np.real(np.fft.ifft(H_full))
            h_centered = np.fft.fftshift(h)
            FIR_LEN = min(len(h), 4096)
            center = len(h_centered) // 2
            half_len = FIR_LEN // 2
            h_final = h_centered[center - half_len : center + half_len]
            h_final *= signal.windows.hamming(len(h_final))
            h_final = h_final.reshape((1,) * (y.ndim - 1) + (-1,))
            out = signal.oaconvolve(y, h_final, mode="same")
        else:
            raise InvalidAudioInputError(f"Unknown phase type: {tp}")

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = (
            out[..., :orig_len]
            if out.shape[-1] > orig_len
            else np.pad(
                out,
                (*((0, 0),) * (out.ndim - 1), (0, orig_len - out.shape[-1])),
            )
        )
        return out

    def _compress_band(
        self, band: np.ndarray, cfg: MultibandBand, f_lo: float
    ) -> np.ndarray:
        """Dual-action dynamics: downward control of loud peaks + upward lift of buried details."""
        sr = self.sr
        win = int(np.clip(sr * 1.5 / max(f_lo, 30.0), sr * 0.003, sr * 0.025))
        power = np.max(
            uniform_filter1d(band**2, size=win, axis=-1, mode="nearest"),
            axis=0,
        )
        det_db = 10.0 * np.log10(power + 1e-14)

        active = det_db > (det_db.max() - 60.0)
        if np.count_nonzero(active) < 16 or det_db.max() < -90.0:
            return band

        if self.multiband_params.relative_threshold:
            ref_db = 10.0 * np.log10(np.mean(power[active]) + 1e-14)
            thr = ref_db + cfg.base_threshold
        else:
            thr = cfg.base_threshold + cfg.makeup_db

        over = det_db - thr
        knee = max(cfg.knee_db, 1e-6)
        slope = 1.0 - 1.0 / max(cfg.ratio, 1.0)

        gr = np.zeros_like(over)
        hi = over > knee / 2.0
        kn = (over > -knee / 2.0) & ~hi
        gr[hi] = over[hi] * slope
        gr[kn] = slope * (over[kn] + knee / 2.0) ** 2 / (2.0 * knee)

        ac = np.exp(-1.0 / (sr * cfg.attack_ms / 1000.0 + 1e-9))
        rc = np.exp(-1.0 / (sr * cfg.release_ms / 1000.0 + 1e-9))
        gr = _decoupled_envelope(np.ascontiguousarray(gr), ac, rc)
        downward_gain = 10.0 ** (-gr / 20.0)

        upward_gain = np.ones_like(det_db)
        if cfg.upward_boost_db > 0.0:
            noise_floor_db = -58.0
            l_thr = cfg.upward_low_thr_db
            h_thr = cfg.upward_high_thr_db
            max_boost = cfg.upward_boost_db

            up_boost_db = np.zeros_like(det_db)

            in_zone = (det_db >= l_thr) & (det_db < h_thr)
            if np.any(in_zone):
                x = (det_db[in_zone] - l_thr) / max(h_thr - l_thr, 1e-3)
                up_boost_db[in_zone] = (
                    max_boost * 0.5 * (1.0 + np.cos(np.pi * x))
                )

            ramp_zone = (det_db > noise_floor_db) & (det_db < l_thr)
            if np.any(ramp_zone):
                x_r = (det_db[ramp_zone] - noise_floor_db) / max(
                    l_thr - noise_floor_db, 1e-3
                )
                up_boost_db[ramp_zone] = (
                    max_boost * 0.5 * (1.0 - np.cos(np.pi * x_r))
                )

            up_env_db = _decoupled_envelope(
                np.ascontiguousarray(up_boost_db), ac, rc
            )
            upward_gain = 10.0 ** (up_env_db / 20.0)

        total_gain = downward_gain * upward_gain

        if self.multiband_params.auto_makeup:
            e_in = float(np.sum(band**2))
            e_out = float(np.sum((band * total_gain) ** 2))
            if e_out > 1e-20:
                total_gain = total_gain * min(np.sqrt(e_in / e_out), 1.41)

        total_gain = total_gain * (10.0 ** (cfg.makeup_db / 20.0))
        return band * total_gain

    def _multiband_compress(self, y: np.ndarray) -> np.ndarray:
        was_1d = y.ndim == 1
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))

        lo = float(self.low_cut or 28.0)
        hi = float(self.high_cut or self.sr / 2.0 * 0.98)
        sorted_bands = sorted(self.bands, key=lambda b: b.fc)
        kept = [b for b in sorted_bands if lo * 1.05 < b.fc < hi * 0.95]

        if y2.shape[-1] < 512 or not kept:
            return y

        fcs = [b.fc for b in kept]
        cfgs = kept + [sorted_bands[-1]]

        out = np.zeros_like(y2)
        current = y2
        for i, cfg in enumerate(cfgs):
            if i < len(fcs):
                wn = float(np.clip(fcs[i] / (self.sr / 2.0), 1e-4, 0.985))
                sos = signal.butter(2, wn, btype="low", output="sos")
                band = signal.sosfiltfilt(sos, current, axis=-1)
                current = current - band
            else:
                band = current
            f_lo = fcs[i - 1] if i > 0 else lo
            out += self._compress_band(band, cfg, f_lo)

        return out[0] if was_1d else out

    def _upward_expand(self, y: np.ndarray) -> np.ndarray:
        """Per-band upward expansion keyed on level below each band's 3 s mean."""
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
        if (
            self.expansion_db <= 0.0
            or self.expansion_bands < 1
            or y2.shape[-1] < EXPANSION_MIN_SAMPLES
        ):
            return y
        sr = self.sr
        nyq = sr / 2.0
        lo = float(np.clip(self.expansion_low_hz, 20.0, nyq * 0.9))
        hi = float(np.clip(self.expansion_high_hz, lo * 1.1, nyq * 0.95))
        edges = np.geomspace(lo, hi, self.expansion_bands + 1)
        max_boost = float(min(self.expansion_db, 6.0))
        ratio = float(np.clip(self.expansion_ratio, 0.0, 1.0))
        det_win = max(1, int(sr * 0.01))
        ref_win = max(det_win + 1, int(sr * self.expansion_window_ms / 1000.0))
        ac = float(np.exp(-1.0 / (sr * self.expansion_attack_ms / 1000.0)))
        rc = float(np.exp(-1.0 / (sr * self.expansion_release_ms / 1000.0)))

        out = np.array(y2, copy=True)
        for f_lo, f_hi in pairwise(edges):
            sos = signal.butter(
                2, [f_lo / nyq, f_hi / nyq], btype="band", output="sos"
            )
            band = signal.sosfiltfilt(sos, y2, axis=-1)
            power = np.max(
                uniform_filter1d(
                    band**2, size=det_win, axis=-1, mode="nearest"
                ),
                axis=0,
            )
            det_db = 10.0 * np.log10(power + 1e-14)
            if det_db.max() < -90.0:
                continue
            ref_db = 10.0 * np.log10(
                uniform_filter1d(power, size=ref_win, mode="nearest") + 1e-14
            )
            below_db = ref_db - det_db
            boost_db = np.clip(below_db * ratio, 0.0, max_boost)
            boost_db[det_db < -60.0] = 0.0
            boost_db = _decoupled_envelope(
                np.ascontiguousarray(boost_db), ac, rc
            )
            gain = 10.0 ** (boost_db / 20.0) - 1.0
            out += band * gain[None, :]
        return out[0] if np.ndim(y) == 1 else out

    def _stereo_widen(
        self, y: np.ndarray, width: float | None = None
    ) -> np.ndarray:
        """3D soundstage with mono bass anchor + side-channel low-level lift."""
        width = self.stereo_width if width is None else width
        mono_hz = self.bass_mono_hz
        if y.ndim < 2 or y.shape[0] != 2:
            return y

        mid = (y[0] + y[1]) * 0.5
        side = (y[0] - y[1]) * 0.5

        if mono_hz:
            sos = signal.butter(
                2, mono_hz / (self.sr / 2.0), btype="high", output="sos"
            )
            side = signal.sosfiltfilt(sos, side)

        if abs(width - 1.0) >= 1e-3:
            sos_w = signal.butter(
                2, 350.0 / (self.sr / 2.0), btype="high", output="sos"
            )
            side_presence = signal.sosfiltfilt(sos_w, side)
            side = side + (width - 1.0) * side_presence

        return np.stack([mid + side, mid - side])

    def _limiter_gain_db(
        self,
        peak: np.ndarray,
        limit_lin: float,
        lookahead_samp: int,
        sr: float,
        release_fast_ms: float,
        release_slow_ms: float,
        slow_weight: float,
    ) -> np.ndarray:
        req_db = 20.0 * np.log10(
            np.minimum(1.0, limit_lin / np.maximum(peak, 1e-12))
        )
        L = max(1, int(lookahead_samp))
        gmin = minimum_filter1d(
            req_db, size=L, origin=-(L // 2), mode="constant", cval=0.0
        )
        gs = uniform_filter1d(
            gmin, size=L, origin=(L - 1) // 2, mode="constant", cval=0.0
        )
        gs = np.minimum(gs, req_db)

        rc_fast = float(np.exp(-1.0 / (sr * release_fast_ms / 1000.0)))
        rc_slow = float(np.exp(-1.0 / (sr * release_slow_ms / 1000.0)))
        return _limiter_release(
            np.ascontiguousarray(gs), rc_fast, rc_slow, float(slow_weight)
        )

    def _limiter_prepare(
        self, y: np.ndarray, p: LimiterParams
    ) -> list[np.ndarray]:
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
        kernel = _build_oversampling_kernel(self.sr, p.os_factor, p.kaiser_beta)

        def up(a):
            rows = [
                signal.resample_poly(row, p.os_factor, 1, window=kernel).astype(
                    np.float32
                )
                for row in a
            ]
            return np.ascontiguousarray(np.stack(rows))

        if not p.split_hz or y2.shape[-1] < 512:
            return [up(y2)]

        wn = float(np.clip(p.split_hz / (self.sr / 2.0), 1e-4, 0.45))
        sos = signal.butter(2, wn, btype="low", output="sos")
        lo = signal.sosfiltfilt(sos, y2, axis=-1)
        return [up(lo), up(y2 - lo)]

    def _limit_band_inplace(
        self,
        x: np.ndarray,
        ceiling_lin: float,
        lookahead_ms: float,
        release_fast_ms: float,
        release_slow_ms: float,
        slow_weight: float,
        os_factor: int,
    ) -> np.ndarray:
        n_ch, n_os = x.shape
        n = n_os // os_factor
        view = x[:, : n * os_factor].reshape(n_ch, n, os_factor)
        peak = np.maximum(view.max(axis=2), -view.min(axis=2)).max(axis=0)
        peak = peak.astype(np.float64)

        lookahead = max(1, int(round(lookahead_ms * self.sr / 1000.0)))
        gain_db = self._limiter_gain_db(
            peak,
            ceiling_lin,
            lookahead,
            self.sr,
            release_fast_ms,
            release_slow_ms,
            slow_weight,
        )
        view *= (10.0 ** (gain_db / 20.0)).astype(x.dtype)[None, :, None]
        return gain_db

    @staticmethod
    def _soft_clip_inplace(x: np.ndarray, knee: float, limit: float) -> None:
        span = float(limit) - float(knee)
        block = 1 << 20
        for s0 in range(0, x.shape[-1], block):
            xb = x[:, s0 : s0 + block]
            a = np.abs(xb)
            over = a > knee
            if np.any(over):
                xb[over] = np.sign(xb[over]) * (
                    knee + span * np.tanh((a[over] - knee) / span)
                )

    def _limit_oversampled(
        self,
        bands_os: list[np.ndarray],
        drive_db: float,
        ceil_db: float,
        p: LimiterParams,
    ) -> np.ndarray:
        os_factor = p.os_factor
        limit_lin = 10.0 ** ((ceil_db - p.margin_db) / 20.0)
        lim1 = limit_lin * 10.0 ** (p.clip_ceiling_db / 20.0)
        drive = 10.0 ** (drive_db / 20.0)

        if len(bands_os) == 2:
            ceil_lo = lim1 * p.band_ceiling_lo
            lo = bands_os[0] * drive
            self._limit_band_inplace(
                lo,
                ceil_lo * 10.0 ** (p.bass_clip_db / 20.0),
                p.lo_lookahead_ms,
                p.lo_release_fast_ms,
                p.lo_release_slow_ms,
                p.lo_slow_weight,
                os_factor,
            )
            self._soft_clip_inplace(lo, p.bass_clip_knee * ceil_lo, ceil_lo)
            x = bands_os[1] * drive
            x += lo
            del lo
        else:
            x = bands_os[0] * drive

        self._limit_band_inplace(
            x,
            lim1,
            p.lookahead_ms,
            p.release_fast_ms,
            p.release_slow_ms,
            p.slow_weight,
            os_factor,
        )

        knee = float(p.clip_knee * limit_lin)
        span = float(limit_lin) - knee
        block = 1 << 20
        for s0 in range(0, x.shape[-1], block):
            xb = x[:, s0 : s0 + block]
            a = np.abs(xb)
            over = a > knee
            if np.any(over):
                xb[over] = np.sign(xb[over]) * (
                    knee + span * np.tanh((a[over] - knee) / span)
                )

        out = signal.resample_poly(
            x,
            1,
            os_factor,
            axis=-1,
            window=_build_oversampling_kernel(
                self.sr, os_factor, p.kaiser_beta
            ),
        )
        lim_full = 10.0 ** (ceil_db / 20.0)
        return np.clip(out.astype(np.float64), -lim_full, lim_full)

    def _apply_limiter(
        self,
        y: np.ndarray,
        drive_db: float,
        ceil_db: float,
        **overrides,
    ) -> np.ndarray:
        p = (
            replace(self.limiter_defaults, **overrides)
            if overrides
            else self.limiter_defaults
        )
        was_1d = np.ndim(y) == 1
        orig_len = np.shape(y)[-1]
        x_os = self._limiter_prepare(y, p)
        out = self._limit_oversampled(x_os, drive_db, ceil_db, p)
        out = out[..., :orig_len]
        return out[0] if was_1d else out

    def _limit_to_target(
        self,
        y: np.ndarray,
        target_lufs: float | None,
        p: LimiterParams | None = None,
    ) -> np.ndarray:
        p = p or self.limiter_defaults
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
        orig_len = y2.shape[-1]
        x_os = self._limiter_prepare(y2, p)

        current = self._measure_lufs(y2)
        if target_lufs is None or current is None:
            return self._limit_oversampled(x_os, 0.0, p.ceil_db, p)[
                ..., :orig_len
            ]

        norm = 10.0 ** ((target_lufs - current) / 20.0)
        for band_os in x_os:
            band_os *= norm
        drive = 2.5
        history: list[tuple[float, float]] = []
        best = None
        best_err = np.inf

        for _it in range(max(1, p.max_iterations)):
            out = self._limit_oversampled(x_os, drive, p.ceil_db, p)[
                ..., :orig_len
            ]
            measured = self._measure_lufs(out)
            if measured is None:
                best = out
                break
            err = target_lufs - measured

            if abs(err) < best_err:
                best_err, best = abs(err), out

            if abs(err) <= p.loudness_tol_lu:
                break

            history.append((drive, measured))
            if (
                len(history) >= 2
                and abs(history[-1][0] - history[-2][0]) > 1e-3
            ):
                s = (history[-1][1] - history[-2][1]) / (
                    history[-1][0] - history[-2][0]
                )
                if s < p.min_efficiency_lu_per_db:
                    break
                s = float(np.clip(s, 0.25, 1.2))
            else:
                s = 0.6

            new_drive = float(np.clip(drive + err / s, 0.0, p.max_drive_db))
            if abs(new_drive - drive) < 0.02:
                break
            drive = new_drive

        return best

    @staticmethod
    def _k_weighting_coeffs(sr: float):
        f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
        k = np.tan(np.pi * f0 / sr)
        vh = 10.0 ** (g / 20.0)
        vb = vh**0.4996667741545416
        a0 = 1.0 + k / q + k * k
        b1 = np.array(
            [
                (vh + vb * k / q + k * k) / a0,
                2.0 * (k * k - vh) / a0,
                (vh - vb * k / q + k * k) / a0,
            ]
        )
        a1 = np.array(
            [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
        )

        f0, q = 38.13547087602444, 0.5003270373238773
        k = np.tan(np.pi * f0 / sr)
        a0 = 1.0 + k / q + k * k
        b2 = np.array([1.0, -2.0, 1.0])
        a2 = np.array(
            [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
        )
        return b1, a1, b2, a2

    def _measure_lufs(self, y: np.ndarray) -> float | None:
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
        b1, a1, b2, a2 = self._k_weighting_coeffs(self.sr)
        y_filt = signal.lfilter(b1, a1, y2, axis=-1)
        y_filt = signal.lfilter(b2, a2, y_filt, axis=-1)
        y_sq = np.sum(y_filt**2, axis=0)

        n_samples = y_sq.shape[0]
        win_size = int(self.sr * 0.4)
        hop_size = max(1, int(self.sr * 0.1))

        if n_samples == 0:
            return None
        if n_samples < win_size:
            ms = np.array([np.mean(y_sq)])
        else:
            cs = np.concatenate([[0.0], np.cumsum(y_sq)])
            starts = np.arange(0, n_samples - win_size + 1, hop_size)
            ms = (cs[starts + win_size] - cs[starts]) / win_size

        abs_threshold = 10.0 ** ((-70.0 + 0.691) / 10.0)
        ms_gated = ms[ms > abs_threshold]
        if len(ms_gated) == 0:
            return None

        gamma_rel = 0.1 * np.mean(ms_gated)
        ms_final = ms_gated[ms_gated > gamma_rel]
        loudness_ms = np.mean(ms_final) if len(ms_final) else np.mean(ms_gated)
        return float(-0.691 + 10.0 * np.log10(max(loudness_ms, 1e-12)))

    def _apply_lufs(self, y: np.ndarray, target_lufs: float) -> np.ndarray:
        current = self._measure_lufs(y)
        if current is None:
            return y
        gain_lin = 10.0 ** ((target_lufs - current) / 20.0)
        return y * gain_lin

    @staticmethod
    def _validate_input(y, sr) -> None:
        if y is None:
            raise InvalidAudioInputError(EMPTY_AUDIO_MESSAGE)
        if sr is not None:
            validate_sample_rate(sr)
        if not isinstance(y, np.ndarray) or y.ndim == 0:
            raise InvalidAudioInputError(
                "Audio must be a 1D or 2D numpy array."
            )
        if y.size == 0:
            return
        if y.ndim > 2:
            raise InvalidAudioInputError(
                f"Audio must be 1D or 2D, got {y.ndim} dimensions."
            )
        if not (
            np.issubdtype(y.dtype, np.floating)
            or np.issubdtype(y.dtype, np.integer)
        ):
            raise InvalidAudioInputError(
                f"Audio must be numeric, got dtype {y.dtype}."
            )
        if np.issubdtype(y.dtype, np.floating) and not np.all(np.isfinite(y)):
            raise InvalidAudioInputError(
                "Audio contains NaN or infinite values."
            )

    def process(self, y: np.ndarray, sr: int = None) -> tuple[np.ndarray, int]:
        self._validate_input(y, sr)
        if sr is not None:
            self.sr = validate_sample_rate(sr)
        if is_empty_audio(y):
            return np.array(y, copy=True), self.sr

        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float64) / (float(np.iinfo(y.dtype).max) + 1.0)

        if y.ndim == 2 and y.shape[0] > y.shape[1]:
            y = y.T

        orig_len = y.shape[-1]
        orig_sr = self.sr

        y_resamp = y
        if self.sr != self.resampling_target:
            y_resamp = resample(y_resamp, self.sr, self.resampling_target)
            self.sr = self.resampling_target

        expected_len = int(round(orig_len * self.sr / orig_sr))
        y_stereo = stereo(y_resamp)

        y_clean = freq_cut(
            y_stereo, self.sr, low_cut=self.low_cut, high_cut=None
        )

        y_work = self._apply_lufs(y_clean, self.work_lufs)

        y_eq = self._phase(y_work, self.phase_type)

        y_mbc = self._multiband_compress(y_eq)

        if self.bass_enhance_mix > 0.0:
            y_mbc = apply_bass_enhancer(
                y_mbc, self.sr, mix=self.bass_enhance_mix
            )

        if self.exciter_mix > 0.0:
            y_air = apply_exciter(
                y_mbc,
                self.sr,
                low_hz=EXCITER_LOW_HZ,
                high_hz=self.high_cut or EXCITER_HIGH_HZ,
                mix=self.exciter_mix,
            )
        else:
            y_air = y_mbc

        y_exp = self._upward_expand(y_air)

        y_wide = self._stereo_widen(stereo(y_exp))

        y_band = freq_cut(
            y_wide, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        pre_lufs = self._measure_lufs(y_band)
        rise_cap = self.work_lufs + self.pre_limiter_rise_lu
        if pre_lufs is not None and pre_lufs > rise_cap:
            logger.info(
                "Pre-limiter loudness %.2f LUFS exceeds %.2f LUFS; renormalizing",
                pre_lufs,
                rise_cap,
            )
            y_band = y_band * 10.0 ** ((rise_cap - pre_lufs) / 20.0)

        lp = self.limiter_defaults
        if lp.drive_db is None:
            y_lim = self._limit_to_target(y_band, self.target_lufs, lp)
        else:
            y_lim = self._apply_limiter(y_band, lp.drive_db, lp.ceil_db)

        lin_amp = 10.0 ** ((lp.ceil_db - TRUE_PEAK_GUARD_DB) / 20.0)
        y_clip = np.clip(y_lim, -lin_amp, lin_amp)

        if y_clip.shape[-1] != expected_len:
            if y_clip.shape[-1] > expected_len:
                y_clip = y_clip[..., :expected_len]
            else:
                y_clip = np.pad(
                    y_clip,
                    (
                        *((0, 0),) * (y_clip.ndim - 1),
                        (0, expected_len - y_clip.shape[-1]),
                    ),
                )

        return y_clip, self.sr


def validation_report(master: SmartMastering, y: np.ndarray, sr: int):
    validate_sample_rate(sr)
    y = np.asarray(y, dtype=np.float64)
    input_mono = np.mean(y, axis=0) if y.ndim > 1 else y
    f_axis, psd_in = signal.welch(
        input_mono,
        fs=sr,
        nperseg=master.nperseg,
        noverlap=master.nperseg // 2,
        window="hann",
        scaling="density",
        average="median",
    )
    y_out, _ = master.process(y, sr=sr)
    output_mono = np.mean(y_out, axis=0) if y_out.ndim > 1 else y_out
    _, psd_out = signal.welch(
        output_mono,
        fs=sr,
        nperseg=master.nperseg,
        noverlap=master.nperseg // 2,
        window="hann",
        scaling="density",
        average="median",
    )
    in_db = 10.0 * np.log10(np.maximum(psd_in, 1e-24))
    out_db = 10.0 * np.log10(np.maximum(psd_out, 1e-24))
    return f_axis, in_db, out_db


def benchmark(
    master: SmartMastering, duration_sec: float = 5.0, sr: int = 44100
) -> dict:
    import numpy as _np

    master.enable_profiling(True)
    n_samples = int(duration_sec * sr)
    test_audio = _np.random.randn(2, n_samples).astype(_np.float64)
    _ = master.process(test_audio, sr=sr)
    return master.get_profile()
