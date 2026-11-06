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

EXCITER_LOW_HZ = 5.0
EXCITER_HIGH_HZ = 22000.0
EXCITER_BAND_COUNT = 8
EXCITER_FILTER_ORDER = 4
EXCITER_MAX_HARMONIC_ORDER = 5
EXCITER_PEAK_DRIVE = 1.0
EXCITER_NYQUIST_FRACTION = 0.99
EXCITER_SILENCE_LEVEL = 1e-9
EXCITER_KAISER_BETA = 12.0
EXCITER_GUARD_RATIO = 2.0 ** (1.0 / 3.0)
EXCITER_NPERSEG = 16384
EXCITER_FIR_TAPS = 8193
EXCITER_MIN_SAMPLES = 1024
EXCITER_BANDS_PER_OCTAVE = 6
EXCITER_FILL_SLOPE_DB_PER_OCTAVE = 3.0
EXCITER_MAX_FILL_GAIN_DB = 60.0
EXCITER_POWER_FLOOR = 1e-24
OVERSAMPLING_TAPS_PER_FACTOR = 40
TRUE_PEAK_GUARD_DB = 0.1


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


@dataclass
class MultibandParams:
    """Multiband compressor settings.

    Thresholds are *relative*: `bass_threshold_db` / `treb_threshold_db` are
    offsets (dB) from each band's own long-term RMS level, so the amount of
    compression is the same whatever the input level or spectral tilt is
    (an absolute dBFS threshold compresses the loud bass bands constantly and
    never touches the quiet treble bands, which is what causes pumping).
    Set `relative_threshold=False` to use absolute dBFS thresholds.
    """

    num_bands: int = 32
    intensity: float = 1.0

    bass_ratio: float = 3.5
    bass_attack_ms: float = 30.0
    bass_release_ms: float = 180.0
    bass_threshold_db: float = 32.0

    treb_ratio: float = 2.5
    treb_attack_ms: float = 8.0
    treb_release_ms: float = 80.0
    treb_threshold_db: float = -18.0

    knee_db: float = 8.0
    relative_threshold: bool = True
    auto_makeup: bool = True

    def make_bands(
        self, fcs: list[float], freq_min: float, freq_max: float
    ) -> list[MultibandBand]:
        if not len(fcs):
            return []

        fcs_arr = np.array(fcs, dtype=float)

        ref_min = np.log2(freq_min)
        ref_max = np.log2(freq_max)
        span = max(ref_max - ref_min, 1e-9)

        fcs_safe = np.clip(fcs_arr, freq_min, freq_max)
        positions = (np.log2(fcs_safe) - ref_min) / span

        base_thr = (
            self.bass_threshold_db
            + (self.treb_threshold_db - self.bass_threshold_db) * positions
        )
        ratios = self.bass_ratio + (self.treb_ratio - self.bass_ratio) * positions
        attacks = (
            self.bass_attack_ms
            + (self.treb_attack_ms - self.bass_attack_ms) * positions
        )
        releases = (
            self.bass_release_ms
            + (self.treb_release_ms - self.bass_release_ms) * positions
        )

        thr = np.mean(base_thr) + (base_thr - np.mean(base_thr)) * self.intensity

        ratios = 1.0 + (ratios - 1.0) * self.intensity
        attacks = attacks * (1.0 - 0.5 * self.intensity)
        releases = releases * (1.0 + 0.5 * self.intensity)

        makeups = np.zeros_like(fcs_arr, dtype=float)
        knees = np.full_like(fcs_arr, self.knee_db, dtype=float)

        return [
            MultibandBand(
                fc=float(fcs_arr[i]),
                base_threshold=float(thr[i]),
                ratio=float(ratios[i]),
                attack_ms=float(max(attacks[i], 0.1)),
                release_ms=float(max(releases[i], 1.0)),
                makeup_db=float(makeups[i]),
                knee_db=float(knees[i]),
            )
            for i in range(len(fcs_arr))
        ]
    

@dataclass(frozen=True)
class ExciterSettings:
    drive: float | None
    oversample: int
    ceiling_hz: float


@dataclass
class LimiterParams:
    drive_db: float | None = None
    ceil_db: float = -0.3
    max_drive_db: float = 0.0

    lookahead_ms: float = 3.0
    release_fast_ms: float = 20.0
    release_slow_ms: float = 280.0
    slow_weight: float = 0.25

    clip_ceiling_db: float = 0.3
    clip_knee: float = 12.0

    os_factor: int = 4
    kaiser_beta: float = 12.0
    margin_db: float = 0.5

    split_hz: float | None = 400.0
    band_ceiling_lo: float = 0.85
    bass_clip_db: float = 2.5
    bass_clip_knee: float = 0.5
    lo_lookahead_ms: float = 6.0
    lo_release_fast_ms: float = 60.0
    lo_release_slow_ms: float = 400.0
    lo_slow_weight: float = 0.6

    loudness_tol_lu: float = 0.1
    min_efficiency_lu_per_db: float = 0.1
    max_iterations: int = 6


@dataclass
class SmartMasteringConfig:
    sample_rate: int | None = None
    slope_db: float = 3.0
    slope_hz: float = 1000.0
    smoothing_fraction: float = 0.0
    mix: MixParams = field(default_factory=MixParams)
    multiband: MultibandParams = field(default_factory=MultibandParams)
    limiter: LimiterParams = field(default_factory=LimiterParams)
    target_lufs: float = -9.0
    work_lufs: float = -18.0
    correction_strength: float = 1.0
    max_correction_db: float = 60.0

    bass_db: float = 4.0
    bass_hz: float = 65.0
    bass_octaves: float = 1.7
    warmth_db: float = 1.8
    warmth_hz: float = 190.0
    warmth_octaves: float = 1.4
    presence_db: float = 5.0
    presence_hz: float = 3200.0
    presence_octaves: float = 1.6
    air_db: float = 2.5
    air_hz: float = 11000.0
    air_octaves: float = 1.6

    bass_enhance_mix: float = 0.3
    bass_mono_hz: float | None = 90.0
    exciter_mix: float = 0.5
    stereo_width: float = 1.2

    low_cut: int | None = 30
    high_cut: int | None = 18000
    phase_type: str = "minimal"


@passthrough_empty_audio
def resample(y, sr, target_sr=44100):
    validate_sample_rate(sr)
    validate_sample_rate(target_sr)
    if np.issubdtype(y.dtype, np.integer):
        y = librosa.util.buf_to_float(y)
    else:
        y = y.astype(np.float64)

    y_out = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_out


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
def _limiter_smooth_env(env, attack_coef, release_coef):
    out = np.empty_like(env)
    prev = env[0]

    is_adaptive = release_coef.ndim > 0

    for i in range(len(env)):
        rc = release_coef[i] if is_adaptive else release_coef

        coef = attack_coef if env[i] > prev else rc

        prev = coef * prev + (1.0 - coef) * env[i]
        out[i] = prev

    return out


@njit(cache=True, fastmath=True)
def _limiter_release(gs_db, rc_fast, rc_slow, w_slow):
    """Dual-release gain recovery (all values in dB, <= 0).

    Both followers jump to the required gain instantly (the look-ahead already
    made that ramp smooth) and recover exponentially. The output is a blend of
    a fast and a slow follower, so it is never above `gs_db` and a lone
    transient recovers quickly while sustained reduction recovers slowly.
    """
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


def _saturate(band: np.ndarray, sr: int, drive: float, oversample: int) -> np.ndarray:
    if oversample <= 1:
        return np.tanh(band * drive)

    kernel = _build_oversampling_kernel(sr, oversample, EXCITER_KAISER_BETA)
    rows = []
    for row in band:
        up = resample_poly(row, up=oversample, down=1, window=kernel)
        distorted = np.tanh(up.astype(np.float32) * drive)
        rows.append(
            resample_poly(distorted, up=1, down=oversample, window=kernel)[
                : row.shape[-1]
            ]
        )
    return np.stack(rows).astype(np.float64)


def _excite_band(
    y: np.ndarray,
    sr: int,
    band_hz: tuple[float, float],
    settings: ExciterSettings,
) -> np.ndarray:
    nyquist = sr / 2.0
    low_hz, high_hz = band_hz
    band_sos = butter(
        EXCITER_FILTER_ORDER,
        [low_hz / nyquist, high_hz / nyquist],
        btype="band",
        output="sos",
    )
    band = sosfilt(band_sos, y)
    peak_band = np.max(np.abs(band))
    if peak_band <= EXCITER_SILENCE_LEVEL:
        return np.zeros_like(y)

    drive = EXCITER_PEAK_DRIVE / peak_band if settings.drive is None else settings.drive
    is_oversampling_needed = high_hz * EXCITER_MAX_HARMONIC_ORDER > nyquist
    saturated = _saturate(
        band, sr, drive, settings.oversample if is_oversampling_needed else 1
    )

    harmonics_sos = butter(
        EXCITER_FILTER_ORDER,
        [high_hz * EXCITER_GUARD_RATIO / nyquist, settings.ceiling_hz / nyquist],
        btype="band",
        output="sos",
    )
    harmonics = sosfilt(harmonics_sos, saturated)
    rms_band = np.sqrt(np.mean(band**2))
    rms_harmonics = np.sqrt(np.mean(harmonics**2))

    return harmonics * (rms_band / (rms_harmonics + 1e-12))


def _generate_harmonics(
    y: np.ndarray,
    sr: int,
    edges_hz: np.ndarray,
    settings: ExciterSettings,
) -> np.ndarray:
    harmonics = np.zeros_like(y)
    for band_hz in pairwise(edges_hz[:-1]):
        harmonics += _excite_band(y, sr, band_hz, settings)
    return harmonics


def _compute_band_centers_hz(low_hz: float, ceiling_hz: float) -> np.ndarray:
    count = int(np.ceil(np.log2(ceiling_hz / low_hz) * EXCITER_BANDS_PER_OCTAVE))
    return low_hz * 2.0 ** (np.arange(count + 1) / EXCITER_BANDS_PER_OCTAVE)


def _measure_band_powers_db(
    y: np.ndarray, sr: int, centers_hz: np.ndarray
) -> np.ndarray:
    mono = np.mean(y, axis=0)
    freqs, psd = welch(mono, fs=sr, nperseg=min(EXCITER_NPERSEG, mono.shape[-1]))
    half_band = 2.0 ** (0.5 / EXCITER_BANDS_PER_OCTAVE)
    powers = np.array(
        [
            np.sum(psd[(freqs >= center / half_band) & (freqs < center * half_band)])
            for center in centers_hz
        ]
    )
    return 10.0 * np.log10(np.maximum(powers, EXCITER_POWER_FLOOR))


def _project_levels_db(dry_db: np.ndarray, centers_hz: np.ndarray) -> np.ndarray:
    octaves_between = np.log2(centers_hz[None, :] / centers_hz[:, None])
    projections = dry_db[:, None] - EXCITER_FILL_SLOPE_DB_PER_OCTAVE * octaves_between
    is_lower_band = octaves_between > 0.0
    return np.max(np.where(is_lower_band, projections, -np.inf), axis=0)


def _compute_fill_gains(
    dry_db: np.ndarray, harmonics_db: np.ndarray, centers_hz: np.ndarray, mix: float
) -> np.ndarray:
    target_db = np.maximum(_project_levels_db(dry_db, centers_hz), dry_db)
    deficit_power = 10.0 ** (target_db / 10.0) - 10.0 ** (dry_db / 10.0)
    gains = np.sqrt(mix * deficit_power / 10.0 ** (harmonics_db / 10.0))
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
    return firwin2(EXCITER_FIR_TAPS, grid_hz / (sr / 2.0), curve)


@passthrough_empty_audio
def apply_exciter(
    y: np.ndarray,
    sr: int,
    low_hz: float = EXCITER_LOW_HZ,
    high_hz: float = EXCITER_HIGH_HZ,
    drive: float = None,
    oversample: int = 4,
    mix: float = 0.2,
) -> np.ndarray:
    validate_sample_rate(sr)
    y = stereo(y, sr=sr)
    if mix <= 0.0 or y.shape[-1] < EXCITER_MIN_SAMPLES:
        return y

    ceiling_hz = min(high_hz, sr / 2.0 * EXCITER_NYQUIST_FRACTION)
    if not 0.0 < low_hz < ceiling_hz:
        raise InvalidAudioInputError(
            f"Exciter span must satisfy 0 < low_hz < high_hz, got "
            f"low_hz={low_hz} and usable high_hz={ceiling_hz:.1f} at {sr} Hz",
            recovery_hint="Pass a span such as low_hz=20.0, high_hz=20000.0.",
            context={"low_hz": low_hz, "high_hz": high_hz, "sample_rate": sr},
        )
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
    gains = np.where(centers_hz >= edges_hz[1] * EXCITER_GUARD_RATIO, gains, 0.0)
    air = signal.oaconvolve(
        harmonics,
        _build_fill_filter(centers_hz, gains, sr)[None, :],
        mode="same",
        axes=-1,
    )

    logger.debug(
        f"Exciter applied: span {low_hz:.1f}-{ceiling_hz:.1f} Hz, mix {mix:.2f}, "
        f"air RMS {np.sqrt(np.mean(air**2)):.6f}"
    )

    return y + air


@passthrough_empty_audio
def apply_bass_enhancer(
    y: np.ndarray,
    sr: int,
    f_lo: float = 35.0,
    f_hi: float = 160.0,
    f_hp: float = 90.0,
    f_lp: float = 900.0,
    mix: float = 0.3,
    drive: float = 2.5,
) -> np.ndarray:
    """Psychoacoustic bass: add the 2nd/3rd harmonics of the 35-160 Hz band.

    The harmonics (not the fundamental) are what make bass audible and "thick"
    on laptop/phone/car speakers and give it weight on big systems, without
    spending sub-bass headroom. They are generated from the mono sum of the
    bass so they stay centred, and level-matched to the bass (`mix` is a
    ratio: 0.3 = harmonics at 30 % of the bass RMS).
    """
    validate_sample_rate(sr)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if mix <= 0.0 or y.shape[-1] < 256:
        return y

    ny = sr / 2.0
    bass = sosfiltfilt(
        butter(2, [f_lo / ny, f_hi / ny], btype="band", output="sos"), y, axis=-1
    )
    rms_b = float(np.sqrt(np.mean(bass**2)))
    if rms_b < 1e-9:
        return y

    u = np.mean(bass, axis=0) / (3.0 * rms_b)
    a = np.abs(u)
    harm = np.tanh(drive * u) + 0.5 * (a - np.mean(a))
    harm = sosfiltfilt(butter(2, f_hp / ny, btype="high", output="sos"), harm)
    harm = sosfiltfilt(butter(2, min(f_lp / ny, 0.99), btype="low", output="sos"), harm)

    rms_h = float(np.sqrt(np.mean(harm**2)))
    harm = harm * (mix * rms_b / (rms_h + 1e-12))

    logger.debug(f"Bass enhancer: bass RMS {rms_b:.5f}, harmonics RMS {mix * rms_b:.5f}")
    return y + harm[None, :]


def stereo(*y_s, lr=True, sr=44100):
    validate_sample_rate(sr)
    for y in y_s:
        validate_audio_not_empty(y)
    stereo_result = None

    if lr and len(y_s) == 2:
        logger.debug("Processing stereo pair with LR mode")
        a, b = y_s
        a = stereo(a)
        b = stereo(b)
        _l, _r = pad_audio(a, b)
        y_s = [np.vstack([_l[0], _r[1]])]

    for y in y_s:
        was_mono = y.ndim == 1
        if was_mono:
            logger.debug(
                f"Input is mono, converting to stereo. Original shape: {y.shape}, dtype: {y.dtype}"
            )
            y = y[np.newaxis, :]
            y = np.vstack([y, y])
        if y.ndim > 2:
            logger.debug(
                f"Input has more than 2 channels, taking first two. Original shape: {y.shape}, dtype: {y.dtype}"
            )
            y = np.vstack([y[0], y[1]])
        if not was_mono and y.shape[0] > 2 and y.shape[1] < y.shape[0]:
            logger.debug(
                f"Transposing input to have shape (channels, samples). Original shape: {y.shape}, dtype: {y.dtype}"
            )
            y = y.T

        y = np.asarray(y, dtype=np.float64)

        if stereo_result is None:
            logger.debug(
                f"Initializing stereo result with first input. Shape: {y.shape}, dtype: {y.dtype}"
            )
            stereo_result = y
        else:
            logger.debug(
                f"Mixing input into stereo result. Input shape: {y.shape}, dtype: {y.dtype}, Current stereo result shape: {stereo_result.shape}, dtype: {stereo_result.dtype}"
            )
            region_s = max(stereo_result.shape[-1], y.shape[-1]) / sr
            stereo_result = mix_audio(
                stereo_result,
                y,
                sr,
                weight=0.5,
                fade_duration=min(2.0, region_s),
            )

    logger.debug(
        f"Final stereo result shape: {stereo_result.shape}, dtype: {stereo_result.dtype}, max: {np.max(np.abs(stereo_result)):.6f}"
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

    logger.debug(
        f"Padded audio to length {max_len}. Y1 shape: {y1_padded.shape}, dtype: {y1_padded.dtype}, Y2 shape: {y2_padded.shape}, dtype: {y2_padded.dtype}"
    )
    logger.debug(
        f"Y1 stats after padding - min: {y1_padded.min():.6f}, max: {y1_padded.max():.6f}, mean: {y1_padded.mean():.6f}"
    )
    logger.debug(
        f"Y2 stats after padding - min: {y2_padded.min():.6f}, max: {y2_padded.max():.6f}, mean: {y2_padded.mean():.6f}"
    )

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
    prof = getattr(SmartMastering, "_profile", None)
    if prof is not None:
        start = __import__("time").perf_counter()

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

    logger.debug(
        f"Applied low-pass filter to Y1 with cutoff {cutoff_hz} Hz. Filtered Y1 stats - min: {y1_filtered.min():.6f}, max: {y1_filtered.max():.6f}, mean: {y1_filtered.mean():.6f}"
    )

    window_size = int(sr * 0.1)
    envelope = np.abs(y2_padded.mean(axis=1))
    envelope = np.convolve(
        envelope, np.ones(window_size) / window_size, mode="same"
    )

    if np.max(envelope) > 0:
        envelope = envelope / np.max(envelope)
    envelope = envelope[:, np.newaxis]

    logger.debug(
        f"Computed envelope for Y2. Envelope stats - min: {envelope.min():.6f}, max: {envelope.max():.6f}, mean: {envelope.mean():.6f}"
    )

    y1_processed = (y1_padded * (1 - envelope)) + (y1_filtered * envelope)

    duck_mask = 1.0 - (envelope * (1.0 - duck_depth))

    y_length = len(y1_padded)

    validate_fade_duration(fade_duration, y_length / sr)
    fade_samples = max(1, min(int(fade_duration * sr), y_length))
    t = np.linspace(0, np.pi / 2, fade_samples).reshape(-1, 1)
    fade_in, fade_out = np.sin(t), np.cos(t)

    logger.debug(
        f"Fade in stats - min: {fade_in.min():.6f}, max: {fade_in.max():.6f}, mean: {fade_in.mean():.6f}"
    )
    logger.debug(
        f"Fade out stats - min: {fade_out.min():.6f}, max: {fade_out.max():.6f}, mean: {fade_out.mean():.6f}"
    )

    mask1 = np.ones((y_length, 1)) * weight
    mask1[-fade_samples:] *= fade_out

    mask2 = np.zeros((y_length, 1))
    mask2[:fade_samples] = (1.0 - weight) * fade_in
    mask2[fade_samples:] = 1.0 - weight

    y1_weighted = y1_processed * mask1 * duck_mask
    y2_weighted = y2_padded * mask2

    logger.debug("Applied mixing masks")
    logger.debug(
        f"Y1 weighted stats - min: {y1_weighted.min():.6f}, max: {y1_weighted.max():.6f}, mean: {y1_weighted.mean():.6f}"
    )
    logger.debug(
        f"Y2 weighted stats - min: {y2_weighted.min():.6f}, max: {y2_weighted.max():.6f}, mean: {y2_weighted.mean():.6f}"
    )

    y_mixed = y1_weighted + y2_weighted

    max_val = np.max(np.abs(y_mixed))
    if max_val > 1.0:
        logger.debug(
            f"Normalizing mixed audio. Max value before normalization: {max_val:.6f}"
        )
        y_mixed /= max_val

    if y_mixed.shape[0] > y_mixed.shape[1]:
        logger.debug(
            f"Transposing mixed audio to have shape (channels, samples). Original shape: {y_mixed.shape}, dtype: {y_mixed.dtype}"
        )
        y_mixed = y_mixed.T

    if prof is not None:
        logger.debug(
            f"Finished mixing audio. Mixed audio shape: {y_mixed.shape}, dtype: {y_mixed.dtype}, max: {np.max(np.abs(y_mixed)):.6f}"
        )
        end = __import__("time").perf_counter()
        prof("mix_audio", end - start)

    logger.debug(
        f"Final mixed audio stats - min: {y_mixed.min():.6f}, max: {y_mixed.max():.6f}, mean: {y_mixed.mean():.6f}"
    )
    logger.debug(
        f"Mixing parameters - weight: {weight:.2f}, fade_duration: {fade_duration:.2f} s, duck_depth: {duck_depth:.2f}, cutoff_hz: {cutoff_hz}"
    )

    return y_mixed


@passthrough_empty_audio
def freq_cut(y: np.ndarray, sr: int, low_cut: int | None, high_cut: int | None):
    validate_sample_rate(sr)
    y = np.array(y, dtype=np.float64)

    if y.size == 0:
        return y

    y -= np.mean(y, axis=-1, keepdims=True)

    nyquist = sr / 2.0
    min_len = 3 * (2 * 4 + 1) * 2 + 1

    if y.shape[-1] > min_len:
        if high_cut and 0 < high_cut < nyquist:
            sos = butter(8, high_cut / nyquist, btype="low", output="sos")
            y = sosfiltfilt(sos, y, axis=-1)

        if low_cut and 0 < low_cut < nyquist:
            sos = butter(4, low_cut / nyquist, btype="high", output="sos")
            y = sosfiltfilt(sos, y, axis=-1)

    logger.debug(
        f"Applied frequency cuts. Low cut: {low_cut} Hz, High cut: {high_cut} Hz"
    )
    logger.debug(
        f"Output shape after freq cut: {y.shape}, dtype: {y.dtype}, max: {np.max(np.abs(y)):.6f}"
    )

    return y


class SmartMastering:
    _profiling_data: dict[str, float] = {}

    def update_bands(self) -> None:
        count = self.multiband_params.num_bands
        freq_min = self.low_cut or 40.0
        freq_max = self.high_cut or min(
            self.resampling_target / 2 - 0.1, 15000.0
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

    def _timed(self, name: str):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                import time

                start = time.perf_counter()
                result = fn(*args, **kwargs)
                end = time.perf_counter()
                self._profile(name, end - start)
                return result

            return wrapper

        return decorator

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

        self.slope_hz = (
            cfg.slope_hz
            if cfg.slope_hz is not None
            else SmartMasteringConfig().slope_hz
        )
        self.smoothing_fraction = (
            cfg.smoothing_fraction
            if cfg.smoothing_fraction is not None
            else SmartMasteringConfig().smoothing_fraction
        )
        self.config = cfg

        self.mix_params = (
            cfg.mix if cfg.mix is not None else SmartMasteringConfig().mix
        )
        self.multiband_params = (
            cfg.multiband
            if cfg.multiband is not None
            else SmartMasteringConfig().multiband
        )

        self.limiter_defaults = (
            cfg.limiter
            if cfg.limiter is not None
            else SmartMasteringConfig().limiter
        )
        self.target_lufs = (
            cfg.target_lufs
            if cfg.target_lufs is not None
            else SmartMasteringConfig().target_lufs
        )
        self.low_cut = (
            cfg.low_cut
            if cfg.low_cut is not None
            else SmartMasteringConfig().low_cut
        )
        self.high_cut = (
            cfg.high_cut
            if cfg.high_cut is not None
            else SmartMasteringConfig().high_cut
        )
        self.correction_strength = (
            cfg.correction_strength
            if cfg.correction_strength is not None
            else SmartMasteringConfig().correction_strength
        )
        self.phase_type = (
            cfg.phase_type
            if cfg.phase_type is not None
            else SmartMasteringConfig().phase_type
        )
        self.work_lufs = cfg.work_lufs
        self.max_correction_db = cfg.max_correction_db
        self.presence_db = cfg.presence_db
        self.presence_hz = cfg.presence_hz
        self.presence_octaves = cfg.presence_octaves
        self.exciter_mix = cfg.exciter_mix
        self.stereo_width = cfg.stereo_width
        self.bass_db, self.bass_hz, self.bass_octaves = cfg.bass_db, cfg.bass_hz, cfg.bass_octaves
        self.warmth_db, self.warmth_hz, self.warmth_octaves = cfg.warmth_db, cfg.warmth_hz, cfg.warmth_octaves
        self.air_db, self.air_hz, self.air_octaves = cfg.air_db, cfg.air_hz, cfg.air_octaves
        self.bass_enhance_mix = cfg.bass_enhance_mix
        self.bass_mono_hz = cfg.bass_mono_hz

        self.update_bands()
        self.multiband_params.bands = self.bands

        self.nperseg = int(2 ** np.ceil(np.log2(self.sr / 5)))

        self.target_freqs_hz = np.array(
            generate_bands(0.1, self.sr / 2.0, self.nperseg),
            dtype=float,
        )

    def _measure_spectrum(self, y_mono: np.ndarray) -> np.ndarray:
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

        logger.debug(
            f"Welch method completed. Frequency bins: {len(f_axis)}, PSD bins: {len(psd)}"
        )

        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
        psd_db = np.clip(psd_db, FLOOR_DB, CEILING_DB)

        logger.debug(
            f"Measured spectrum. Frequency axis shape: {f_axis.shape}, PSD shape: {psd_db.shape}"
        )
        logger.debug(
            f"Frequency axis stats - min: {f_axis.min():.1f} Hz, max: {f_axis.max():.1f} Hz, mean: {f_axis.mean():.1f} Hz"
        )
        logger.debug(
            f"PSD stats - min: {psd_db.min():.2f} dB, max: {psd_db.max():.2f} dB, mean: {psd_db.mean():.2f} dB"
        )

        return psd_db, f_axis

    def _build_target_spectrum(self, f_axis: np.ndarray) -> np.ndarray:
        min_freq = self.low_cut if self.low_cut else 0.1
        max_freq = self.high_cut if self.high_cut else f_axis[-1]

        f_axis_safe = np.maximum(f_axis, min_freq)
        f_axis_safe = np.minimum(f_axis_safe, max_freq)

        octaves = np.log2(f_axis_safe / self.slope_hz)

        target_db = -self.slope_db * octaves

        logger.debug(
            f"Built target spectrum. Shape: {target_db.shape}, dtype: {target_db.dtype}, min: {target_db.min():.2f} dB, max: {target_db.max():.2f} dB, mean: {target_db.mean():.2f} dB"
        )
        logger.debug(
            f"Target spectrum frequency axis stats - min: {f_axis.min():.1f} Hz, max: {f_axis.max():.1f} Hz, mean: {f_axis.mean():.1f} Hz"
        )
        logger.debug(
            f"Target spectrum parameters - slope_db: {self.slope_db:.2f} dB/octave, slope_hz: {self.slope_hz:.1f} Hz, low_cut: {self.low_cut}, high_cut: {self.high_cut}"
        )

        return target_db

    def _smooth_target_curve(
        self,
        target_db: np.ndarray,
        f_axis: np.ndarray,
        smoothing_fraction: float | None = None,
    ) -> np.ndarray:
        """Smooth a dB-vs-frequency curve with a Gaussian in log-frequency.

        `smoothing_fraction` is the smoothing width in octaves (1/3 = third
        octave). The previous implementation did nothing at 0.0 and was an
        O(n^2) python loop otherwise.
        """
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
        """Intentional tonal shaping in dB: bass, warmth, presence bells and an
        air shelf. Applied on top of (not scaled by) correction_strength."""
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
            total += self.air_db / (1.0 + np.exp(-np.log2(f / self.air_hz) / width))
        return total

    def _phase(self, y: np.ndarray, tp: str) -> np.ndarray:
        if y.ndim == 2 and y.shape[1] < y.shape[0]:
            return self._phase(y.T, tp).T

        start = __import__("time").perf_counter()

        orig_len = y.shape[-1]
        y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

        logger.debug(
            f"Measuring spectrum for phase correction. Input shape: {y_mono.shape}, dtype: {y_mono.dtype}, sample rate: {self.sr} Hz"
        )

        input_db, f_axis = self._measure_spectrum(y_mono)
        target_db = self._build_target_spectrum(f_axis)

        logger.debug(
            f"Measured input spectrum. Shape: {input_db.shape}, dtype: {input_db.dtype}"
        )
        logger.debug(
            f"Built target spectrum. Shape: {target_db.shape}, dtype: {target_db.dtype}"
        )

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

        lo = float(self.low_cut or 20.0)
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

        logger.debug(
            f"Calculated correction curve. Shape: {correction_db.shape}, dtype: {correction_db.dtype}, min: {correction_db.min():.2f} dB, max: {correction_db.max():.2f} dB, mean: {correction_db.mean():.2f} dB"
        )

        correction_db_norm = correction_db - np.max(correction_db)
        H_lin = 10.0 ** (correction_db_norm / 20.0)

        logger.debug(
            f"Converted correction to linear scale. Shape: {H_lin.shape}, dtype: {H_lin.dtype}, min: {H_lin.min():.6f}, max: {H_lin.max():.6f}"
        )

        out = None

        if tp == "minimal":
            log_mag = np.log(np.maximum(H_lin, 1e-12))

            cepstrum = np.fft.irfft(log_mag)
            n = len(cepstrum)

            lifter = np.zeros(n)
            lifter[0] = 1.0
            lifter[1 : n // 2] = 2.0
            if n % 2 == 0:
                lifter[n // 2] = 1.0

            logger.debug(
                f"Applying minimal phase correction. Cepstrum length: {n}, lifter shape: {lifter.shape}"
            )

            causal_cepstrum = cepstrum * lifter

            min_phase_spectrum = np.exp(np.fft.rfft(causal_cepstrum))

            h_ir = np.fft.irfft(min_phase_spectrum)

            FIR_LEN = min(len(h_ir), 8192)
            h_ir_cut = h_ir[:FIR_LEN]

            window = np.ones(FIR_LEN)
            fade_size = FIR_LEN // 2
            if fade_size > 0:
                window[-fade_size:] = np.hamming(fade_size * 2)[fade_size:]

            logger.debug(
                f"Applied windowing to impulse response. Window shape: {window.shape}"
            )

            h_ir_final = h_ir_cut * window

            h_ir_final = h_ir_final.reshape((1,) * (y.ndim - 1) + (-1,))

            out = signal.oaconvolve(y, h_ir_final, mode="full")

            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "minimal_phase output contains non-finite values"
            )
            end = __import__("time").perf_counter()
            self._profile("minimal_phase", end - start)

            logger.debug(
                f"Applied minimal phase correction. Output shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
            )
            logger.debug(
                f"Correction curve stats - min: {correction_db.min():.2f} dB, max: {correction_db.max():.2f} dB, mean: {correction_db.mean():.2f} dB"
            )
            logger.debug(
                f"Impulse response stats - min: {h_ir_final.min():.6f}, max: {h_ir_final.max():.6f}, mean: {h_ir_final.mean():.6f}"
            )

        elif tp == "linear":
            H_full = np.concatenate([H_lin, H_lin[-2:0:-1]])

            h = np.real(np.fft.ifft(H_full))

            logger.debug(
                f"Calculated linear phase impulse response. Length: {len(h)}, dtype: {h.dtype}, max: {np.max(np.abs(h)):.6f}"
            )

            h_centered = np.fft.fftshift(h)
            FIR_LEN = min(len(h), 8192)

            logger.debug(
                f"Centered impulse response. Length: {len(h_centered)}, dtype: {h_centered.dtype}, max: {np.max(np.abs(h_centered)):.6f}"
            )

            center = len(h_centered) // 2
            half_len = FIR_LEN // 2
            h_final = h_centered[center - half_len : center + half_len]

            logger.debug(
                f"Extracted central portion of impulse response. Length: {len(h_final)}, dtype: {h_final.dtype}, max: {np.max(np.abs(h_final)):.6f}"
            )

            h_final *= signal.windows.hamming(len(h_final))
            h_final = h_final.reshape((1,) * (y.ndim - 1) + (-1,))

            out = signal.oaconvolve(y, h_final, mode="same")
            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "linear_phase output contains non-finite values"
            )
            end = __import__("time").perf_counter()
            self._profile("linear_phase", end - start)

            logger.debug(
                f"Applied linear phase correction. Output shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
            )
            logger.debug(
                f"Correction curve stats - min: {correction_db.min():.2f} dB, max: {correction_db.max():.2f} dB, mean: {correction_db.mean():.2f} dB"
            )
            logger.debug(
                f"Impulse response stats - min: {h_final.min():.6f}, max: {h_final.max():.6f}, mean: {h_final.mean():.6f}"
            )

        else:
            raise InvalidAudioInputError(
                f"Unknown phase type: {tp}",
                recovery_hint="Use one of the supported phase correction types.",
                context={"phase_type": tp},
            )

        if out is None:
            raise RuntimeError("Phase correction failed to produce output")

        logger.debug(
            f"Output before final length adjustment. Shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
        )

        out = (
            out[..., :orig_len]
            if out.shape[-1] > orig_len
            else np.pad(
                out,
                (*((0, 0),) * (out.ndim - 1), (0, orig_len - out.shape[-1])),
            )
        )

        logger.debug(
            f"Final output after phase correction. Shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
        )

        return out

    def _compress_band(
        self, band: np.ndarray, cfg: MultibandBand, f_lo: float
    ) -> np.ndarray:
        """Compress one band for all channels with a single, linked gain.

        * level detector: sliding RMS (window follows the band's lowest
          frequency) - the old per-sample |x| in dB had -240 dB dips at every
          zero crossing and modulated the gain at audio rate (distortion).
        * gain computer -> ballistics on the *gain reduction* in dB.
        * channels share one gain, so the stereo image does not wobble.
        """
        sr = self.sr
        win = int(np.clip(sr * 1.5 / max(f_lo, 20.0), sr * 0.002, sr * 0.025))
        power = np.max(
            uniform_filter1d(band**2, size=win, axis=-1, mode="nearest"),
            axis=0,
        )
        det_db = 10.0 * np.log10(power + 1e-14)

        active = det_db > (det_db.max() - 60.0)
        if np.count_nonzero(active) < 16 or det_db.max() < -100.0:
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

        gain = 10.0 ** (-gr / 20.0)

        if self.multiband_params.auto_makeup:
            e_in = float(np.sum(band**2))
            e_out = float(np.sum((band * gain) ** 2))
            if e_out > 1e-20:
                gain = gain * min(np.sqrt(e_in / e_out), 10.0 ** (6.0 / 20.0))

        gain = gain * 10.0 ** (cfg.makeup_db / 20.0)

        logger.debug(
            f"Band <= {cfg.fc:.0f} Hz: thr {thr:.1f} dB, ratio {cfg.ratio:.2f}, "
            f"attack {cfg.attack_ms:.1f} ms, release {cfg.release_ms:.1f} ms, "
            f"mean GR {gr.mean():.2f} dB, max GR {gr.max():.2f} dB"
        )

        return band * gain

    def _multiband_compress(self, y: np.ndarray) -> np.ndarray:
        start = __import__("time").perf_counter()

        was_1d = y.ndim == 1
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))

        lo = float(self.low_cut or 40.0)
        hi = float(self.high_cut or self.sr / 2.0)
        sorted_bands = sorted(self.bands, key=lambda b: b.fc)
        kept = [b for b in sorted_bands if lo * 1.02 < b.fc < hi * 0.98]

        if y2.shape[-1] < 512 or not kept:
            return y

        fcs = [b.fc for b in kept]
        cfgs = kept + [sorted_bands[-1]]

        out = np.zeros_like(y2)
        current = y2
        for i, cfg in enumerate(cfgs):
            if i < len(fcs):
                wn = float(np.clip(fcs[i] / (self.sr / 2.0), 1e-4, 0.999))
                sos = signal.butter(2, wn, btype="low", output="sos")
                band = signal.sosfiltfilt(sos, current, axis=-1)
                current = current - band
            else:
                band = current
            f_lo = fcs[i - 1] if i > 0 else lo
            out += self._compress_band(band, cfg, f_lo)

        self._profile("multiband_compress", __import__("time").perf_counter() - start)

        return out[0] if was_1d else out

    def _stereo_widen(self, y: np.ndarray, width: float | None = None) -> np.ndarray:
        """Bass-mono + widening of the side signal above 200 Hz.

        * everything below `bass_mono_hz` is mono: tighter, more powerful low
          end that collapses safely to mono systems.
        * `width` > 1 widens the side signal above ~200 Hz (mid untouched).
          Wider = more spacious / emotional; narrower = closer to the listener.
        """
        width = self.stereo_width if width is None else width
        widen = abs(width - 1.0) >= 1e-3
        mono_hz = self.bass_mono_hz
        if y.ndim < 2 or y.shape[0] != 2 or not (widen or mono_hz):
            return y

        mid = (y[0] + y[1]) * 0.5
        side = (y[0] - y[1]) * 0.5

        if mono_hz:
            sos = signal.butter(2, mono_hz / (self.sr / 2.0), btype="high", output="sos")
            side = signal.sosfiltfilt(sos, side)
        if widen:
            sos = signal.butter(2, 200.0 / (self.sr / 2.0), btype="high", output="sos")
            side = side + (width - 1.0) * signal.sosfiltfilt(sos, side)

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
        """Gain (dB, <= 0) from a linked true-peak envelope at rate `sr`.

        gain[n] <= limit/peak[n] for every n (hard guarantee, no overshoot):
        1. required reduction per sample (dB),
        2. look-ahead minimum over [n, n+L-1],
        3. causal moving average over [n-L+1, n] - every term of the average
           contains the sample n in its minimum window, so the mean is still
           <= the requirement at n. This gives a smooth L-sample attack ramp.
        4. dual-release recovery.
        """
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

    def _limiter_prepare(self, y: np.ndarray, p: LimiterParams) -> list[np.ndarray]:
        """Oversample; optionally split into [bass, rest] (zero-phase LR4).

        Limiting the bass separately is what stops a kick from ducking the
        vocal: a single broadband limiter has to turn *everything* down when a
        50 Hz peak arrives, which is heard as "breathing".
        """
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
        kernel = _build_oversampling_kernel(self.sr, p.os_factor, p.kaiser_beta)

        def up(a):
            rows = [
                signal.resample_poly(
                    row, p.os_factor, 1, window=kernel
                ).astype(np.float32)
                for row in a
            ]
            return np.ascontiguousarray(np.stack(rows))

        if not p.split_hz or y2.shape[-1] < 512:
            return [up(y2)]

        wn = float(np.clip(p.split_hz / (self.sr / 2.0), 1e-4, 0.5))
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
        """Channel-linked look-ahead limiter on an oversampled array (in place).

        The peak envelope is the true (oversampled) peak per base-rate sample,
        and the gain is applied to every oversampled sample of that interval,
        so `|x| <= ceiling_lin` afterwards is guaranteed.
        """
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
        """tanh soft clipper (continuous slope at `knee`, asymptote `limit`)."""
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
        """drive -> bass limiter -> broadband limiter -> soft clipper -> base rate."""
        os_factor = p.os_factor
        limit_lin = 10.0 ** ((ceil_db - p.margin_db) / 20.0)
        lim1 = limit_lin * 10.0 ** (p.clip_ceiling_db / 20.0)
        drive = 10.0 ** (drive_db / 20.0)

        if len(bands_os) == 2:
            ceil_lo = lim1 * p.band_ceiling_lo
            lo = bands_os[0] * drive
            g_lo = self._limit_band_inplace(
                lo, ceil_lo * 10.0 ** (p.bass_clip_db / 20.0), p.lo_lookahead_ms,
                p.lo_release_fast_ms, p.lo_release_slow_ms, p.lo_slow_weight,
                os_factor,
            )
            self._soft_clip_inplace(lo, p.bass_clip_knee * ceil_lo, ceil_lo)
            x = bands_os[1] * drive
            x += lo
            del lo
            logger.debug(
                f"Bass limiter GR: mean {g_lo.mean():.2f} dB / max {g_lo.min():.2f} dB"
            )
        else:
            x = bands_os[0] * drive

        gain_db = self._limit_band_inplace(
            x, lim1, p.lookahead_ms,
            p.release_fast_ms, p.release_slow_ms, p.slow_weight, os_factor,
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
            window=_build_oversampling_kernel(self.sr, os_factor, p.kaiser_beta),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Limiter: drive {drive_db:.2f} dB, ceiling {ceil_db:.2f} dB, "
                f"broadband GR mean {gain_db.mean():.2f} dB / max {gain_db.min():.2f} dB"
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
        """Single-shot clipper + limiter with a fixed drive."""
        start = __import__("time").perf_counter()

        p = replace(self.limiter_defaults, **overrides) if overrides else self.limiter_defaults
        was_1d = np.ndim(y) == 1
        orig_len = np.shape(y)[-1]

        x_os = self._limiter_prepare(y, p)
        out = self._limit_oversampled(x_os, drive_db, ceil_db, p)
        out = out[..., :orig_len]

        self._profile("limiter", __import__("time").perf_counter() - start)
        return out[0] if was_1d else out

    def _limit_to_target(
        self, y: np.ndarray, target_lufs: float | None, p: LimiterParams | None = None
    ) -> np.ndarray:
        start = __import__("time").perf_counter()
        p = p or self.limiter_defaults
        y2 = np.atleast_2d(np.asarray(y, dtype=np.float64))
        orig_len = y2.shape[-1]

        x_os = self._limiter_prepare(y2, p)

        current = self._measure_lufs(y2)
        if target_lufs is None or current is None:
            out = self._limit_oversampled(x_os, 0.0, p.ceil_db, p)[..., :orig_len]
            return out

        norm = 10.0 ** ((target_lufs - current) / 20.0)
        for band_os in x_os:
            band_os *= norm
        drive = 3.0
        history: list[tuple[float, float]] = []
        best = None
        previous_out = None
        best_err = np.inf

        for it in range(max(1, p.max_iterations)):
            out = self._limit_oversampled(x_os, drive, p.ceil_db, p)[..., :orig_len]
            measured = self._measure_lufs(out)
            if measured is None:
                best = out
                break
            err = target_lufs - measured

            if abs(err) < best_err:
                best_err, best = abs(err), out

            logger.debug(
                f"Limiter iteration {it}: drive {drive:.2f} dB -> {measured:.2f} LUFS "
                f"(target {target_lufs:.2f}, error {err:+.2f} LU)"
            )

            if abs(err) <= p.loudness_tol_lu:
                break

            history.append((drive, measured))
            if len(history) >= 2 and abs(history[-1][0] - history[-2][0]) > 1e-3:
                s = (history[-1][1] - history[-2][1]) / (history[-1][0] - history[-2][0])
                if s < p.min_efficiency_lu_per_db:
                    logger.debug(
                        f"Stopped at {history[-2][1]:.1f} LUFS and "
                        f"{history[-2][0]:.1f} dB drive: more drive adds "
                        f"{s:.2f} LU per dB, below "
                        f"{p.min_efficiency_lu_per_db:.2f}. Lower target_lufs."
                    )
                    best = previous_out
                    break
                s = float(np.clip(s, 0.25, 1.2))
            else:
                s = 0.6

            new_drive = float(np.clip(drive + err / s, 0.0, p.max_drive_db))
            if abs(new_drive - drive) < 0.02:
                if err > p.loudness_tol_lu and new_drive >= p.max_drive_db - 1e-6:
                    logger.warning(
                        f"Could not reach {target_lufs:.1f} LUFS: reached "
                        f"{measured:.1f} LUFS at max drive {p.max_drive_db:.1f} dB. "
                        "Raise limiter.max_drive_db or lower target_lufs."
                    )
                break
            drive = new_drive
            previous_out = out

        self._profile("limiter", __import__("time").perf_counter() - start)
        return best

    def _apply_rms(self, y: np.ndarray, rms: float):
        y *= rms / (
            (
                np.max(np.sqrt(np.mean(y**2, axis=1)))
                if y.ndim > 1
                else np.sqrt(np.mean(y**2))
            )
            + 1e-12
        )

        logger.debug(
            f"Applied RMS normalization. Target RMS: {rms:.6f}, Output stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )
        logger.debug(
            f"Original signal stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )

        return y

    @staticmethod
    def _k_weighting_coeffs(sr: float):
        """ITU-R BS.1770 K-weighting (shelf + RLB high-pass) for any sample
        rate. The previously hard-coded coefficients are only valid at 48 kHz.
        """
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
        a1 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])

        f0, q = 38.13547087602444, 0.5003270373238773
        k = np.tan(np.pi * f0 / sr)
        a0 = 1.0 + k / q + k * k
        b2 = np.array([1.0, -2.0, 1.0])
        a2 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
        return b1, a1, b2, a2

    def _measure_lufs(self, y: np.ndarray) -> float | None:
        """Gated integrated loudness (BS.1770-4). `None` if silent."""
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
        start = __import__("time").perf_counter()

        current = self._measure_lufs(y)
        if current is None:
            return y

        gain_lin = 10.0 ** ((target_lufs - current) / 20.0)
        out = y * gain_lin

        self._profile("lufs", __import__("time").perf_counter() - start)
        logger.debug(
            f"LUFS gain: {current:.2f} -> {target_lufs:.2f} LUFS "
            f"({20.0 * np.log10(gain_lin):+.2f} dB)"
        )
        return out

    @staticmethod
    def _validate_input(y, sr) -> None:
        if y is None:
            raise InvalidAudioInputError(
                EMPTY_AUDIO_MESSAGE,
                recovery_hint="Provide a non-empty numpy audio array.",
            )
        if sr is not None:
            try:
                validate_sample_rate(sr)
            except ValueError as exc:
                raise InvalidAudioInputError(
                    str(exc),
                    recovery_hint="Provide a sample rate between 8000 and 192000 Hz.",
                    context={"sample_rate": repr(sr)},
                ) from exc
        if not isinstance(y, np.ndarray) or y.ndim == 0:
            raise InvalidAudioInputError(
                f"Audio must be a 1D or 2D numpy array, got {type(y).__name__}",
                recovery_hint="Convert the input with np.asarray(audio, dtype=float).",
                context={"input_type": type(y).__name__},
            )
        if y.size == 0:
            return
        if y.ndim > 2:
            raise InvalidAudioInputError(
                f"Audio must be 1D or 2D, got {y.ndim} dimensions",
                recovery_hint="Reshape the audio to (channels, samples).",
                context={"shape": tuple(y.shape)},
            )
        if not (
            np.issubdtype(y.dtype, np.floating)
            or np.issubdtype(y.dtype, np.integer)
        ):
            raise InvalidAudioInputError(
                f"Audio must be numeric, got dtype {y.dtype}",
                recovery_hint="Convert the input to a float array.",
                context={"dtype": str(y.dtype)},
            )
        if np.issubdtype(y.dtype, np.floating) and not np.all(np.isfinite(y)):
            raise InvalidAudioInputError(
                "Audio contains NaN or infinite values",
                recovery_hint="Remove or replace non-finite samples before processing.",
            )

    def process(self, y: np.ndarray, sr: int = None) -> np.ndarray:
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

        logger.debug(
            f"Starting processing: shape {y.shape}, dtype {y.dtype}, sr {orig_sr} Hz"
        )

        y_resamp = y
        if self.sr != self.resampling_target:
            logger.debug(f"Resampling to {self.resampling_target} Hz...")
            y_resamp = resample(y_resamp, self.sr, self.resampling_target)
            self.sr = self.resampling_target

        expected_len = int(round(orig_len * self.sr / orig_sr))

        y_stereo = stereo(y_resamp)

        y_clean = freq_cut(y_stereo, self.sr, low_cut=self.low_cut, high_cut=None)

        y_work = self._apply_lufs(y_clean, self.work_lufs)

        y_eq = self._phase(y_work, self.phase_type)

        y_mbc = self._multiband_compress(y_eq)

        if self.bass_enhance_mix > 0.0:
            y_mbc = apply_bass_enhancer(y_mbc, self.sr, mix=self.bass_enhance_mix)

        if self.exciter_mix > 0.0:
            y_air = apply_exciter(
                y_mbc,
                self.sr,
                low_hz=self.low_cut or EXCITER_LOW_HZ,
                high_hz=self.high_cut or EXCITER_HIGH_HZ,
                mix=self.exciter_mix,
            )
        else:
            y_air = y_mbc

        y_wide = self._stereo_widen(stereo(y_air))

        y_band = freq_cut(
            y_wide, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

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

        if logger.isEnabledFor(logging.DEBUG):
            final_lufs = self._measure_lufs(y_clip)
            logger.debug(
                f"Done. peak {20 * np.log10(np.max(np.abs(y_clip)) + 1e-12):.2f} dBFS, "
                f"loudness {final_lufs if final_lufs is None else round(final_lufs, 2)} LUFS "
                f"(target {self.target_lufs}), samples at ceiling "
                f"{int(np.sum(np.abs(y_clip) >= lin_amp - 1e-9))}"
            )

        return y_clip, self.sr


def validation_report(master: SmartMastering, y: np.ndarray, sr: int):
    validate_sample_rate(sr)
    from scipy import signal

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
