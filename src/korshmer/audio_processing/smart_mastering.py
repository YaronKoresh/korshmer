import logging
from dataclasses import dataclass, field

import librosa
import numpy as np
from numba import njit
from scipy import signal
from scipy.ndimage import maximum_filter1d
from scipy.signal import (
    butter,
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
    num_bands: int = 5
    intensity: float = 1.0

    bass_ratio: float = 3.0
    bass_attack_ms: float = 20.0
    bass_release_ms: float = 150.0
    bass_threshold_db: float = -40.0

    treb_ratio: float = 2.0
    treb_attack_ms: float = 5.0
    treb_release_ms: float = 15.0
    treb_threshold_db: float = -20.0

    @classmethod
    def _make_bands_from_fcs(
        cls, fcs: list[float], freq_min: int, freq_max: int
    ) -> list[MultibandBand]:
        if not fcs:
            return []

        fcs_arr = np.array(fcs, dtype=float)

        ref_min = np.log2(freq_min)
        ref_max = np.log2(freq_max)

        fcs_safe = np.clip(fcs_arr, freq_min, freq_max)

        fcs_log = np.log2(fcs_safe)
        positions = (fcs_log - ref_min) / (ref_max - ref_min)

        base_thr = (
            cls.bass_threshold_db
            + (cls.treb_threshold_db - cls.bass_threshold_db) * positions
        )
        ratios = cls.bass_ratio + (cls.treb_ratio - cls.bass_ratio) * positions
        attacks = (
            cls.bass_attack_ms
            + (cls.treb_attack_ms - cls.bass_attack_ms) * positions
        )
        releases = (
            cls.bass_release_ms
            + (cls.treb_release_ms - cls.bass_release_ms) * positions
        )

        thr = np.mean(base_thr) + (base_thr - np.mean(base_thr)) * cls.intensity

        makeups = np.zeros_like(fcs_arr, dtype=float)

        ratios = 1.0 + (ratios - 1.0) * cls.intensity
        attacks *= 1.0 - 0.5 * cls.intensity
        releases *= 1.0 + 0.5 * cls.intensity

        knees = np.full_like(fcs, 6.0, dtype=float)

        return [
            MultibandBand(
                fc=float(fcs_arr[i]),
                base_threshold=float(thr[i]),
                ratio=float(ratios[i]),
                attack_ms=float(attacks[i]),
                release_ms=float(releases[i]),
                makeup_db=float(makeups[i]),
                knee_db=float(knees[i]),
            )
            for i in range(len(fcs_arr))
        ]


@dataclass
class LimiterParams:
    drive_db: float = 0.0
    ceil_db: float = -0.7


@dataclass
class SmartMasteringConfig:
    sample_rate: int | None = None
    slope_db: float = 3.5
    slope_hz: float = 1000.0
    smoothing_fraction: float = 1.0 / 3.0
    mix: MixParams = field(default_factory=MixParams)
    multiband: MultibandParams = field(default_factory=MultibandParams)
    limiter: LimiterParams = field(default_factory=LimiterParams)
    target_lufs: float = -7.0
    correction_strength: float = 1.0
    low_cut: int | None = 30
    high_cut: int | None = 17000
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


def calculate_dynamic_cutoff(
    y: np.ndarray, sr: int, rolloff_percent: float
) -> float:
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

    freqs, psd = welch(y_mono, fs=sr, nperseg=min(4096, y_mono.shape[-1]))

    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]

    rolloff_index = np.argmax(
        cumulative_energy >= (rolloff_percent * total_energy)
    )
    optimal_cutoff_hz = freqs[rolloff_index]

    optimal_cutoff_hz = np.clip(optimal_cutoff_hz, 4000.0, 12000.0)

    logger.debug(
        f"Calculated dynamic exciter cutoff: {optimal_cutoff_hz:.1f} Hz (rolloff at {rolloff_percent * 100:.1f}%)"
    )
    logger.debug(
        f"Total energy: {total_energy:.6e}, Rolloff energy: {cumulative_energy[rolloff_index]:.6e}"
    )
    logger.debug(
        f"Rolloff frequency: {freqs[rolloff_index]:.1f} Hz, PSD at rolloff: {psd[rolloff_index]:.6e}"
    )

    return float(optimal_cutoff_hz)


@passthrough_empty_audio
def apply_exciter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = None,
    drive: float = None,
    oversample: int = 2,
    mix: float = 1.0,
) -> np.ndarray:
    validate_sample_rate(sr)
    nyquist = sr / 2.0

    if cutoff_hz is None:
        cutoff_hz = calculate_dynamic_cutoff(y, sr, rolloff_percent=0.7)

    cutoff_hz = float(np.clip(cutoff_hz, 1.0, nyquist * 0.99))
    sos = butter(4, cutoff_hz / nyquist, btype="high", output="sos")
    high_band = sosfilt(sos, y)
    peak_highs = np.max(np.abs(high_band)) if high_band.size else 0.0

    logger.debug(
        f"Extracted high band for exciter. Cutoff: {cutoff_hz:.1f} Hz, High band stats - min: {high_band.min():.6f}, max: {high_band.max():.6f}"
    )

    if drive is None:
        drive = 8.0 / (peak_highs + 1e-6)

        logger.debug(f"Calculated drive for exciter: {drive:.2f}")

    y_up = resample_poly(high_band, up=oversample, down=1, axis=-1)

    logger.debug(
        f"Oversampled high band. Oversample factor: {oversample}, Shape: {y_up.shape}, dtype: {y_up.dtype}, max: {np.max(np.abs(y_up)):.6f}"
    )

    distorted_up = np.tanh(y_up * drive)
    distorted_up *= peak_highs / (np.max(np.abs(distorted_up)) + 1e-6)

    distorted_highs = resample_poly(
        distorted_up, up=1, down=oversample, axis=-1
    )

    if distorted_highs.shape[-1] > high_band.shape[-1]:
        logger.debug(
            f"Trimming distorted highs from {distorted_highs.shape[-1]} to {high_band.shape[-1]} samples"
        )

        distorted_highs = distorted_highs[..., : high_band.shape[-1]]

    pure_air = sosfilt(sos, distorted_highs)

    logger.debug(
        f"Generated pure air signal. Shape: {pure_air.shape}, dtype: {pure_air.dtype}, max: {np.max(np.abs(pure_air)):.6f}"
    )

    y, pure_air = pad_audio(y, pure_air)
    y_enhanced = y + (pure_air * mix)

    logger.debug(
        f"Exciter applied with cutoff {cutoff_hz:.1f} Hz, drive {drive:.2f}, mix {mix:.2f}"
    )
    logger.debug(f"Peak of pure air signal: {np.max(np.abs(pure_air)):.6f}")
    logger.debug(f"Peak of enhanced signal: {np.max(np.abs(y_enhanced)):.6f}")

    return y_enhanced


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
            sos = butter(8, low_cut / nyquist, btype="high", output="sos")
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
        self.bands = MultibandParams._make_bands_from_fcs(
            fcs, freq_min, freq_max
        )

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
        if smoothing_fraction is None:
            smoothing_fraction = self.smoothing_fraction

        smoothed = np.copy(target_db)

        for i, f in enumerate(f_axis):
            bandwidth = f * (2**smoothing_fraction - 2 ** (-smoothing_fraction))

            low_f = f - bandwidth / 2
            high_f = f + bandwidth / 2

            mask = (f_axis >= low_f) & (f_axis <= high_f)

            if np.any(mask):
                smoothed[i] = np.mean(target_db[mask])

        logger.debug(
            f"Smoothed target spectrum. Shape: {smoothed.shape}, dtype: {smoothed.dtype}"
        )
        logger.debug(
            f"Smoothed target spectrum stats - min: {smoothed.min():.2f} dB, max: {smoothed.max():.2f} dB, mean: {smoothed.mean():.2f} dB"
        )

        return smoothed

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

        target_db = self._smooth_target_curve(
            target_db, f_axis, self.smoothing_fraction
        )

        logger.debug(
            f"Smoothed target spectrum. Shape: {target_db.shape}, dtype: {target_db.dtype}"
        )
        logger.debug(
            f"Target spectrum stats - min: {target_db.min():.2f} dB, max: {target_db.max():.2f} dB, mean: {target_db.mean():.2f} dB"
        )

        min_len = min(len(target_db), len(input_db))

        target_db = target_db[:min_len]
        input_db = input_db[:min_len]

        logger.debug(
            f"Padded spectra. Target shape: {target_db.shape}, Input shape: {input_db.shape}"
        )

        mean_target = np.mean(target_db)
        mean_input = np.mean(input_db)
        target_db = target_db - mean_target + mean_input

        correction_db = target_db - input_db
        correction_db = (target_db - input_db) * self.correction_strength

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

    def _multiband_compress(self, y: np.ndarray) -> np.ndarray:

        if hasattr(self, "_profile"):
            start = __import__("time").perf_counter()

        def lr4(x, fc):

            if x.shape[-1] <= 9:
                return x, x

            sr2 = self.sr / 2.0
            sos_l = signal.butter(2, fc / sr2, btype="low", output="sos")
            sos_h = signal.butter(2, fc / sr2, btype="high", output="sos")
            lp = signal.sosfiltfilt(sos_l, x)
            hp = signal.sosfiltfilt(sos_h, x)

            logger.debug(
                f"Applied LR4 crossover at {fc:.1f} Hz. LP max: {np.max(np.abs(lp)):.6f}, HP max: {np.max(np.abs(hp)):.6f}"
            )
            logger.debug(
                f"LP stats - min: {lp.min():.6f}, max: {lp.max():.6f}, mean: {lp.mean():.6f}"
            )
            logger.debug(
                f"HP stats - min: {hp.min():.6f}, max: {hp.max():.6f}, mean: {hp.mean():.6f}"
            )

            return lp, hp

        def compress(
            x,
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            knee_db=3.0,
        ):
            ac = np.exp(-1.0 / (self.sr * attack_ms / 1000.0 + 1e-9))
            rc = np.exp(-1.0 / (self.sr * release_ms / 1000.0 + 1e-9))
            mk = 10.0 ** (makeup_db / 20.0)
            env = _decoupled_envelope(
                20.0 * np.log10(np.abs(x) + 1e-12), ac, rc
            )
            hk = knee_db / 2.0
            gain = np.zeros_like(env)
            above = env > threshold_db + hk
            in_kn = (env > threshold_db - hk) & ~above
            gain[above] = -(env[above] - threshold_db) * (1.0 - 1.0 / ratio)
            ki = env[in_kn] - threshold_db + hk
            gain[in_kn] = -(ki**2) / (2.0 * knee_db) * (1.0 - 1.0 / ratio)

            logger.debug(
                f"Compressor applied. Threshold: {threshold_db:.2f} dB, Ratio: {ratio:.2f}, Attack: {attack_ms:.2f} ms, Release: {release_ms:.2f} ms, Makeup: {makeup_db:.2f} dB, Knee: {knee_db:.2f} dB"
            )
            logger.debug(
                f"Envelope stats - min: {env.min():.2f} dB, max: {env.max():.2f} dB, mean: {env.mean():.2f} dB"
            )
            logger.debug(
                f"Gain stats - min: {gain.min():.2f} dB, max: {gain.max():.2f} dB, mean: {gain.mean():.2f} dB"
            )
            logger.debug(
                f"Input signal stats - min: {x.min():.6f}, max: {x.max():.6f}, mean: {x.mean():.6f}"
            )
            logger.debug(
                f"Output signal stats - min: {(x * 10.0 ** (gain / 20.0) * mk).min():.6f}, max: {(x * 10.0 ** (gain / 20.0) * mk).max():.6f}, mean: {(x * 10.0 ** (gain / 20.0) * mk).mean():.6f}"
            )

            return x * 10.0 ** (gain / 20.0) * mk

        def split_bands(ch, fcs):
            bands = []
            current = ch
            for fc in fcs[:-1]:
                lo, current = lr4(current, fc)
                bands.append(lo)
            bands.append(current)
            return bands

        def process_ch(ch):

            fcs = [b.fc for b in self.bands if b.fc > 0]
            if not fcs:
                return ch

            sorted_bands = sorted(self.bands, key=lambda b: b.fc)
            fcs_sorted = [b.fc for b in sorted_bands if b.fc > 0]
            signal_parts = split_bands(ch, fcs_sorted)

            comps: list[np.ndarray] = []
            for band_cfg, sig in zip(sorted_bands, signal_parts, strict=False):
                thr = band_cfg.base_threshold + band_cfg.makeup_db
                comps.append(
                    compress(
                        sig,
                        thr,
                        band_cfg.ratio,
                        band_cfg.attack_ms,
                        band_cfg.release_ms,
                        band_cfg.makeup_db,
                        knee_db=band_cfg.knee_db,
                    )
                )
            logger.debug(
                f"Processed multiband compression for one channel. Number of bands: {len(comps)}"
            )
            logger.debug(
                f"Band thresholds: {[b.base_threshold for b in sorted_bands]}"
            )
            logger.debug(f"Band ratios: {[b.ratio for b in sorted_bands]}")
            logger.debug(
                f"Band attack times: {[b.attack_ms for b in sorted_bands]} ms"
            )
            logger.debug(
                f"Band release times: {[b.release_ms for b in sorted_bands]} ms"
            )
            logger.debug(
                f"Band makeup gains: {[b.makeup_db for b in sorted_bands]} dB"
            )
            logger.debug(
                f"Band knee values: {[b.knee_db for b in sorted_bands]} dB"
            )
            logger.debug(
                f"First band stats - min: {comps[0].min():.6f}, max: {comps[0].max():.6f}, mean: {comps[0].mean():.6f}"
            )
            logger.debug(
                f"Last band stats - min: {comps[-1].min():.6f}, max: {comps[-1].max():.6f}, mean: {comps[-1].mean():.6f}"
            )
            logger.debug(
                f"Sum of bands stats - min: {np.sum(comps, axis=0).min():.6f}, max: {np.sum(comps, axis=0).max():.6f}, mean: {np.sum(comps, axis=0).mean():.6f}"
            )
            logger.debug(
                f"Original channel stats - min: {ch.min():.6f}, max: {ch.max():.6f}, mean: {ch.mean():.6f}"
            )

            return sum(comps)

        if y.ndim > 1:
            out = np.stack([process_ch(ch) for ch in y])
        else:
            out = process_ch(y)
        if hasattr(self, "_profile"):
            end = __import__("time").perf_counter()
            self._profile("multiband_compress", end - start)
        return out

    def _stereo_widen(self, y: np.ndarray, width: float = 1.1) -> np.ndarray:
        if y.ndim < 2 or y.shape[0] != 2:
            return y
        coef = np.sqrt(2.0 / (1.0 + width**2))
        mid = (y[0] + y[1]) * 0.5
        side = (y[0] - y[1]) * 0.5 * width

        logger.debug(
            f"Applying stereo widening. Width: {width:.2f}, Coefficient: {coef:.4f}"
        )
        logger.debug(
            f"Mid signal stats - min: {mid.min():.6f}, max: {mid.max():.6f}, mean: {mid.mean():.6f}"
        )
        logger.debug(
            f"Side signal stats - min: {side.min():.6f}, max: {side.max():.6f}, mean: {side.mean():.6f}"
        )
        logger.debug(
            f"Output left channel stats - min: {(mid + side).min():.6f}, max: {(mid + side).max():.6f}, mean: {(mid + side).mean():.6f}"
        )
        logger.debug(
            f"Output right channel stats - min: {(mid - side).min():.6f}, max: {(mid - side).max():.6f}, mean: {(mid - side).mean():.6f}"
        )
        logger.debug(
            f"Original left channel stats - min: {y[0].min():.6f}, max: {y[0].max():.6f}, mean: {y[0].mean():.6f}"
        )
        logger.debug(
            f"Original right channel stats - min: {y[1].min():.6f}, max: {y[1].max():.6f}, mean: {y[1].mean():.6f}"
        )
        logger.debug(f"Width factor applied to side signal: {width:.2f}")
        logger.debug(
            f"Energy of mid signal before widening: {np.sum(mid**2):.6f}"
        )
        logger.debug(
            f"Energy of side signal before widening: {np.sum(side**2):.6f}"
        )

        return np.stack([(mid + side) * coef, (mid - side) * coef])

    def _apply_limiter(
        self,
        y: np.ndarray,
        drive_db: float,
        ceil_db: float,
        attack_ms: float = 0.1,
        os_factor: int = 2,
        lookahead_ms: float = 0.5,
        release_ms_min: float = 15.0,
        release_ms_max: float = 35.0,
        soft_clip_ratio: float = 0.0,
        up_beta: float = 18.0,
        down_beta: float = 18.0,
    ) -> np.ndarray:

        if hasattr(self, "_profile"):
            start = __import__("time").perf_counter()

        orig_len = y.shape[-1]
        limit_lin = 10.0 ** (ceil_db / 20.0)
        drive_lin = 10.0 ** (drive_db / 20.0)
        sr_os = self.sr * os_factor

        y_os = signal.resample_poly(
            y, os_factor, 1, axis=-1, window=("kaiser", up_beta)
        )
        y_driven = y_os * drive_lin

        lookahead_samp = max(1, int(lookahead_ms * sr_os / 1000.0))
        head_pad = (*((0, 0),) * (y_driven.ndim - 1), (lookahead_samp, 0))

        y_env = np.pad(
            (
                np.max(np.abs(y_driven), axis=0)
                if y_driven.ndim > 1
                else np.abs(y_driven)
            ),
            (0, lookahead_samp),
        )
        peak_env = maximum_filter1d(y_env, size=lookahead_samp)

        rms_win = max(1, int(sr_os * 0.05))
        rms_env = np.sqrt(maximum_filter1d(y_env**2, size=rms_win))
        crest = peak_env / (rms_env + 1e-12)

        rel_ms = np.clip(
            release_ms_max / (crest + 1e-12), release_ms_min, release_ms_max
        )
        ac = np.exp(-1.0 / (sr_os * attack_ms / 1000.0))
        rc = np.exp(-1.0 / (sr_os * rel_ms / 1000.0))

        peak_smooth = _limiter_smooth_env(peak_env, ac, rc)

        gain = np.where(
            peak_smooth > limit_lin, limit_lin / (peak_smooth + 1e-12), 1.0
        )

        y_delayed = np.pad(y_driven, head_pad)

        y_lim = y_delayed * gain
        y_sat = (
            y_lim * (1.0 - soft_clip_ratio)
            + np.tanh(y_lim / limit_lin) * limit_lin * soft_clip_ratio
        )

        y_down = signal.resample_poly(
            y_sat, 1, os_factor, axis=-1, window=("kaiser", down_beta)
        )

        delay_samp = int(lookahead_ms * self.sr / 1000.0)
        out = y_down[..., delay_samp : delay_samp + orig_len]

        out -= (
            np.mean(out, axis=-1, keepdims=True)
            if out.ndim > 1
            else np.mean(out)
        )

        out = np.clip(out, -limit_lin, limit_lin)

        if hasattr(self, "_profile"):
            self._profile("limiter", __import__("time").perf_counter() - start)

        logger.debug(
            f"Applied limiter. Drive: {drive_db} dB, Ceiling: {ceil_db} dB, Lookahead: {lookahead_ms} ms, Attack: {attack_ms} ms, Release: {release_ms_min}-{release_ms_max} ms, Soft clip ratio: {soft_clip_ratio}, Up beta: {up_beta}, Down beta: {down_beta}"
        )
        logger.debug(
            f"Limiter input stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )
        logger.debug(
            f"Limiter output stats - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}"
        )
        logger.debug(
            f"Limiter gain stats - min: {10.0 * np.log10(gain.min() + 1e-12):.2f} dB, max: {10.0 * np.log10(gain.max() + 1e-12):.2f} dB, mean: {10.0 * np.log10(gain.mean() + 1e-12):.2f} dB"
        )
        logger.debug(
            f"Limiter peak envelope stats - min: {peak_env.min():.6f}, max: {peak_env.max():.6f}, mean: {peak_env.mean():.6f}"
        )
        logger.debug(
            f"Limiter RMS envelope stats - min: {rms_env.min():.6f}, max: {rms_env.max():.6f}, mean: {rms_env.mean():.6f}"
        )
        logger.debug(
            f"Limiter crest stats - min: {crest.min():.6f}, max: {crest.max():.6f}, mean: {crest.mean():.6f}"
        )
        logger.debug(
            f"Limiter smoothed peak envelope stats - min: {peak_smooth.min():.6f}, max: {peak_smooth.max():.6f}, mean: {peak_smooth.mean():.6f}"
        )
        logger.debug(
            f"Limiter delayed signal stats - min: {y_delayed.min():.6f}, max: {y_delayed.max():.6f}, mean: {y_delayed.mean():.6f}"
        )
        logger.debug(
            f"Limiter driven signal stats - min: {y_driven.min():.6f}, max: {y_driven.max():.6f}, mean: {y_driven.mean():.6f}"
        )

        return out

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

    def _apply_lufs(self, y: np.ndarray, target_lufs: float) -> np.ndarray:
        if hasattr(self, "_profile"):
            start = __import__("time").perf_counter()

        b1, a1 = (
            [1.5309096471, -2.6511690392, 1.1691662955],
            [1.0, -1.6906592931, 0.7324805897],
        )
        b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.9900474541, 0.9900722501]

        y_filt = signal.lfilter(b1, a1, y, axis=-1)
        y_filt = signal.lfilter(b2, a2, y_filt, axis=-1)

        win_size = int(self.sr * 0.4)
        hop_size = max(1, int(self.sr * 0.1))

        if y_filt.ndim > 1:
            y_sq = np.sum(y_filt**2, axis=0)
            n_samples = y_filt.shape[1]
        else:
            y_sq = y_filt**2
            n_samples = len(y_sq)

        if n_samples < win_size:
            return y

        shape = ((n_samples - win_size) // hop_size + 1, win_size)
        strides = (y_sq.strides[0] * hop_size, y_sq.strides[0])
        y_strided = np.lib.stride_tricks.as_strided(
            y_sq, shape=shape, strides=strides
        )
        ms = np.mean(y_strided, axis=-1)

        abs_threshold = 10.0 ** (-70.0 / 10.0)
        ms_gated = ms[ms > abs_threshold]

        if len(ms_gated) == 0:
            return y

        gamma_rel = 0.1 * np.mean(ms_gated)
        ms_final = ms_gated[ms_gated > gamma_rel]

        loudness_ms = (
            np.mean(ms_final) if len(ms_final) > 0 else np.mean(ms_gated)
        )

        current_lufs = -0.691 + 10.0 * np.log10(np.maximum(loudness_ms, 1e-12))

        gain_lin = 10.0 ** ((target_lufs - current_lufs) / 20.0)
        out = y * gain_lin

        if hasattr(self, "_profile"):
            self._profile("lufs", __import__("time").perf_counter() - start)

        logger.debug(
            f"Applied LUFS normalization. Target LUFS: {target_lufs:.2f}, Current LUFS: {current_lufs:.2f}, Gain applied: {gain_lin:.4f} ({20.0 * np.log10(gain_lin):.2f} dB)"
        )
        logger.debug(
            f"Input signal stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )
        logger.debug(
            f"Output signal stats - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}"
        )
        logger.debug(
            f"Measured loudness stats - min: {loudness_ms:.6f}, max: {loudness_ms:.6f}, mean: {loudness_ms:.6f}"
        )
        logger.debug(
            f"Number of frames used for loudness measurement: {len(ms_final)}"
        )
        logger.debug(
            f"Absolute threshold for gating: {abs_threshold:.6e}, Relative threshold for gating: {gamma_rel:.6e}"
        )
        logger.debug(
            f"LUFS normalization applied with biquad filters. Filter 1 - b: {b1}, a: {a1}, Filter 2 - b: {b2}, a: {a2}"
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

        orig_len = y.shape[-1]

        logger.debug(
            f"Starting processing with input shape: {y.shape}, dtype: {y.dtype}, sample rate: {sr if sr is not None else self.sr} Hz"
        )

        if sr is not None:
            self.sr = validate_sample_rate(sr)

        y_resamp = y

        if self.sr != self.resampling_target:
            logger.debug(
                f"Resampling to {self.resampling_target} Hz for processing..."
            )

            y_resamp = resample(y_resamp, self.sr, self.resampling_target)
            self.sr = self.resampling_target

        logger.debug(
            f"Resampled shape: {y_resamp.shape}, dtype: {y_resamp.dtype}, sample rate: {self.sr} Hz"
        )

        y_stereo = stereo(y_resamp)

        y_exciter = apply_exciter(y_stereo, self.sr)

        logger.debug(
            f"Applied exciter. Shape: {y_exciter.shape}, dtype: {y_exciter.dtype}, sample rate: {self.sr} Hz"
        )

        y_mbc = self._multiband_compress(y_exciter)

        logger.debug(
            f"Applied multiband compression. Shape: {y_mbc.shape}, dtype: {y_mbc.dtype}, sample rate: {self.sr} Hz"
        )

        self.update_bands()

        logger.debug(
            f"Updated multiband compressor bands based on current slope settings. Bands: {[(b.fc, b.base_threshold, b.ratio) for b in self.bands]}"
        )

        p = self._phase(y_mbc, self.phase_type)

        y_stereo2 = stereo(p)

        y_widen = self._stereo_widen(y_stereo2)

        logger.debug(
            f"Applied stereo widening. Shape: {y_widen.shape}, dtype: {y_widen.dtype}, sample rate: {self.sr} Hz"
        )

        y_cut = freq_cut(
            y_widen, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        y_lufs = self._apply_lufs(y_cut, self.target_lufs)

        logger.debug(
            f"Applied LUFS normalization. Shape: {y_lufs.shape}, dtype: {y_lufs.dtype}, sample rate: {self.sr} Hz"
        )

        y_limiter = self._apply_limiter(
            y_lufs,
            drive_db=self.limiter_defaults.drive_db,
            ceil_db=self.limiter_defaults.ceil_db,
        )

        logger.debug(
            f"Applied limiter with drive {self.limiter_defaults.drive_db} dB and ceiling {self.limiter_defaults.ceil_db} dB. Shape: {y_limiter.shape}, dtype: {y_limiter.dtype}, sample rate: {self.sr} Hz"
        )

        y_lufs2 = self._apply_lufs(y_limiter, self.target_lufs)

        y_cut2 = freq_cut(
            y_lufs2, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        lin_amp = 10 ** (self.limiter_defaults.ceil_db / 20.0)

        logger.debug(
            f"Applied final LUFS normalization and frequency cut. Shape: {y_cut2.shape}, dtype: {y_cut2.dtype}, sample rate: {self.sr} Hz"
        )

        y_clip = np.clip(y_cut2, -lin_amp, lin_amp)

        logger.debug(
            f"Applied final clipping to ceiling of {self.limiter_defaults.ceil_db} dB. Shape: {y_clip.shape}, dtype: {y_clip.dtype}, sample rate: {self.sr} Hz"
        )

        if y_clip.shape[-1] != orig_len:
            if y_clip.shape[-1] > orig_len:
                logger.debug(
                    f"Output is longer than original by {y_clip.shape[-1] - orig_len} samples, trimming to original length."
                )
                y_clip = y_clip[..., :orig_len]
            else:
                logger.debug(
                    f"Output is shorter than original by {orig_len - y_clip.shape[-1]} samples, padding with zeros to original length."
                )
                y_clip = np.pad(
                    y_clip,
                    (
                        *((0, 0),) * (y_clip.ndim - 1),
                        (0, orig_len - y_clip.shape[-1]),
                    ),
                )

        logger.debug(f"Final output peak: {np.max(np.abs(y_clip)):.6f}")
        logger.debug(
            f"Final output spectrum (dB): {10.0 * np.log10(np.maximum(np.mean(y_clip**2, axis=-1), 1e-12))}"
        )
        logger.debug(
            f"Final output shape: {y_clip.shape}, dtype: {y_clip.dtype}"
        )
        logger.debug(
            f"Limiter settings - Drive: {self.limiter_defaults.drive_db} dB, Ceiling: {self.limiter_defaults.ceil_db} dB"
        )
        logger.debug(
            f"Sample rate: {self.sr} Hz, Original length: {orig_len} samples"
        )
        logger.debug(
            f"Multiband compression settings: {[(b.fc, b.base_threshold, b.ratio, b.attack_ms, b.release_ms, b.makeup_db) for b in self.bands]}"
        )
        logger.debug(
            f"Phase correction settings: Slope dB/octave: {self.slope_db}, Slope reference frequency: {self.slope_hz} Hz, Smoothing fraction: {self.smoothing_fraction}"
        )
        logger.debug(
            f"Frequency cut settings: Low cut: {self.low_cut} Hz, High cut: {self.high_cut} Hz"
        )
        logger.debug(f"Target LUFS: {self.target_lufs} LUFS")
        logger.debug(
            f"Output stats - Mean: {np.mean(y_clip):.6e}, Std: {np.std(y_clip):.6e}, Max: {np.max(y_clip):.6f}, Min: {np.min(y_clip):.6f}"
        )
        logger.debug(
            f"Output clipping - Samples at ceiling: {(y_clip >= lin_amp).sum()}, Samples at floor: {(y_clip <= -lin_amp).sum()}"
        )
        logger.debug(
            f"Output headroom: {10.0 * np.log10(lin_amp / (np.max(np.abs(y_clip)) + 1e-12)):.2f} dB"
        )
        logger.debug(
            f"Output spectrum after final processing (dB): {10.0 * np.log10(np.maximum(np.mean(y_clip**2, axis=-1), 1e-12))}"
        )
        logger.debug(f"Output shape: {y_clip.shape}, dtype: {y_clip.dtype}")
        logger.debug("Processing steps completed successfully without errors.")

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
