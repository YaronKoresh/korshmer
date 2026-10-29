import librosa
import numpy as np

from .validation import (
    MAX_N_FFT,
    reject_empty_audio,
    validate_effect_integer,
    validate_effect_number,
    validate_hop_length,
    validate_n_fft,
    validate_sample_rate,
)


class AudioMetering:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = validate_sample_rate(sample_rate)

    @reject_empty_audio
    def peak_level(self, audio_data: np.ndarray) -> float:
        return 20 * np.log10(np.max(np.abs(audio_data)) + 1e-10)

    @reject_empty_audio
    def rms_level(self, audio_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio_data**2))
        return 20 * np.log10(rms + 1e-10)

    @reject_empty_audio
    def lufs(self, audio_data: np.ndarray) -> float:
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(self.sr)
            loudness = meter.integrated_loudness(
                audio_data.T if audio_data.ndim > 1 else audio_data
            )
            return loudness
        except ImportError:
            return self.rms_level(audio_data)

    @reject_empty_audio
    def true_peak(self, audio_data: np.ndarray) -> float:
        upsampled = librosa.resample(
            audio_data, orig_sr=self.sr, target_sr=self.sr * 4
        )
        peak = np.max(np.abs(upsampled))
        return 20 * np.log10(peak + 1e-10)

    @reject_empty_audio
    def crest_factor(self, audio_data: np.ndarray) -> float:
        peak = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(audio_data**2))
        return 20 * np.log10(peak / (rms + 1e-10))

    @reject_empty_audio
    def dynamic_range(
        self, audio_data: np.ndarray, percentile: int = 95
    ) -> float:
        validate_effect_number(percentile, "percentile", 0, 100)
        sorted_levels = np.sort(np.abs(audio_data), axis=None)
        last = len(sorted_levels) - 1
        peak_level = sorted_levels[
            min(int(len(sorted_levels) * percentile / 100), last)
        ]
        noise_floor = sorted_levels[min(int(len(sorted_levels) * 0.1), last)]
        return 20 * np.log10(peak_level / (noise_floor + 1e-10))

    @reject_empty_audio
    def stereo_correlation(self, audio_data: np.ndarray) -> float:
        if audio_data.ndim != 2 or audio_data.shape[0] != 2:
            return 1.0

        left = audio_data[0]
        right = audio_data[1]

        if np.ptp(left) == 0 or np.ptp(right) == 0:
            both_constant = np.ptp(left) == 0 and np.ptp(right) == 0
            return 1.0 if both_constant and left[0] == right[0] else 0.0

        correlation = np.corrcoef(left, right)[0, 1]
        return correlation

    @reject_empty_audio
    def spectrum_analysis(
        self, audio_data: np.ndarray, n_fft: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        validate_hop_length(hop_length, n_fft)
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        return magnitude_db

    @reject_empty_audio
    def spectral_centroid(
        self, audio_data: np.ndarray, n_fft: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        validate_hop_length(hop_length, n_fft)
        centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )
        return centroid[0]

    @reject_empty_audio
    def zero_crossing_rate(
        self,
        audio_data: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        frame_length = validate_effect_integer(
            frame_length, "frame_length", 1, MAX_N_FFT
        )
        validate_hop_length(hop_length, frame_length)
        zcr = librosa.feature.zero_crossing_rate(
            audio_data, frame_length=frame_length, hop_length=hop_length
        )
        return zcr[0]


class Analyzer:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = validate_sample_rate(sample_rate)
        self.metering = AudioMetering(sample_rate)

    @reject_empty_audio
    def analyze_audio(
        self, audio_data: np.ndarray
    ) -> dict[str, float | np.ndarray]:
        analysis = {
            "peak_db": self.metering.peak_level(audio_data),
            "rms_db": self.metering.rms_level(audio_data),
            "crest_factor_db": self.metering.crest_factor(audio_data),
            "dynamic_range_db": self.metering.dynamic_range(audio_data),
        }

        if audio_data.ndim == 2 and audio_data.shape[0] == 2:
            analysis["stereo_correlation"] = self.metering.stereo_correlation(
                audio_data
            )

        return analysis

    @reject_empty_audio
    def frequency_analysis(
        self, audio_data: np.ndarray, n_fft: int = 2048
    ) -> dict[str, float | np.ndarray]:
        validate_n_fft(n_fft)
        spectrum = self.metering.spectrum_analysis(audio_data, n_fft=n_fft)
        centroid = self.metering.spectral_centroid(audio_data, n_fft=n_fft)

        return {"spectrum": spectrum, "spectral_centroid_hz": np.mean(centroid)}
