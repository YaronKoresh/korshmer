import numpy as np
from scipy import signal

from . import kernels
from .validation import (
    passthrough_empty_audio,
    validate_compression_ratio,
    validate_effect_number,
    validate_sample_rate,
)


class Compressor:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(
        self,
        audio_data,
        threshold_db=-20,
        ratio=4.0,
        attack_ms=10,
        release_ms=100,
        makeup_db=0,
    ) -> np.ndarray:
        validate_compression_ratio(ratio)
        validate_effect_number(threshold_db, "Threshold", -120.0, 0.0)
        validate_effect_number(attack_ms, "Attack time", 0.0, 10000.0)
        validate_effect_number(release_ms, "Release time", 0.0, 10000.0)
        validate_effect_number(makeup_db, "Makeup gain", -60.0, 60.0)
        threshold_linear = 10 ** (threshold_db / 20.0)
        makeup_linear = 10 ** (makeup_db / 20.0)

        attack_samples = int(self.sr * attack_ms / 1000.0)
        release_samples = int(self.sr * release_ms / 1000.0)

        attack_coef = np.exp(-1.0 / attack_samples) if attack_samples > 0 else 0
        release_coef = (
            np.exp(-1.0 / release_samples) if release_samples > 0 else 0
        )

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output: np.ndarray = np.zeros_like(audio_data)

        kernels.compress(
            audio_data,
            output,
            threshold_linear,
            ratio,
            attack_coef,
            release_coef,
            makeup_linear,
        )

        if output.shape[0] == 1:
            result: np.ndarray = output[0]
            return result
        return output


class MultiBandCompressor:
    CROSSOVER_FREQS = (100, 500, 2000, 8000)
    MAX_BANDS = len(CROSSOVER_FREQS) + 1

    def __init__(self, sample_rate=44100, n_bands=4):
        self.sr = validate_sample_rate(sample_rate)
        if (
            isinstance(n_bands, (bool, np.bool_))
            or not isinstance(n_bands, (int, np.integer))
            or not 1 <= n_bands <= self.MAX_BANDS
        ):
            raise ValueError(
                f"Band count must be an integer between 1 and "
                f"{self.MAX_BANDS}, got {n_bands!r}"
            )
        self.n_bands = int(n_bands)
        self.crossover_freqs = list(self.CROSSOVER_FREQS[: n_bands - 1])
        self.compressors = [Compressor(sample_rate) for _ in range(n_bands)]

    def _crossover(self, freq):
        low = signal.butter(2, freq, "low", fs=self.sr, output="sos")
        high = signal.butter(2, freq, "high", fs=self.sr, output="sos")
        return np.vstack([low, low]), np.vstack([high, high])

    def _split_bands(self, audio_data):
        bands = []
        current = audio_data
        crossovers = [self._crossover(freq) for freq in self.crossover_freqs]

        for index, (sos_low, sos_high) in enumerate(crossovers):
            low_band = signal.sosfilt(sos_low, current, axis=-1)
            current = signal.sosfilt(sos_high, current, axis=-1)
            for later_low, later_high in crossovers[index + 1 :]:
                low_band = signal.sosfilt(
                    later_low, low_band, axis=-1
                ) + signal.sosfilt(later_high, low_band, axis=-1)
            bands.append(low_band)

        bands.append(current)
        return bands

    @passthrough_empty_audio
    def process(
        self,
        audio_data,
        threshold_db=-20,
        ratio=4.0,
        attack_ms=10,
        release_ms=100,
    ):
        validate_compression_ratio(ratio)
        bands = self._split_bands(audio_data)

        processed_bands = []
        for i, band in enumerate(bands):
            processed = self.compressors[i].process(
                band, threshold_db, ratio, attack_ms, release_ms
            )
            processed_bands.append(processed)

        output = np.sum(processed_bands, axis=0)
        return output


class Limiter:
    def __init__(self, sample_rate=44100):
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(self, audio_data, threshold_db=-0.3, release_ms=50):
        validate_effect_number(threshold_db, "Threshold", -120.0, 0.0)
        validate_effect_number(release_ms, "Release time", 0.0, 10000.0)
        threshold_linear = 10 ** (threshold_db / 20.0)
        release_samples = int(self.sr * release_ms / 1000.0)
        release_coef = (
            np.exp(-1.0 / release_samples) if release_samples > 0 else 0
        )

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output = np.zeros_like(audio_data)

        kernels.limit(audio_data, output, threshold_linear, release_coef)

        if output.shape[0] == 1:
            return output[0]
        return output
