from typing import Literal

import librosa
import numpy as np

from .validation import (
    passthrough_empty_audio,
    reject_empty_audio,
    validate_effect_integer,
    validate_effect_number,
    validate_n_fft,
    validate_sample_rate,
    validate_spectral_gate_threshold,
)


class Distortion:
    def __init__(self) -> None:
        pass

    @passthrough_empty_audio
    def soft_clip(
        self, audio_data: np.ndarray, drive_db: float = 10
    ) -> np.ndarray:
        validate_effect_number(drive_db, "Drive", -60.0, 60.0)
        drive_linear = 10 ** (drive_db / 20.0)
        return np.tanh(audio_data * drive_linear)

    @passthrough_empty_audio
    def hard_clip(
        self, audio_data: np.ndarray, threshold: float = 0.8
    ) -> np.ndarray:
        validate_effect_number(threshold, "Clip threshold", 0.0, 1.0)
        return np.clip(audio_data, -threshold, threshold)

    @passthrough_empty_audio
    def waveshaper(
        self,
        audio_data: np.ndarray,
        curve: Literal["sigmoid", "cubic"] = "sigmoid",
    ) -> np.ndarray:
        if curve == "sigmoid":
            shaped = 2 / (1 + np.exp(-5 * audio_data)) - 1
            return np.clip(shaped, -1.0, 1.0)
        elif curve == "cubic":
            return audio_data - (audio_data**3) / 3
        else:
            return audio_data

    @passthrough_empty_audio
    def bit_crusher(
        self,
        audio_data: np.ndarray,
        bits: int = 8,
        sample_rate_reduction: float = 1.0,
    ) -> np.ndarray:
        bits = validate_effect_integer(bits, "Bit depth", 1, 24)
        validate_effect_number(
            sample_rate_reduction, "Sample rate reduction", 1.0, 1000.0
        )
        levels = 2**bits
        crushed = np.round(audio_data * levels) / levels

        if sample_rate_reduction > 1.0:
            downsample_factor = int(sample_rate_reduction)
            indices = np.arange(0, crushed.shape[-1], downsample_factor)
            downsampled = crushed[..., indices]
            crushed = np.repeat(downsampled, downsample_factor, axis=-1)[
                ..., : audio_data.shape[-1]
            ]

        return crushed

    @passthrough_empty_audio
    def fuzz(
        self, audio_data: np.ndarray, gain_db: float = 20, texture: float = 0.5
    ) -> np.ndarray:
        validate_effect_number(gain_db, "Gain", -60.0, 60.0)
        validate_effect_number(texture, "Texture", 0.0, 10.0)
        gain_linear = 10 ** (gain_db / 20.0)
        driven = audio_data * gain_linear

        fuzzed = np.sign(driven) * (1 - np.exp(-abs(driven) * texture))
        return fuzzed


class Saturation:
    def __init__(self) -> None:
        pass

    @passthrough_empty_audio
    def tape_saturation(
        self, audio_data: np.ndarray, drive: float = 1.5
    ) -> np.ndarray:
        validate_effect_number(drive, "Drive", 0.0, 20.0)
        x = audio_data * drive
        return x / (1 + abs(x) ** 1.5) ** (1 / 1.5)

    @passthrough_empty_audio
    def tube_saturation(
        self, audio_data: np.ndarray, drive: float = 2.0
    ) -> np.ndarray:
        validate_effect_number(drive, "Drive", 0.0, 20.0)
        x = audio_data * drive
        return np.sign(x) * (1 - np.exp(-abs(x)))

    @passthrough_empty_audio
    def transistor_saturation(
        self, audio_data: np.ndarray, drive: float = 1.5
    ) -> np.ndarray:
        validate_effect_number(drive, "Drive", 0.0, 20.0)
        x = audio_data * drive
        return 2 / (1 + np.exp(-2 * x)) - 1

    @passthrough_empty_audio
    def harmonic_exciter(
        self, audio_data: np.ndarray, amount: float = 0.3, harmonic: int = 2
    ) -> np.ndarray:
        validate_effect_number(amount, "Amount", 0.0, 5.0)
        harmonic = validate_effect_integer(harmonic, "Harmonic", 1, 16)
        excited = audio_data + amount * np.sin(
            harmonic * np.arcsin(np.clip(audio_data, -1, 1))
        )
        return np.clip(excited, -1, 1)


class NoiseGate:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = validate_sample_rate(sample_rate)

    @passthrough_empty_audio
    def process(
        self,
        audio_data: np.ndarray,
        threshold_db: float = -40,
        attack_ms: float = 5,
        release_ms: float = 100,
    ) -> np.ndarray:
        validate_effect_number(threshold_db, "Threshold", -120.0, 0.0)
        validate_effect_number(attack_ms, "Attack time", 0.0, 10000.0)
        validate_effect_number(release_ms, "Release time", 0.0, 10000.0)
        threshold_linear = 10 ** (threshold_db / 20.0)
        attack_samples = int(self.sr * attack_ms / 1000.0)
        release_samples = int(self.sr * release_ms / 1000.0)

        attack_coef = np.exp(-1.0 / attack_samples) if attack_samples > 0 else 0
        release_coef = (
            np.exp(-1.0 / release_samples) if release_samples > 0 else 0
        )

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        output = np.zeros_like(audio_data)

        for ch in range(audio_data.shape[0]):
            envelope = 0.0
            gate_envelope = 1.0

            for i in range(audio_data.shape[1]):
                input_level = abs(audio_data[ch, i])

                if input_level > envelope:
                    envelope = (
                        attack_coef * envelope + (1 - attack_coef) * input_level
                    )
                else:
                    envelope = (
                        release_coef * envelope
                        + (1 - release_coef) * input_level
                    )

                if envelope > threshold_linear:
                    target_gain = 1.0
                else:
                    target_gain = 0.0

                if target_gain > gate_envelope:
                    gate_envelope = (
                        attack_coef * gate_envelope
                        + (1 - attack_coef) * target_gain
                    )
                else:
                    gate_envelope = (
                        release_coef * gate_envelope
                        + (1 - release_coef) * target_gain
                    )

                output[ch, i] = audio_data[ch, i] * gate_envelope

        if output.shape[0] == 1:
            return output[0]
        return output


class NoiseReduction:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = validate_sample_rate(sample_rate)
        self.noise_profile: np.ndarray | None = None

    @reject_empty_audio
    def learn_noise_profile(
        self, noise_sample: np.ndarray, n_fft: int = 2048
    ) -> None:
        validate_n_fft(n_fft)
        noise_stft = librosa.stft(noise_sample, n_fft=n_fft)
        self.noise_profile = np.mean(np.abs(noise_stft), axis=1)

    @passthrough_empty_audio
    def process(
        self,
        audio_data: np.ndarray,
        reduction_amount: float = 0.8,
        n_fft: int = 2048,
        spectral_gate_db: float | None = None,
    ) -> np.ndarray:
        validate_n_fft(n_fft)
        validate_effect_number(reduction_amount, "Reduction amount", 0.0, 1.0)
        if spectral_gate_db is not None:
            validate_spectral_gate_threshold(spectral_gate_db)

        if self.noise_profile is None and spectral_gate_db is None:
            return audio_data

        stft = librosa.stft(audio_data, n_fft=n_fft)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        if self.noise_profile is not None:
            noise_profile_expanded = self.noise_profile[:, np.newaxis]
            magnitude = np.maximum(
                magnitude - noise_profile_expanded * reduction_amount, 0
            )

        if spectral_gate_db is not None:
            magnitude_db = 20.0 * np.log10(magnitude + 1e-10)
            magnitude = np.where(
                magnitude_db < spectral_gate_db, 0.0, magnitude
            )

        reduced_stft = magnitude * np.exp(1j * phase)
        reduced_audio = librosa.istft(reduced_stft, length=len(audio_data))

        return reduced_audio
