import librosa
import numpy as np
from scipy import signal

from korshmer.audio.loader import check_audio_file_channels
from korshmer.audio_processing.validation import (
    pan_to_unit,
    passthrough_empty_audio,
    validate_effect_number,
    validate_pan_position,
)


class SmartMixingAssistant:
    def __init__(self):
        self.reference_spectrum = None
        self.target_lufs = -14.0

    def analyze_reference_track(self, audio_path: str) -> dict:
        check_audio_file_channels(audio_path)
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        if audio.ndim == 1:
            audio = np.array([audio, audio])

        features = {
            "spectrum": self._compute_spectrum(audio, sr),
            "stereo_width": self._compute_stereo_width(audio),
            "dynamics": self._compute_dynamics(audio),
            "frequency_balance": self._compute_frequency_balance(audio, sr),
        }

        return features

    def _compute_spectrum(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        stft = librosa.stft(audio_mono, n_fft=4096)
        magnitude = np.abs(stft)
        spectrum = np.mean(magnitude, axis=1)

        return spectrum

    def _compute_stereo_width(self, audio: np.ndarray) -> float:
        if audio.ndim == 1:
            return 0.0

        left = audio[0]
        right = audio[1]

        mid = (left + right) / 2
        side = (left - right) / 2

        mid_energy = np.sum(mid**2)
        side_energy = np.sum(side**2)

        if mid_energy + side_energy > 0:
            width = side_energy / (mid_energy + side_energy)
        else:
            width = 0.0

        return width

    def _compute_dynamics(self, audio: np.ndarray) -> dict:
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        rms = np.sqrt(np.mean(audio_mono**2))
        peak = np.max(np.abs(audio_mono))

        if rms > 0:
            crest_factor = peak / rms
        else:
            crest_factor = 1.0

        return {"rms": rms, "peak": peak, "crest_factor": crest_factor}

    def _compute_frequency_balance(self, audio: np.ndarray, sr: int) -> dict:
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        stft = librosa.stft(audio_mono, n_fft=4096)
        magnitude = np.abs(stft)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "high_mid": (2000, 4000),
            "high": (4000, 8000),
            "air": (8000, 20000),
        }

        balance = {}
        for band_name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if np.any(mask):
                band_energy = np.mean(magnitude[mask])
                balance[band_name] = float(band_energy)
            else:
                balance[band_name] = 0.0

        return balance

    def balance_levels(
        self,
        stems: list[tuple[np.ndarray, int]],
        target_balance: dict | None = None,
    ) -> list[np.ndarray]:
        if target_balance is None:
            target_balance = {
                "vocals": 0.0,
                "drums": -3.0,
                "bass": -6.0,
                "other": -9.0,
            }

        balanced_stems = []
        stem_types = list(target_balance.keys())

        for idx, (audio, _sr) in enumerate(stems):
            if idx < len(stem_types):
                stem_type = stem_types[idx]
            else:
                stem_type = "other"

            gain_db = target_balance.get(stem_type, -9.0)
            gain_linear = 10 ** (gain_db / 20)

            balanced_audio = audio * gain_linear
            balanced_stems.append(balanced_audio)

        return balanced_stems

    def auto_pan_stems(
        self,
        stems: list[tuple[np.ndarray, int]],
        pan_positions: list[float] | None = None,
    ) -> list[np.ndarray]:
        panned_stems = []

        if pan_positions is None:
            pan_positions = [-50, 50, -30, 30, 0]

        validated = [validate_pan_position(p) for p in pan_positions]

        for idx, (audio, _sr) in enumerate(stems):
            if audio.ndim == 1:
                audio = np.array([audio, audio])

            pan = pan_to_unit(validated[idx % len(validated)])

            pan_radians = (pan + 1.0) * (np.pi / 4.0)
            left_gain = np.cos(pan_radians)
            right_gain = np.sin(pan_radians)

            panned = np.zeros_like(audio)
            panned[0] = audio[0] * left_gain
            panned[1] = audio[1] * right_gain

            panned_stems.append(panned)

        return panned_stems

    def match_eq_to_reference(
        self, audio: np.ndarray, sr: int, reference_spectrum: np.ndarray
    ) -> np.ndarray:
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
            stereo = True
        else:
            audio_mono = audio
            stereo = False

        current_spectrum = self._compute_spectrum(audio, sr)

        min_len = min(len(current_spectrum), len(reference_spectrum))
        current_spectrum = current_spectrum[:min_len]
        reference_spectrum = reference_spectrum[:min_len]

        eq_curve = np.zeros(min_len)
        for i in range(min_len):
            if current_spectrum[i] > 1e-10:
                eq_curve[i] = reference_spectrum[i] / current_spectrum[i]
            else:
                eq_curve[i] = 1.0

        eq_curve = np.clip(eq_curve, 0.1, 10.0)

        stft = librosa.stft(audio_mono, n_fft=4096)

        magnitude = np.abs(stft)
        phase = np.angle(stft)

        eq_curve_expanded = eq_curve[:, np.newaxis]

        if magnitude.shape[0] < eq_curve_expanded.shape[0]:
            eq_curve_expanded = eq_curve_expanded[: magnitude.shape[0]]
        else:
            padding = magnitude.shape[0] - eq_curve_expanded.shape[0]
            eq_curve_expanded = np.pad(
                eq_curve_expanded, ((0, padding), (0, 0)), mode="edge"
            )

        magnitude_eq = magnitude * eq_curve_expanded

        stft_eq = magnitude_eq * np.exp(1j * phase)

        audio_eq = librosa.istft(stft_eq)

        if stereo:
            audio_eq = np.array([audio_eq, audio_eq])

        return audio_eq

    @passthrough_empty_audio
    def auto_compress(
        self, audio: np.ndarray, sr: int, track_type: str = "general"
    ) -> np.ndarray:
        compression_settings = {
            "vocals": {
                "threshold": 0.3,
                "ratio": 3.0,
                "attack": 0.005,
                "release": 0.1,
            },
            "drums": {
                "threshold": 0.4,
                "ratio": 4.0,
                "attack": 0.001,
                "release": 0.05,
            },
            "bass": {
                "threshold": 0.35,
                "ratio": 3.5,
                "attack": 0.01,
                "release": 0.15,
            },
            "general": {
                "threshold": 0.4,
                "ratio": 2.5,
                "attack": 0.01,
                "release": 0.1,
            },
        }

        settings = compression_settings.get(
            track_type, compression_settings["general"]
        )

        if audio.ndim == 2:
            compressed = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                compressed[ch] = self._apply_compression(
                    audio[ch],
                    sr,
                    settings["threshold"],
                    settings["ratio"],
                    settings["attack"],
                    settings["release"],
                )
        else:
            compressed = self._apply_compression(
                audio,
                sr,
                settings["threshold"],
                settings["ratio"],
                settings["attack"],
                settings["release"],
            )

        return compressed

    def _apply_compression(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: float,
        ratio: float,
        attack: float,
        release: float,
    ) -> np.ndarray:
        envelope = np.abs(audio)

        attack_samples = max(int(attack * sr), 1)
        release_samples = max(int(release * sr), 1)

        gain_reduction = np.ones_like(envelope)

        if ratio > 1.0:
            over = envelope > threshold
            gain_reduction[over] = (threshold / envelope[over]) ** (
                (ratio - 1.0) / ratio
            )

        alpha_attack = 1.0 - np.exp(-1.0 / attack_samples)
        alpha_release = 1.0 - np.exp(-1.0 / release_samples)

        b_attack = [alpha_attack]
        a_attack = [1.0, -(1.0 - alpha_attack)]
        smoothed_attack = signal.lfilter(b_attack, a_attack, gain_reduction)

        b_release = [alpha_release]
        a_release = [1.0, -(1.0 - alpha_release)]
        smoothed_release = signal.lfilter(b_release, a_release, gain_reduction)

        delta_gain = np.diff(gain_reduction, prepend=gain_reduction[0])
        is_attack = delta_gain < 0

        smoothed_gain = np.where(is_attack, smoothed_attack, smoothed_release)

        compressed = audio * smoothed_gain

        return compressed

    def detect_masking(self, stems: list[tuple[np.ndarray, int]]) -> list[dict]:
        masking_issues = []

        for i in range(len(stems)):
            for j in range(i + 1, len(stems)):
                audio1, sr1 = stems[i]
                audio2, sr2 = stems[j]

                if sr1 != sr2:
                    continue

                spec1 = self._compute_spectrum(audio1, sr1)
                spec2 = self._compute_spectrum(audio2, sr2)

                min_len = min(len(spec1), len(spec2))
                spec1 = spec1[:min_len]
                spec2 = spec2[:min_len]

                correlation = np.corrcoef(spec1, spec2)[0, 1]

                if np.isnan(correlation):
                    continue

                if correlation > 0.7:
                    masking_issues.append(
                        {
                            "stem1": i,
                            "stem2": j,
                            "correlation": correlation,
                            "severity": "high"
                            if correlation > 0.85
                            else "medium",
                        }
                    )

        return masking_issues

    def optimize_stereo_width(
        self, audio: np.ndarray, target_width: float = 0.5
    ) -> np.ndarray:
        validate_effect_number(target_width, "Target width", 0.0, 4.0)
        if audio.ndim == 1:
            return np.array([audio, audio])

        left = audio[0]
        right = audio[1]

        mid = (left + right) / 2
        side = (left - right) / 2

        side_adjusted = side * target_width

        left_out = mid + side_adjusted
        right_out = mid - side_adjusted

        return np.array([left_out, right_out])

    def optimize_dynamic_range(
        self, audio: np.ndarray, sr: int, target_lufs: float = -14.0
    ) -> np.ndarray:
        validate_effect_number(target_lufs, "Target loudness", -70.0, 0.0)
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        rms = np.sqrt(np.mean(audio_mono**2))

        if rms > 0:
            current_lufs = -23 + 20 * np.log10(rms)
        else:
            current_lufs = -100

        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)

        normalized = audio * gain_linear

        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)

        return normalized


class GenreMixingPresets:
    def __init__(self):
        self.presets = {
            "rock": {
                "levels": {
                    "vocals": 0.0,
                    "drums": -3.0,
                    "bass": -6.0,
                    "guitars": -4.0,
                },
                "compression": {
                    "vocals": "medium",
                    "drums": "heavy",
                    "bass": "medium",
                },
                "eq": {"bass_boost": 2.0, "high_shelf": 1.0},
                "reverb": "medium_hall",
            },
            "pop": {
                "levels": {
                    "vocals": 0.0,
                    "drums": -4.0,
                    "bass": -5.0,
                    "synths": -6.0,
                },
                "compression": {
                    "vocals": "heavy",
                    "drums": "medium",
                    "bass": "light",
                },
                "eq": {"bass_boost": 1.0, "high_shelf": 2.0},
                "reverb": "small_room",
            },
            "jazz": {
                "levels": {
                    "vocals": 0.0,
                    "drums": -8.0,
                    "bass": -6.0,
                    "piano": -5.0,
                },
                "compression": {
                    "vocals": "light",
                    "drums": "light",
                    "bass": "light",
                },
                "eq": {"bass_boost": 0.0, "high_shelf": 0.5},
                "reverb": "large_hall",
            },
            "electronic": {
                "levels": {
                    "vocals": -2.0,
                    "drums": 0.0,
                    "bass": -3.0,
                    "synths": -4.0,
                },
                "compression": {
                    "vocals": "medium",
                    "drums": "heavy",
                    "bass": "heavy",
                },
                "eq": {"bass_boost": 3.0, "high_shelf": 2.0},
                "reverb": "small_room",
            },
        }

    def get_preset(self, genre: str) -> dict:
        return self.presets.get(genre.lower(), self.presets["pop"])

    def apply_preset(
        self,
        stems: list[tuple[np.ndarray, int]],
        genre: str,
        mixer: SmartMixingAssistant,
    ) -> list[np.ndarray]:
        preset = self.get_preset(genre)

        balanced = mixer.balance_levels(stems, preset["levels"])

        compressed = []
        for audio, sr in zip(balanced, [s[1] for s in stems], strict=False):
            compressed_audio = mixer.auto_compress(audio, sr, "general")
            compressed.append((compressed_audio, sr))

        final_mix = compressed

        return [audio for audio, _ in final_mix]
