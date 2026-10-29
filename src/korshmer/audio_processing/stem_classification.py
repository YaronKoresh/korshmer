import librosa
import numpy as np

from korshmer.utils.tempo import scalar_tempo

from .validation import (
    passthrough_empty_audio,
    reject_empty_audio,
    validate_audio_finite,
    validate_pitch_shift_semitones,
    validate_sample_rate,
)


class StemClassifier:
    def __init__(self):
        self.instrument_profiles = self._build_instrument_profiles()

    def _build_instrument_profiles(self) -> dict:
        profiles = {
            "vocals": {
                "freq_range": (80, 1200),
                "spectral_centroid_range": (500, 3000),
                "has_vibrato": True,
                "harmonicity": 0.7,
            },
            "drums": {
                "freq_range": (30, 8000),
                "spectral_centroid_range": (2000, 8000),
                "percussive_ratio": 0.8,
                "harmonicity": 0.2,
            },
            "bass": {
                "freq_range": (30, 400),
                "spectral_centroid_range": (80, 300),
                "has_vibrato": False,
                "harmonicity": 0.6,
            },
            "guitar": {
                "freq_range": (80, 1500),
                "spectral_centroid_range": (400, 2000),
                "has_vibrato": True,
                "harmonicity": 0.7,
            },
            "piano": {
                "freq_range": (27, 4200),
                "spectral_centroid_range": (200, 3000),
                "has_vibrato": False,
                "harmonicity": 0.8,
            },
            "synth": {
                "freq_range": (30, 8000),
                "spectral_centroid_range": (500, 5000),
                "has_vibrato": False,
                "harmonicity": 0.5,
            },
            "strings": {
                "freq_range": (100, 3000),
                "spectral_centroid_range": (500, 2500),
                "has_vibrato": True,
                "harmonicity": 0.8,
            },
        }
        return profiles

    @reject_empty_audio
    def classify_stem(self, audio: np.ndarray, sr: int) -> dict:
        validate_sample_rate(sr)
        validate_audio_finite(audio)
        features = self._extract_features(audio, sr)

        scores = {}
        for instrument, profile in self.instrument_profiles.items():
            score = self._calculate_similarity(features, profile)
            scores[instrument] = score

        classified_instrument = max(scores, key=scores.get)
        confidence = scores[classified_instrument]

        return {
            "instrument": classified_instrument,
            "confidence": float(confidence),
            "scores": {k: float(v) for k, v in scores.items()},
        }

    def _extract_features(self, audio: np.ndarray, sr: int) -> dict:
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        avg_centroid = np.mean(spectral_centroids)

        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        total_energy = harmonic_energy + percussive_energy

        if total_energy > 0:
            percussive_ratio = percussive_energy / total_energy
            harmonicity = harmonic_energy / total_energy
        else:
            percussive_ratio = 0.0
            harmonicity = 0.0

        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)

        avg_spectrum = np.mean(magnitude, axis=1)

        energy_by_freq = avg_spectrum * freqs
        total_energy_freq = np.sum(avg_spectrum)

        if total_energy_freq > 0:
            dominant_freq = np.sum(energy_by_freq) / total_energy_freq
        else:
            dominant_freq = 0

        max_freq_idx = np.argmax(avg_spectrum)
        peak_freq = freqs[max_freq_idx]

        has_vibrato = self._detect_vibrato(audio, sr)

        return {
            "spectral_centroid": avg_centroid,
            "percussive_ratio": percussive_ratio,
            "harmonicity": harmonicity,
            "dominant_freq": dominant_freq,
            "peak_freq": peak_freq,
            "has_vibrato": has_vibrato,
        }

    def _calculate_similarity(self, features: dict, profile: dict) -> float:
        score = 0.0

        centroid = features["spectral_centroid"]
        if (
            profile["spectral_centroid_range"][0]
            <= centroid
            <= profile["spectral_centroid_range"][1]
        ):
            score += 0.3

        harmonicity_diff = abs(features["harmonicity"] - profile["harmonicity"])
        score += (1.0 - harmonicity_diff) * 0.3

        if "percussive_ratio" in profile:
            percussive_diff = abs(
                features["percussive_ratio"] - profile["percussive_ratio"]
            )
            score += (1.0 - percussive_diff) * 0.2

        if features["has_vibrato"] == profile.get("has_vibrato", False):
            score += 0.2

        return score

    def _detect_vibrato(self, audio: np.ndarray, sr: int) -> bool:
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )

            if f0 is None or len(f0) == 0:
                return False

            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) < 10:
                return False

            f0_derivative = np.diff(valid_f0)

            vibrato_rate = 4
            vibrato_depth_hz = 10

            crossings = np.sum(np.diff(np.sign(f0_derivative)) != 0)

            max_deviation = np.max(np.abs(f0_derivative))

            has_vibrato = (
                crossings > vibrato_rate and max_deviation > vibrato_depth_hz
            )

            return has_vibrato
        except (ValueError, librosa.ParameterError):
            return False

    @reject_empty_audio
    def detect_playing_technique(
        self, audio: np.ndarray, sr: int, instrument: str
    ) -> list[str]:
        techniques = []

        if instrument in ["strings", "guitar"]:
            if self._detect_pizzicato(audio, sr):
                techniques.append("pizzicato")

            if self._detect_vibrato(audio, sr):
                techniques.append("vibrato")

            if self._detect_tremolo(audio, sr):
                techniques.append("tremolo")

        return techniques

    def _detect_pizzicato(self, audio: np.ndarray, sr: int) -> bool:
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=False
        )

        if len(onset_frames) == 0:
            return False

        onset_strengths = librosa.onset.onset_strength(y=audio, sr=sr)

        avg_strength = np.mean(onset_strengths)

        strong_onsets = onset_strengths[onset_frames] > avg_strength * 1.5

        is_pizzicato = np.sum(strong_onsets) > len(onset_frames) * 0.6

        return is_pizzicato

    def _detect_tremolo(self, audio: np.ndarray, sr: int) -> bool:
        rms = librosa.feature.rms(y=audio)[0]

        rms_diff = np.diff(rms)

        crossings = np.sum(np.diff(np.sign(rms_diff)) != 0)

        expected_tremolo_rate = 8

        has_tremolo = crossings > expected_tremolo_rate

        return has_tremolo


class VocalProcessingAI:
    TEMPO_THRESHOLD_ENERGETIC = 120
    TEMPO_THRESHOLD_CALM = 80
    ENERGY_THRESHOLD_HIGH = 0.1
    ENERGY_THRESHOLD_LOW = 0.05
    SPECTRAL_CENTROID_THRESHOLD = 2000

    def __init__(self):
        self.pitch_correction_strength = 0.8

    @passthrough_empty_audio
    def auto_tune_pitch(
        self,
        audio: np.ndarray,
        sr: int,
        scale: list[int] = None,
        strength: float = 0.8,
    ) -> np.ndarray:
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]

        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )

            if f0 is None or len(f0) == 0:
                return audio

            corrected_f0 = np.copy(f0)

            for i, freq in enumerate(f0):
                if not np.isnan(freq) and voiced_flag[i]:
                    midi_note = librosa.hz_to_midi(freq)

                    pitch_class = int(midi_note) % 12
                    octave = int(midi_note) // 12

                    closest_scale_note = min(
                        scale, key=lambda x: abs((x % 12) - pitch_class)
                    )

                    target_midi = octave * 12 + closest_scale_note

                    corrected_midi = (
                        midi_note * (1 - strength) + target_midi * strength
                    )
                    corrected_f0[i] = librosa.midi_to_hz(corrected_midi)

            mask = (~np.isnan(f0)) & voiced_flag
            if not np.any(mask):
                return audio

            orig_midi = librosa.hz_to_midi(f0[mask])
            corr_midi = librosa.hz_to_midi(corrected_f0[mask])

            semitone_shifts = corr_midi - orig_midi
            if semitone_shifts.size == 0:
                return audio

            global_shift = float(np.median(semitone_shifts))
            if not np.isfinite(global_shift) or abs(global_shift) < 1e-3:
                return audio

            corrected_audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=global_shift
            )

            return corrected_audio

        except (ValueError, librosa.ParameterError):
            return audio

    @passthrough_empty_audio
    def shift_formants(
        self, audio: np.ndarray, sr: int, semitones: float = 0.0
    ) -> np.ndarray:
        validate_pitch_shift_semitones(semitones)
        if semitones == 0:
            return audio

        stretch_factor = 2 ** (-semitones / 12.0)

        pitch_shifted = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=semitones
        )

        formant_shifted = librosa.effects.time_stretch(
            pitch_shifted, rate=stretch_factor
        )

        return formant_shifted

    @passthrough_empty_audio
    def remove_breaths(
        self, audio: np.ndarray, sr: int, threshold_db: float = -40.0
    ) -> np.ndarray:
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]

        db = librosa.amplitude_to_db(rms)

        breath_frames = db < threshold_db

        hop_length = 512
        voice_mask = np.repeat(~breath_frames, hop_length)

        if len(voice_mask) > len(audio):
            voice_mask = voice_mask[: len(audio)]
        elif len(voice_mask) < len(audio):
            voice_mask = np.pad(
                voice_mask, (0, len(audio) - len(voice_mask)), mode="edge"
            )

        cleaned_audio = audio * voice_mask

        return cleaned_audio

    @reject_empty_audio
    def detect_emotion(self, audio: np.ndarray, sr: int) -> dict:
        validate_audio_finite(audio)
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )

        rms_energy = np.mean(librosa.feature.rms(y=audio)[0])

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = scalar_tempo(tempo)

        if (
            tempo > self.TEMPO_THRESHOLD_ENERGETIC
            and rms_energy > self.ENERGY_THRESHOLD_HIGH
        ):
            emotion = "energetic"
            arousal = 0.8
            valence = 0.7
        elif (
            tempo < self.TEMPO_THRESHOLD_CALM
            and spectral_centroid < self.SPECTRAL_CENTROID_THRESHOLD
        ):
            emotion = "calm"
            arousal = 0.2
            valence = 0.6
        elif rms_energy < self.ENERGY_THRESHOLD_LOW:
            emotion = "sad"
            arousal = 0.3
            valence = 0.3
        else:
            emotion = "neutral"
            arousal = 0.5
            valence = 0.5

        return {
            "emotion": emotion,
            "arousal": arousal,
            "valence": valence,
            "tempo": tempo,
            "energy": float(rms_energy),
        }


def classify_stem(audio: np.ndarray, sr: int) -> dict:
    classifier = StemClassifier()
    return classifier.classify_stem(audio, sr)
