import librosa
import numpy as np

from korshmer.utils.tempo import scalar_tempo

from .validation import validate_audio_not_empty


class GenreMoodDetector:
    def __init__(self):
        self.genre_profiles = self._build_genre_profiles()

    def _build_genre_profiles(self) -> dict:
        profiles = {
            "rock": {
                "tempo_range": (100, 140),
                "spectral_centroid_range": (1500, 4000),
                "energy": 0.7,
                "percussive_ratio": 0.6,
            },
            "pop": {
                "tempo_range": (100, 130),
                "spectral_centroid_range": (1000, 3500),
                "energy": 0.6,
                "percussive_ratio": 0.5,
            },
            "jazz": {
                "tempo_range": (80, 200),
                "spectral_centroid_range": (800, 3000),
                "energy": 0.4,
                "percussive_ratio": 0.3,
            },
            "classical": {
                "tempo_range": (60, 120),
                "spectral_centroid_range": (500, 2500),
                "energy": 0.3,
                "percussive_ratio": 0.2,
            },
            "electronic": {
                "tempo_range": (120, 150),
                "spectral_centroid_range": (2000, 6000),
                "energy": 0.8,
                "percussive_ratio": 0.7,
            },
            "hip_hop": {
                "tempo_range": (80, 110),
                "spectral_centroid_range": (1000, 4000),
                "energy": 0.7,
                "percussive_ratio": 0.8,
            },
            "metal": {
                "tempo_range": (120, 180),
                "spectral_centroid_range": (2000, 5000),
                "energy": 0.9,
                "percussive_ratio": 0.7,
            },
            "folk": {
                "tempo_range": (90, 120),
                "spectral_centroid_range": (800, 2500),
                "energy": 0.4,
                "percussive_ratio": 0.3,
            },
            "blues": {
                "tempo_range": (60, 100),
                "spectral_centroid_range": (700, 2000),
                "energy": 0.5,
                "percussive_ratio": 0.4,
            },
            "reggae": {
                "tempo_range": (80, 110),
                "spectral_centroid_range": (1000, 3000),
                "energy": 0.6,
                "percussive_ratio": 0.6,
            },
        }
        return profiles

    def detect_genre(self, audio_path: str) -> dict:
        audio, sr = librosa.load(audio_path, sr=None, duration=30)
        validate_audio_not_empty(audio)

        features = self._extract_genre_features(audio, sr)

        scores = {}
        for genre, profile in self.genre_profiles.items():
            score = self._calculate_genre_similarity(features, profile)
            scores[genre] = score

        detected_genre = max(scores, key=scores.get)
        confidence = scores[detected_genre]

        top_3_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        return {
            "genre": detected_genre,
            "confidence": float(confidence),
            "top_genres": [(g, float(s)) for g, s in top_3_genres],
            "features": features,
        }

    def _extract_genre_features(self, audio: np.ndarray, sr: int) -> dict:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        avg_centroid = np.mean(spectral_centroids)

        rms = librosa.feature.rms(y=audio)[0]
        energy = np.mean(rms)

        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        total_energy = harmonic_energy + percussive_energy

        if total_energy > 0:
            percussive_ratio = percussive_energy / total_energy
        else:
            percussive_ratio = 0.0

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)

        return {
            "tempo": scalar_tempo(tempo),
            "spectral_centroid": float(avg_centroid),
            "energy": float(energy),
            "percussive_ratio": float(percussive_ratio),
            "mfcc": mfcc_mean.tolist(),
            "chroma": chroma_mean.tolist(),
            "spectral_contrast": contrast_mean.tolist(),
        }

    def _calculate_genre_similarity(
        self, features: dict, profile: dict
    ) -> float:
        score = 0.0

        tempo = features["tempo"]
        if profile["tempo_range"][0] <= tempo <= profile["tempo_range"][1]:
            score += 0.3
        else:
            diff = min(
                abs(tempo - profile["tempo_range"][0]),
                abs(tempo - profile["tempo_range"][1]),
            )
            score += max(0, 0.3 - diff / 100.0)

        centroid = features["spectral_centroid"]
        if (
            profile["spectral_centroid_range"][0]
            <= centroid
            <= profile["spectral_centroid_range"][1]
        ):
            score += 0.3

        energy_diff = abs(features["energy"] - profile["energy"])
        score += max(0, 0.2 - energy_diff)

        percussive_diff = abs(
            features["percussive_ratio"] - profile["percussive_ratio"]
        )
        score += max(0, 0.2 - percussive_diff)

        return score

    def detect_mood(self, audio_path: str) -> dict:
        audio, sr = librosa.load(audio_path, sr=None, duration=30)
        validate_audio_not_empty(audio)

        features = self._extract_mood_features(audio, sr)

        arousal = self._calculate_arousal(features)
        valence = self._calculate_valence(features)

        mood = self._map_to_mood(arousal, valence)

        return {
            "mood": mood,
            "arousal": float(arousal),
            "valence": float(valence),
            "energy_level": features["energy_level"],
            "tempo": features["tempo"],
        }

    def _extract_mood_features(self, audio: np.ndarray, sr: int) -> dict:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        rms = librosa.feature.rms(y=audio)[0]
        energy = np.mean(rms)

        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_std = np.std(chroma)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        key, mode = self._estimate_key_mode(audio, sr)

        return {
            "tempo": scalar_tempo(tempo),
            "energy": float(energy),
            "spectral_centroid": float(spectral_centroid),
            "zcr": float(zcr),
            "chroma_std": float(chroma_std),
            "mfcc_mean": mfcc_mean,
            "key": key,
            "mode": mode,
            "energy_level": "high"
            if energy > 0.15
            else "medium"
            if energy > 0.05
            else "low",
        }

    def _calculate_arousal(self, features: dict) -> float:
        arousal = 0.0

        if features["tempo"] > 120:
            arousal += 0.4
        elif features["tempo"] > 90:
            arousal += 0.2

        if features["energy"] > 0.15:
            arousal += 0.3
        elif features["energy"] > 0.05:
            arousal += 0.15

        if features["spectral_centroid"] > 2500:
            arousal += 0.2
        elif features["spectral_centroid"] > 1500:
            arousal += 0.1

        if features["zcr"] > 0.1:
            arousal += 0.1

        return min(arousal, 1.0)

    def _calculate_valence(self, features: dict) -> float:
        valence = 0.5

        if features["mode"] == "major":
            valence += 0.3
        else:
            valence -= 0.2

        if features["tempo"] > 100 and features["energy"] > 0.1:
            valence += 0.2

        if features["spectral_centroid"] > 2000:
            valence += 0.1

        return np.clip(valence, 0.0, 1.0)

    def _map_to_mood(self, arousal: float, valence: float) -> str:
        if arousal > 0.6 and valence > 0.6:
            return "energetic"
        elif arousal > 0.6 and valence < 0.4:
            return "angry"
        elif arousal < 0.4 and valence > 0.6:
            return "calm"
        elif arousal < 0.4 and valence < 0.4:
            return "sad"
        elif valence > 0.6:
            return "happy"
        elif valence < 0.4:
            return "dark"
        else:
            return "neutral"

    def _estimate_key_mode(self, audio: np.ndarray, sr: int) -> tuple[str, str]:
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)

        chroma_avg = np.mean(chroma, axis=1)

        key_names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        key_idx = np.argmax(chroma_avg)
        key = key_names[key_idx]

        major_profile = np.array(
            [
                6.35,
                2.23,
                3.48,
                2.33,
                4.38,
                4.09,
                2.52,
                5.19,
                2.39,
                3.66,
                2.29,
                2.88,
            ]
        )
        minor_profile = np.array(
            [
                6.33,
                2.68,
                3.52,
                5.38,
                2.60,
                3.53,
                2.54,
                4.75,
                3.98,
                2.69,
                3.34,
                3.17,
            ]
        )

        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        chroma_normalized = chroma_avg / (np.sum(chroma_avg) + 1e-10)

        major_correlation = np.corrcoef(
            chroma_normalized, np.roll(major_profile, key_idx)
        )[0, 1]
        minor_correlation = np.corrcoef(
            chroma_normalized, np.roll(minor_profile, key_idx)
        )[0, 1]

        if np.isnan(major_correlation):
            major_correlation = 0.0
        if np.isnan(minor_correlation):
            minor_correlation = 0.0

        mode = "major" if major_correlation > minor_correlation else "minor"

        return key, mode

    def calculate_danceability(self, audio_path: str) -> dict:
        audio, sr = librosa.load(audio_path, sr=None, duration=30)
        validate_audio_not_empty(audio)

        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = scalar_tempo(tempo)

        beat_strength = librosa.onset.onset_strength(y=audio, sr=sr)
        avg_beat_strength = np.mean(beat_strength)

        rms = librosa.feature.rms(y=audio)[0]
        rms_std = np.std(rms)

        regularity = 1.0 / (1.0 + rms_std)

        danceability_score = 0.0

        if 90 <= tempo <= 130:
            danceability_score += 0.4
        elif 80 <= tempo <= 140:
            danceability_score += 0.2

        if avg_beat_strength > 0.5:
            danceability_score += 0.3
        elif avg_beat_strength > 0.3:
            danceability_score += 0.15

        danceability_score += regularity * 0.3

        danceability_score = min(danceability_score, 1.0)

        return {
            "danceability": float(danceability_score),
            "tempo": scalar_tempo(tempo),
            "beat_strength": float(avg_beat_strength),
            "regularity": float(regularity),
        }

    def detect_era(self, audio_path: str) -> dict:
        audio, sr = librosa.load(audio_path, sr=None, duration=30)
        validate_audio_not_empty(audio)

        features = self._extract_genre_features(audio, sr)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        avg_rolloff = np.mean(spectral_rolloff)

        if avg_rolloff < 3000 and features["spectral_centroid"] < 1500:
            era = "1950s-1960s"
            decade_score = 0.7
        elif (
            features["percussive_ratio"] < 0.4 and 80 < features["tempo"] < 120
        ):
            era = "1970s"
            decade_score = 0.6
        elif features["energy"] > 0.6 and features["tempo"] > 110:
            era = "1980s"
            decade_score = 0.6
        elif 100 < features["tempo"] < 130 and features["energy"] > 0.5:
            era = "1990s"
            decade_score = 0.6
        elif features["spectral_centroid"] > 2000 and features["energy"] > 0.6:
            era = "2000s"
            decade_score = 0.6
        else:
            era = "2010s-present"
            decade_score = 0.5

        return {
            "era": era,
            "confidence": float(decade_score),
            "features_used": {
                "spectral_rolloff": float(avg_rolloff),
                "spectral_centroid": features["spectral_centroid"],
                "energy": features["energy"],
            },
        }
