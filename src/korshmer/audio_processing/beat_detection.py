import librosa
import numpy as np

from korshmer.utils.perf_metrics import measure
from korshmer.utils.tempo import MAX_BPM, MIN_BPM, scalar_tempo

from .bar_grid import BarGrid, find_bar_grid
from .validation import (
    validate_audio_not_empty,
    validate_effect_number,
    validate_sample_rate,
    validate_time_signature,
)

MAX_CLICK_TRACK_SECONDS = 3600.0
MIN_VARIATION_WINDOW_SECONDS = 0.1
MAX_VARIATION_WINDOW_SECONDS = 3600.0


class BeatDetector:
    def __init__(self):
        self.tempo_range = (20.0, 300.0)

    @measure("audio.tempo_detection")
    def detect_tempo(
        self, audio_path: str, use_multiple_algorithms: bool = True
    ) -> tuple[float, np.ndarray]:
        audio, sr = librosa.load(audio_path, sr=None)
        validate_audio_not_empty(audio)

        if use_multiple_algorithms:
            tempo1, beats1 = self._librosa_beat_track(audio, sr)
            tempo2 = self._onset_strength_tempo(audio, sr)
            tempo3 = self._autocorrelation_tempo(audio, sr)

            tempos = [
                t
                for t in [tempo1, tempo2, tempo3]
                if self.tempo_range[0] <= t <= self.tempo_range[1]
            ]

            if len(tempos) > 0:
                final_tempo = np.median(tempos)
            else:
                final_tempo = tempo1

            tempo, beats = librosa.beat.beat_track(
                y=audio, sr=sr, start_bpm=final_tempo
            )
        else:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

        beat_times = librosa.frames_to_time(beats, sr=sr)

        return scalar_tempo(tempo), beat_times

    def _librosa_beat_track(
        self, audio: np.ndarray, sr: int
    ) -> tuple[float, np.ndarray]:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        return scalar_tempo(tempo), beats

    def _onset_strength_tempo(self, audio: np.ndarray, sr: int) -> float:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

        avg_tempogram = np.mean(tempogram, axis=1)

        tempo_frequencies = librosa.tempo_frequencies(len(avg_tempogram), sr=sr)
        in_range = (tempo_frequencies >= self.tempo_range[0]) & (
            tempo_frequencies <= self.tempo_range[1]
        )
        if not in_range.any():
            return 120.0

        candidates = np.where(in_range, avg_tempogram, -np.inf)
        return float(tempo_frequencies[np.argmax(candidates)])

    def _autocorrelation_tempo(self, audio: np.ndarray, sr: int) -> float:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

        ac = librosa.autocorrelate(onset_env, max_size=onset_env.shape[0])

        hop_length = 512
        tempo_min_samples = int(60.0 / self.tempo_range[1] * sr / hop_length)
        tempo_max_samples = int(60.0 / self.tempo_range[0] * sr / hop_length)

        tempo_max_samples = min(tempo_max_samples, len(ac))

        if tempo_min_samples >= tempo_max_samples:
            return 120.0

        search_region = ac[tempo_min_samples:tempo_max_samples]

        if len(search_region) == 0:
            return 120.0

        peak_idx = np.argmax(search_region)

        peak_sample = peak_idx + tempo_min_samples
        tempo = 60.0 * sr / (peak_sample * hop_length)

        return float(tempo)

    def detect_downbeats(
        self, audio_path: str, tempo: float | None = None
    ) -> np.ndarray:
        return self.detect_bar_grid(audio_path, tempo).downbeat_times

    def detect_bar_grid(
        self,
        audio_path: str,
        tempo: float | None = None,
        beats_per_bar: int = 4,
    ) -> BarGrid:
        if tempo is not None:
            validate_effect_number(tempo, "Tempo", MIN_BPM, MAX_BPM)
        audio, sr = librosa.load(audio_path, sr=None)
        validate_audio_not_empty(audio)
        if tempo is None:
            tempo, _ = self.detect_tempo(audio_path)
        return find_bar_grid(audio, sr, tempo, beats_per_bar)

    def calculate_beat_confidence(
        self, audio_path: str, beat_times: np.ndarray
    ) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=None)
        validate_audio_not_empty(audio)

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

        confidences = []

        for beat_time in beat_times:
            closest_idx = np.argmin(np.abs(onset_times - beat_time))

            window_size = 5
            start_idx = max(0, closest_idx - window_size)
            end_idx = min(len(onset_env), closest_idx + window_size + 1)

            local_onset = onset_env[start_idx:end_idx]

            if len(local_onset) > 0:
                confidence = onset_env[closest_idx] / (
                    np.mean(local_onset) + 1e-10
                )
                confidences.append(min(confidence, 1.0))
            else:
                confidences.append(0.5)

        return np.array(confidences)

    def detect_tempo_variations(
        self, audio_path: str, window_size: float = 10.0
    ) -> list[dict]:
        validate_effect_number(
            window_size,
            "Window size",
            MIN_VARIATION_WINDOW_SECONDS,
            MAX_VARIATION_WINDOW_SECONDS,
        )
        audio, sr = librosa.load(audio_path, sr=None)
        validate_audio_not_empty(audio)

        window_samples = int(window_size * sr)
        hop_samples = window_samples // 2

        tempo_timeline = []

        for start in range(0, len(audio) - window_samples, hop_samples):
            window_audio = audio[start : start + window_samples]

            try:
                tempo, _ = librosa.beat.beat_track(y=window_audio, sr=sr)

                time_position = start / sr

                tempo_timeline.append(
                    {"time": time_position, "tempo": scalar_tempo(tempo)}
                )
            except (ValueError, librosa.ParameterError):
                continue

        return tempo_timeline

    def detect_swing_shuffle(
        self, audio_path: str, beat_times: np.ndarray
    ) -> dict:
        if len(beat_times) < 3:
            return {"has_swing": False, "swing_ratio": 0.5}

        audio, sr = librosa.load(audio_path, sr=None)

        validate_audio_not_empty(audio)
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        swing_ratios = []

        for i in range(len(beat_times) - 1):
            beat_start = beat_times[i]
            beat_end = beat_times[i + 1]

            onsets_in_beat = onset_times[
                (onset_times >= beat_start) & (onset_times < beat_end)
            ]

            if len(onsets_in_beat) >= 2:
                first_eighth = onsets_in_beat[0] - beat_start
                second_eighth = onsets_in_beat[1] - onsets_in_beat[0]

                total = first_eighth + second_eighth
                if total > 0:
                    ratio = first_eighth / total
                    swing_ratios.append(ratio)

        if len(swing_ratios) > 0:
            avg_ratio = np.mean(swing_ratios)
            has_swing = bool(abs(avg_ratio - 0.5) > 0.05)
        else:
            avg_ratio = 0.5
            has_swing = False

        return {
            "has_swing": has_swing,
            "swing_ratio": float(avg_ratio),
            "swing_amount": float(abs(avg_ratio - 0.5) * 200),
        }

    def generate_click_track(
        self,
        tempo: float,
        duration: float,
        sr: int = 44100,
        time_signature: tuple[int, int] = (4, 4),
    ) -> np.ndarray:
        validate_effect_number(tempo, "Tempo", MIN_BPM, MAX_BPM)
        validate_effect_number(
            duration, "Duration", 0.0, MAX_CLICK_TRACK_SECONDS
        )
        validate_sample_rate(sr)
        time_signature = validate_time_signature(time_signature)
        beat_duration = 60.0 / tempo

        num_beats = int(duration / beat_duration) + 1

        click_track = np.zeros(int(duration * sr))

        for beat_idx in range(num_beats):
            beat_time = beat_idx * beat_duration
            beat_sample = int(beat_time * sr)

            if beat_sample >= len(click_track):
                break

            is_downbeat = beat_idx % time_signature[0] == 0

            if is_downbeat:
                freq = 1000
                click_duration = 0.05
            else:
                freq = 800
                click_duration = 0.03

            click_samples = int(click_duration * sr)
            t = np.linspace(0, click_duration, click_samples)
            click = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)

            end_sample = min(beat_sample + len(click), len(click_track))
            click_track[beat_sample:end_sample] = click[
                : end_sample - beat_sample
            ]

        return click_track


class MetricModulationDetector:
    def __init__(self):
        pass

    def detect_metric_modulation(
        self, audio_path: str, tempo_timeline: list[dict]
    ) -> list[dict]:
        modulations = []

        for i in range(len(tempo_timeline) - 1):
            current_tempo = tempo_timeline[i]["tempo"]
            next_tempo = tempo_timeline[i + 1]["tempo"]

            if not (
                np.isfinite(current_tempo)
                and np.isfinite(next_tempo)
                and current_tempo > 0
                and next_tempo > 0
            ):
                continue

            ratio = next_tempo / current_tempo

            common_ratios = {
                3 / 2: "triplet to duplet",
                2 / 3: "duplet to triplet",
                4 / 3: "quarter to dotted quarter",
                3 / 4: "dotted quarter to quarter",
                2 / 1: "half tempo",
                1 / 2: "double tempo",
            }

            for common_ratio, description in common_ratios.items():
                if abs(ratio - common_ratio) < 0.05:
                    modulations.append(
                        {
                            "time": tempo_timeline[i + 1]["time"],
                            "from_tempo": current_tempo,
                            "to_tempo": next_tempo,
                            "ratio": ratio,
                            "type": description,
                        }
                    )
                    break

        return modulations


class RubatoDetector:
    def __init__(self):
        self.tolerance = 0.1

    def detect_rubato(
        self, beat_times: np.ndarray, expected_tempo: float
    ) -> dict:
        validate_effect_number(expected_tempo, "Tempo", MIN_BPM, MAX_BPM)
        if len(beat_times) < 2:
            return {"has_rubato": False, "rubato_regions": []}

        expected_interval = 60.0 / expected_tempo

        intervals = np.diff(beat_times)

        deviations = np.abs(intervals - expected_interval) / expected_interval

        has_rubato = bool(np.any(deviations > self.tolerance))

        rubato_regions = []
        in_rubato = False
        rubato_start = 0

        for i, deviation in enumerate(deviations):
            if deviation > self.tolerance:
                if not in_rubato:
                    rubato_start = i
                    in_rubato = True
            else:
                if in_rubato:
                    rubato_regions.append(
                        {
                            "start_beat": rubato_start,
                            "end_beat": i,
                            "start_time": beat_times[rubato_start],
                            "end_time": beat_times[i],
                            "max_deviation": float(
                                np.max(deviations[rubato_start:i])
                            ),
                        }
                    )
                    in_rubato = False

        if in_rubato:
            end = len(deviations)
            rubato_regions.append(
                {
                    "start_beat": rubato_start,
                    "end_beat": end,
                    "start_time": beat_times[rubato_start],
                    "end_time": beat_times[end],
                    "max_deviation": float(np.max(deviations[rubato_start:])),
                }
            )

        return {
            "has_rubato": has_rubato,
            "rubato_regions": rubato_regions,
            "avg_deviation": float(np.mean(deviations)),
        }
