import logging
from collections.abc import Callable

import librosa
import numpy as np
from scipy import signal as scipy_signal

from .validation import (
    is_empty_audio,
    validate_effect_number,
    validate_sample_rate,
)

logger = logging.getLogger(__name__)


class SyllableDetector:
    VOICE_RANGES: dict = {
        "bass": (80, 350),
        "tenor": (130, 520),
        "alto": (175, 700),
        "soprano": (260, 2000),
    }

    def __init__(
        self,
        sample_rate: int | float = 44100,
        voice_type: str = "soprano",
    ) -> None:
        validate_sample_rate(sample_rate)

        self.sr: int = int(sample_rate)
        self.voice_type: str = voice_type
        self.fmin: float
        self.fmax: float
        self.fmin, self.fmax = self.VOICE_RANGES.get(voice_type, (80, 2000))

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        return audio - np.mean(audio)

    def _noise_gate(
        self,
        audio: np.ndarray,
        threshold_db: float = -40,
        attack_time: float = 0.001,
        release_time: float = 0.05,
    ) -> np.ndarray:
        if threshold_db >= 0:
            raise ValueError(
                f"threshold_db must be negative (typical range: -60 to -20 dB), got {threshold_db}"
            )

        if is_empty_audio(audio) or np.allclose(audio, 0):
            return audio

        audio = self._remove_dc_offset(audio)

        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio

        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=max_val)
        gate_mask = audio_db > threshold_db

        attack_samples = int(attack_time * self.sr)
        release_samples = int(release_time * self.sr)

        smoothed_mask = np.zeros_like(gate_mask, dtype=float)
        envelope = 0.0

        for i in range(len(gate_mask)):
            target = 1.0 if gate_mask[i] else 0.0
            if target > envelope:
                step = 1.0 / max(1, attack_samples)
            else:
                step = 1.0 / max(1, release_samples)
            envelope += (target - envelope) * step
            smoothed_mask[i] = envelope

        gated_audio = audio * smoothed_mask
        return gated_audio

    def _smooth_envelope(
        self, audio: np.ndarray, frame_length: int = 2048
    ) -> np.ndarray:
        hop_length = max(1, frame_length // 4)
        envelope = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        envelope = scipy_signal.resample(envelope, len(audio))

        return envelope

    def _find_onset_times(self, audio: np.ndarray) -> np.ndarray:
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sr, units="frames", hop_length=512, backtrack=True
        )

        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sr, hop_length=512
        )

        return onset_times

    def _segment_onsets(
        self,
        audio: np.ndarray,
        onset_times: np.ndarray,
        min_syllable_duration: float,
        max_syllable_duration: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[tuple[float, float]]:
        syllables: list[tuple[float, float]] = []
        total_onsets = len(onset_times)
        for i, start_time in enumerate(onset_times):
            if progress_callback and total_onsets > 0:
                progress_callback(i / total_onsets)

            if i < len(onset_times) - 1:
                end_time = onset_times[i + 1]
            else:
                end_time = len(audio) / self.sr

            duration = end_time - start_time

            if duration < min_syllable_duration:
                continue

            if duration > max_syllable_duration:
                num_splits = int(np.ceil(duration / max_syllable_duration))
                split_duration = duration / num_splits
                for j in range(num_splits):
                    split_start = start_time + j * split_duration
                    split_end = start_time + (j + 1) * split_duration
                    if split_end - split_start >= min_syllable_duration:
                        syllables.append((split_start, split_end))
            else:
                syllables.append((start_time, end_time))

        return syllables

    def _merge_close_syllables(
        self,
        syllables: list[tuple[float, float]],
        min_gap_duration: float,
    ) -> list[tuple[float, float]]:
        filtered_syllables: list[tuple[float, float]] = []
        for i, (start, end) in enumerate(syllables):
            if i > 0:
                prev_end = filtered_syllables[-1][1]
                gap = start - prev_end
                if gap < min_gap_duration:
                    filtered_syllables[-1] = (filtered_syllables[-1][0], end)
                    continue
            filtered_syllables.append((start, end))
        return filtered_syllables

    def detect_syllables(
        self,
        audio: np.ndarray,
        min_syllable_duration: float = 0.05,
        max_syllable_duration: float = 5.0,
        min_gap_duration: float = 0.03,
        noise_threshold_db: float = -40,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[tuple[float, float]]:
        validate_effect_number(
            min_syllable_duration, "Minimum syllable duration", 0.0, 10.0
        )
        validate_effect_number(
            max_syllable_duration, "Maximum syllable duration", 0.01, 60.0
        )
        validate_effect_number(min_gap_duration, "Minimum gap", 0.0, 10.0)
        validate_effect_number(
            noise_threshold_db, "Noise threshold", -120.0, 0.0
        )
        if min_syllable_duration > max_syllable_duration:
            raise ValueError(
                "Minimum syllable duration cannot exceed the maximum, got "
                f"{min_syllable_duration!r} and {max_syllable_duration!r}"
            )
        if is_empty_audio(audio) or np.allclose(audio, 0):
            return []

        gated_audio = self._noise_gate(audio, threshold_db=noise_threshold_db)

        onset_times = self._find_onset_times(gated_audio)

        if len(onset_times) == 0:
            return []

        syllables = self._segment_onsets(
            audio,
            onset_times,
            min_syllable_duration,
            max_syllable_duration,
            progress_callback,
        )

        filtered_syllables = self._merge_close_syllables(
            syllables, min_gap_duration
        )

        if progress_callback:
            progress_callback(1.0)

        return filtered_syllables
