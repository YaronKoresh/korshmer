import logging
from collections.abc import Callable

import librosa
import numpy as np

from .pitch_estimator import autocorrelation_pitch, piptrack_pitch
from .syllable_detection import SyllableDetector
from .validation import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    is_empty_audio,
    validate_confidence_threshold,
)

logger = logging.getLogger(__name__)


class VocalNote:
    def __init__(
        self,
        start_time: float,
        end_time: float,
        pitch: int,
        audio_segment: np.ndarray | None = None,
        velocity: int = 100,
    ) -> None:
        if start_time >= end_time:
            raise ValueError(
                f"start_time ({start_time}) must be less than end_time ({end_time})"
            )
        if not (0 <= pitch <= 127):
            raise ValueError(f"MIDI pitch ({pitch}) must be in range 0-127")
        if not (0 <= velocity <= 127):
            raise ValueError(f"Velocity ({velocity}) must be in range 0-127")

        self.start = start_time
        self.end = end_time
        self.pitch = pitch
        self.audio_segment = audio_segment
        self.velocity = velocity
        self.duration = end_time - start_time

    def __repr__(self) -> str:
        return f"VocalNote(start={self.start:.3f}, end={self.end:.3f}, pitch={self.pitch}, dur={self.duration:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VocalNote):
            return False
        return (
            abs(self.start - other.start) < 1e-6
            and abs(self.end - other.end) < 1e-6
            and self.pitch == other.pitch
            and self.velocity == other.velocity
        )

    def __hash__(self) -> int:
        return hash(
            (
                round(self.start, 6),
                round(self.end, 6),
                self.pitch,
                self.velocity,
            )
        )


class VocalSyllableDetector(SyllableDetector):
    def _autocorrelation_pitch(
        self,
        audio_segment: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> tuple[float, float]:
        return autocorrelation_pitch(
            audio_segment, self.sr, self.fmin, self.fmax, confidence_threshold
        )

    def extract_vocal_notes(
        self,
        audio: np.ndarray,
        syllables: list[tuple[float, float]],
        progress_callback: Callable[[float], None] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> list[VocalNote]:
        validate_confidence_threshold(confidence_threshold)

        if is_empty_audio(audio):
            logger.warning("Empty audio array provided to extract_vocal_notes")
            return []

        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio must be numpy array, got {type(audio)}")

        vocal_notes = []

        if len(syllables) == 0:
            return vocal_notes

        total_syllables = len(syllables)
        for idx, (start_time, end_time) in enumerate(syllables):
            if progress_callback:
                progress = (idx / total_syllables) if total_syllables else 0.0
                progress_callback(progress)

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            audio_segment = audio[start_sample:end_sample]

            if len(audio_segment) == 0:
                logger.warning(
                    f"Empty audio segment at {start_time:.3f}s - {end_time:.3f}s"
                )
                continue

            midi_pitch = 57
            confidence = 0.0

            try:
                pitch_hz, max_magnitude = piptrack_pitch(
                    audio_segment, self.sr, self.fmin, self.fmax
                )

                if pitch_hz > 0 and max_magnitude > 0.01:
                    midi_pitch = int(
                        np.clip(librosa.hz_to_midi(pitch_hz), 0, 127)
                    )
                    confidence = max_magnitude
                else:
                    raise ValueError("Low confidence pitch detection")

            except (ValueError, librosa.ParameterError, IndexError) as e:
                logger.warning(
                    f"Primary pitch estimation failed at {start_time:.3f}s: {e}. Trying autocorrelation fallback."
                )

                try:
                    pitch_hz, confidence = self._autocorrelation_pitch(
                        audio_segment, confidence_threshold
                    )

                    if pitch_hz > 0:
                        midi_pitch = int(
                            np.clip(librosa.hz_to_midi(pitch_hz), 0, 127)
                        )
                        logger.info(
                            f"Autocorrelation pitch: {pitch_hz:.1f} Hz (MIDI {midi_pitch}), confidence: {confidence:.3f}"
                        )
                    else:
                        logger.warning(
                            "Autocorrelation also failed, using default pitch"
                        )

                except (OSError, ValueError, RuntimeError) as fallback_error:
                    logger.error(
                        f"Fallback pitch detection failed at {start_time:.3f}s: {fallback_error}"
                    )

            if confidence < confidence_threshold:
                logger.debug(
                    f"Low confidence ({confidence:.3f}) pitch at {start_time:.3f}s, MIDI {midi_pitch}"
                )

            if not (self.fmin <= librosa.midi_to_hz(midi_pitch) <= self.fmax):
                logger.warning(
                    f"Detected pitch {midi_pitch} ({librosa.midi_to_hz(midi_pitch):.1f} Hz) "
                    f"outside expected {self.voice_type} range ({self.fmin}-{self.fmax} Hz)"
                )

            try:
                note = VocalNote(
                    start_time=start_time,
                    end_time=end_time,
                    pitch=midi_pitch,
                    audio_segment=audio_segment,
                    velocity=100,
                )
                vocal_notes.append(note)
            except ValueError as e:
                logger.error(f"Failed to create VocalNote: {e}")
                continue

        if progress_callback:
            progress_callback(1.0)

        return vocal_notes


def detect_vocal_syllables(
    audio_path: str,
    sample_rate: int = 44100,
    voice_type: str = "soprano",
    progress_callback: Callable[[float], None] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[VocalNote]:
    validate_confidence_threshold(confidence_threshold)
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    detector = VocalSyllableDetector(sample_rate=sr, voice_type=voice_type)

    syllables = detector.detect_syllables(
        audio, progress_callback=progress_callback
    )

    vocal_notes = detector.extract_vocal_notes(
        audio,
        syllables,
        progress_callback=progress_callback,
        confidence_threshold=confidence_threshold,
    )

    return vocal_notes
