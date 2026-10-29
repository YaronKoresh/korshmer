import librosa
import numpy as np

from korshmer.audio_processing.validation import validate_sample_rate

HOP_LENGTH = 512
MIN_BEATS_FOR_FIT = 8
MAX_FIT_DEVIATION = 0.04
REFINE_SPAN = 0.015
REFINE_STEP = 0.0002
MIN_ALIGNMENT = 0.3
MIN_BPM_ESTIMATE = 20.0
MAX_BPM_ESTIMATE = 300.0
DECIMALS = 2


def _beat_line_tempo(envelope, sr, candidate):
    _, beats = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=candidate,
        tightness=100,
        units="time",
    )
    if len(beats) < MIN_BEATS_FOR_FIT:
        return None
    slope = np.polyfit(np.arange(len(beats)), beats, 1)[0]
    if slope <= 0:
        return None
    fitted = 60.0 / float(slope)
    if abs(fitted - candidate) / candidate > MAX_FIT_DEVIATION:
        return None
    return fitted


def _refine_by_alignment(onset_times, tempo):
    times = np.asarray(onset_times, dtype=float)
    if len(times) < MIN_BEATS_FOR_FIT:
        return tempo
    steps = int(round(REFINE_SPAN / REFINE_STEP))
    tempos = tempo * (1.0 + REFINE_STEP * np.arange(-steps, steps + 1))
    best_tempo, best_alignment = tempo, 0.0
    for divisions in (1, 2):
        freqs = divisions * tempos / 60.0
        phases = np.exp(2j * np.pi * np.outer(freqs, times))
        alignment = np.abs(phases.mean(axis=1))
        index = int(np.argmax(alignment))
        if alignment[index] > best_alignment:
            best_alignment, best_tempo = (
                float(alignment[index]),
                float(tempos[index]),
            )
    return best_tempo if best_alignment >= MIN_ALIGNMENT else tempo


def estimate_tempo(y, sr):
    sr = validate_sample_rate(sr)
    audio = np.asarray(y, dtype=np.float32)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        return None
    envelope = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )
    candidate = float(
        librosa.feature.tempo(
            onset_envelope=envelope, sr=sr, hop_length=HOP_LENGTH
        )[0]
    )
    if not np.isfinite(candidate) or not (
        MIN_BPM_ESTIMATE <= candidate <= MAX_BPM_ESTIMATE
    ):
        return None
    tempo = _beat_line_tempo(envelope, sr, candidate) or candidate
    onsets = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sr, hop_length=HOP_LENGTH, units="time"
    )
    tempo = _refine_by_alignment(onsets, tempo)
    return round(float(tempo), DECIMALS)
