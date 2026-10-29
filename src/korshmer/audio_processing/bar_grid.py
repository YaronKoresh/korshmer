from dataclasses import dataclass, field

import librosa
import numpy as np

from korshmer.utils.tempo import MAX_BPM, MIN_BPM

from .validation import (
    validate_audio_not_empty,
    validate_effect_integer,
    validate_effect_number,
    validate_sample_rate,
)

HOP_LENGTH = 128
PHASE_STEP_SECONDS = 0.002
SILENCE_RATIO = 0.02
PICKUP_TOLERANCE = 0.25
ACCENT_BEFORE_SECONDS = 0.02
ACCENT_AFTER_SECONDS = 0.1
LOW_BAND_HZ = 250.0
MIN_ACCENT_GAIN = 1.1
MIN_BEATS_PER_BAR = 2
MAX_BEATS_PER_BAR = 12
TAIL_GUARD_SECONDS = 0.05
ENVELOPE_CHUNK_SECONDS = 20.0
ENVELOPE_PAD_SECONDS = 1.0
ACCENT_FFT = 2048
ACCENT_HOP = 512


@dataclass
class BarGrid:
    tempo: float
    beats_per_bar: int
    content_start: float
    content_end: float
    beat_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    downbeat_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pickup_beats: int = 0
    last_bar_beats: int = 0

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.tempo

    @property
    def first_downbeat(self) -> float:
        if len(self.downbeat_times):
            return float(self.downbeat_times[0])
        return self.content_start

    @property
    def pickup_seconds(self) -> float:
        return max(0.0, self.first_downbeat - self.content_start)

    @property
    def ends_mid_bar(self) -> bool:
        return 0 < self.last_bar_beats < self.beats_per_bar


def _content_span(audio, sr):
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return 0.0, 0.0
    loud = np.flatnonzero(np.abs(audio) > peak * SILENCE_RATIO)
    return float(loud[0]) / sr, float(loud[-1] + 1) / sr


def _onset_envelope(audio, sr):
    hop = HOP_LENGTH
    piece = (int(ENVELOPE_CHUNK_SECONDS * sr) // hop) * hop
    pad = (int(ENVELOPE_PAD_SECONDS * sr) // hop) * hop
    if len(audio) <= piece + 2 * pad:
        return librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    parts = []
    for start in range(0, len(audio), piece):
        first = max(0, start - pad)
        chunk = audio[first : start + piece + pad]
        env = librosa.onset.onset_strength(y=chunk, sr=sr, hop_length=hop)
        skip = (start - first) // hop
        parts.append(env[skip : skip + piece // hop])
    envelope = np.concatenate(parts)
    return envelope[: len(audio) // hop + 1]


def _best_phase(envelope, frame_seconds, period, duration):
    phases = np.arange(0.0, period, PHASE_STEP_SECONDS)
    frame_times = np.arange(len(envelope)) * frame_seconds
    best_phase, best_score = 0.0, -1.0
    for phase in phases:
        beats = np.arange(phase, duration, period)
        score = float(np.sum(np.interp(beats, frame_times, envelope)))
        if score > best_score + 1e-9:
            best_phase, best_score = float(phase), score
    return best_phase


def _accents(audio, sr, beats):
    freqs = np.fft.rfftfreq(ACCENT_FFT, 1.0 / sr)
    low_bins = freqs < LOW_BAND_HZ
    before = int(round(ACCENT_BEFORE_SECONDS * sr))
    after = int(round(ACCENT_AFTER_SECONDS * sr))
    window = np.hanning(ACCENT_FFT).astype(np.float32)
    total = np.zeros(len(beats))
    bass = np.zeros(len(beats))
    for index, at in enumerate(beats):
        centre = int(round(at * sr))
        first = max(0, centre - before)
        segment = audio[first : centre + after]
        if len(segment) < ACCENT_FFT:
            segment = np.pad(segment, (0, ACCENT_FFT - len(segment)))
        starts = range(0, len(segment) - ACCENT_FFT + 1, ACCENT_HOP)
        frames = np.stack(
            [segment[s : s + ACCENT_FFT] * window for s in starts]
        )
        magnitude = np.abs(np.fft.rfft(frames, axis=1))
        total[index] = magnitude.sum(axis=1).max()
        bass[index] = magnitude[:, low_bins].sum(axis=1).max()
    scores = np.zeros(len(beats))
    for peaks in (total, bass):
        scale = float(np.percentile(peaks, 90)) if len(peaks) else 0.0
        if scale > 0:
            scores += peaks / scale
    return scores


def find_bar_grid(audio, sr, tempo, beats_per_bar=4):
    sr = validate_sample_rate(sr)
    validate_audio_not_empty(audio)
    tempo = float(validate_effect_number(tempo, "Tempo", MIN_BPM, MAX_BPM))
    beats_per_bar = validate_effect_integer(
        beats_per_bar, "Beats per bar", MIN_BEATS_PER_BAR, MAX_BEATS_PER_BAR
    )
    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sr
    period = 60.0 / tempo
    start, end = _content_span(audio, sr)
    grid = BarGrid(tempo, beats_per_bar, start, end)
    if end - start <= 0:
        return grid
    frame_seconds = HOP_LENGTH / sr
    envelope = _onset_envelope(audio, sr)
    phase = _best_phase(envelope, frame_seconds, period, duration)
    beats = np.arange(phase, duration, period)
    beats = beats[
        (beats >= start - PICKUP_TOLERANCE * period)
        & (beats <= end - TAIL_GUARD_SECONDS)
    ]
    grid.beat_times = beats
    if len(beats) == 0:
        grid.downbeat_times = np.array([start])
        return grid
    accents = _accents(audio, sr, beats)
    shift = 0
    if len(beats) >= beats_per_bar:
        totals = [
            accents[p::beats_per_bar].mean() for p in range(beats_per_bar)
        ]
        best = int(np.argmax(totals))
        others = [t for i, t in enumerate(totals) if i != best]
        if totals[best] > MIN_ACCENT_GAIN * max(others):
            shift = best
    grid.pickup_beats = shift
    grid.downbeat_times = beats[shift::beats_per_bar]
    last = float(grid.downbeat_times[-1])
    in_last = int(np.sum(beats >= last - 1e-6))
    grid.last_bar_beats = min(beats_per_bar, in_last)
    return grid
