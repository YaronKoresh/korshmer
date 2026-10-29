import logging
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths

from .validation import validate_effect_integer, validate_effect_number
from .vocal_processing import VocalNote

logger = logging.getLogger(__name__)

MIN_MIDI = 24
MAX_MIDI = 100
CHUNK_SECONDS = 60.0
MIN_TUNING_FRAMES = 20
MIN_TUNING_CONCENTRATION = 0.3
MAX_STATE_COST = 2.5
GLIDE_SECONDS = 0.09
LEVEL_FRAME_LENGTH = 512
DIP_DB = 10.0
MIN_LEVEL_DB = -60.0
MIN_VELOCITY = 45
MAX_VELOCITY = 110


@dataclass(frozen=True)
class VocalMelodyOptions:
    sample_rate: int = 22050
    hop_length: int = 256
    frame_length: int = 2048
    fmin: float = 65.0
    fmax: float = 2100.0
    pitch_resolution: float = 0.2
    max_gap_frames: int = 4
    smooth_frames: int = 5
    switch_cost: float = 3.0
    min_note_s: float = 0.06
    max_level_drop_db: float = 30.0
    min_voiced_prob: float = 0.02
    speech_gate: bool = True
    gate_window_s: float = 3.0
    gate_run_s: float = 0.25
    gate_tolerance: float = 1.0
    gate_step_s: float = 0.06
    gate_step_tolerance: float = 0.15
    gate_min_voiced_s: float = 0.5
    gate_min_share: float = 0.5

    def __post_init__(self):
        validate_effect_integer(self.sample_rate, "Sample rate", 8000, 96000)
        validate_effect_integer(self.hop_length, "Hop length", 32, 4096)
        validate_effect_integer(self.frame_length, "Frame length", 512, 16384)
        validate_effect_number(self.fmin, "Minimum frequency", 20, 2000)
        validate_effect_number(self.fmax, "Maximum frequency", 40, 4000)
        if self.fmin >= self.fmax:
            raise ValueError(
                "Minimum frequency must be below the maximum, "
                f"got {self.fmin!r} and {self.fmax!r}"
            )
        validate_effect_number(
            self.pitch_resolution, "Pitch resolution", 0.05, 1
        )
        validate_effect_integer(self.max_gap_frames, "Maximum gap", 0, 50)
        validate_effect_integer(self.smooth_frames, "Smoothing frames", 1, 51)
        validate_effect_number(self.switch_cost, "Switch cost", 0.1, 20)
        validate_effect_number(self.min_note_s, "Minimum note length", 0, 2)
        validate_effect_number(
            self.max_level_drop_db, "Maximum level drop", 1, 120
        )
        validate_effect_number(
            self.min_voiced_prob, "Minimum voiced probability", 0, 1
        )
        validate_effect_number(self.gate_window_s, "Gate window", 0.5, 20)
        validate_effect_number(self.gate_run_s, "Gate run length", 0.05, 5)
        validate_effect_number(self.gate_tolerance, "Gate tolerance", 0.1, 6)
        validate_effect_number(self.gate_step_s, "Gate step length", 0.02, 5)
        validate_effect_number(
            self.gate_step_tolerance, "Gate step tolerance", 0.05, 6
        )
        validate_effect_number(
            self.gate_min_voiced_s, "Gate minimum voiced time", 0, 20
        )
        validate_effect_number(self.gate_min_share, "Gate minimum share", 0, 1)


def _report(progress, fraction, text):
    if progress is not None:
        progress(fraction, text)


def _track_pitch(y, options, progress, should_stop):
    hop = options.hop_length
    chunk = max(int(CHUNK_SECONDS * options.sample_rate) // hop, 1) * hop
    starts = list(range(0, len(y), chunk))
    if len(starts) > 1 and len(y) - starts[-1] < options.frame_length:
        starts.pop()
    ends = starts[1:] + [len(y)]
    pitches = []
    probabilities = []
    for number, (start, end) in enumerate(zip(starts, ends, strict=False)):
        if should_stop is not None and should_stop():
            raise InterruptedError("Stopped by user")
        f0, _, probability = librosa.pyin(
            y[start:end],
            fmin=options.fmin,
            fmax=options.fmax,
            sr=options.sample_rate,
            frame_length=options.frame_length,
            hop_length=hop,
            resolution=options.pitch_resolution,
            fill_na=np.nan,
        )
        keep = None if end == len(y) else (end - start) // hop
        pitches.append(f0[:keep])
        probabilities.append(probability[:keep])
        _report(
            progress,
            0.1 + 0.8 * (number + 1) / len(starts),
            "Following the singer's pitch...",
        )
    return np.concatenate(pitches), np.concatenate(probabilities)


def _tuning_offset(midi):
    values = midi[np.isfinite(midi)]
    if values.size <= MIN_TUNING_FRAMES:
        return 0.0
    angle = 2 * np.pi * (values - np.round(values))
    cos_mean = np.cos(angle).mean()
    sin_mean = np.sin(angle).mean()
    if np.hypot(cos_mean, sin_mean) <= MIN_TUNING_CONCENTRATION:
        return 0.0
    return float(np.arctan2(sin_mean, cos_mean) / (2 * np.pi))


def _voiced_runs(voiced, max_gap_frames):
    index = np.flatnonzero(voiced)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) > max_gap_frames + 1) + 1
    return [(run[0], run[-1] + 1) for run in np.split(index, breaks)]


def _smooth_contour(midi, options):
    smooth = np.full(midi.shape, np.nan)
    runs = _voiced_runs(np.isfinite(midi), options.max_gap_frames)
    for start, stop in runs:
        frames = np.arange(start, stop)
        known = frames[np.isfinite(midi[start:stop])]
        segment = np.interp(frames, known, midi[known])
        if len(segment) >= options.smooth_frames > 1:
            segment = median_filter(
                segment, size=options.smooth_frames, mode="nearest"
            )
        smooth[start:stop] = segment
    return smooth, runs


def _steady_frames(smooth, window, tolerance):
    steady = np.zeros(len(smooth), dtype=bool)
    if len(smooth) < window:
        return steady
    views = np.lib.stride_tricks.sliding_window_view(smooth, window)
    spread = views.max(axis=1) - views.min(axis=1)
    starts = (spread <= 2 * tolerance).astype(int)
    return np.convolve(starts, np.ones(window, dtype=int))[: len(smooth)] > 0


def singing_share(smooth, hop_s, options=None):
    options = options or VocalMelodyOptions()
    voiced = np.isfinite(smooth)
    steady = np.zeros(len(smooth), dtype=bool)
    for run_s, tolerance in (
        (options.gate_run_s, options.gate_tolerance),
        (options.gate_step_s, options.gate_step_tolerance),
    ):
        window = max(int(round(run_s / hop_s)), 2)
        steady |= _steady_frames(smooth, window, tolerance)
    steady &= voiced
    half = int(round(options.gate_window_s / hop_s / 2))
    frames = np.arange(len(smooth))
    low = np.clip(frames - half, 0, len(smooth))
    high = np.clip(frames + half + 1, 0, len(smooth))
    voiced_sum = np.concatenate([[0], np.cumsum(voiced)])
    steady_sum = np.concatenate([[0], np.cumsum(steady)])
    voiced_count = voiced_sum[high] - voiced_sum[low]
    steady_count = steady_sum[high] - steady_sum[low]
    share = np.full(len(smooth), np.nan)
    enough = voiced_count * hop_s >= options.gate_min_voiced_s
    share[enough] = steady_count[enough] / voiced_count[enough]
    return share


def _viterbi_semitones(segment, switch_cost):
    low = max(int(np.floor(segment.min())) - 1, MIN_MIDI)
    high = min(int(np.ceil(segment.max())) + 1, MAX_MIDI)
    states = np.arange(low, high + 1)
    cost = np.minimum(np.abs(segment[:, None] - states), MAX_STATE_COST)
    jumped = np.zeros(cost.shape, dtype=bool)
    source = np.zeros(len(segment), dtype=int)
    total = cost[0].copy()
    for frame in range(1, len(segment)):
        best = int(total.argmin())
        alternative = total[best] + switch_cost
        jumped[frame] = alternative < total
        source[frame] = best
        total = np.where(jumped[frame], alternative, total) + cost[frame]
    state = int(total.argmin())
    path = np.empty(len(segment), dtype=int)
    for frame in range(len(segment) - 1, -1, -1):
        path[frame] = states[state]
        if jumped[frame, state]:
            state = source[frame]
    return path


def _run_spans(smooth, run, options):
    start, stop = run
    path = _viterbi_semitones(smooth[start:stop], options.switch_cost)
    cuts = np.flatnonzero(np.diff(path)) + 1
    firsts = np.concatenate([[0], cuts])
    lasts = np.concatenate([cuts, [len(path)]])
    frame_s = options.hop_length / options.sample_rate
    spans = [
        (first, last, int(path[first]))
        for first, last in zip(firsts, lasts, strict=False)
        if (last - first) * frame_s >= options.min_note_s
    ]
    kept = []
    glide_start = None
    for index, (first, last, pitch) in enumerate(spans):
        if 0 < index < len(spans) - 1 and (last - first) * frame_s < (
            GLIDE_SECONDS
        ):
            before, after = spans[index - 1][2], spans[index + 1][2]
            if min(before, after) < pitch < max(before, after):
                glide_start = first
                continue
        if glide_start is not None:
            first, glide_start = glide_start, None
        if kept and kept[-1][2] == pitch:
            kept[-1] = (kept[-1][0], last, pitch)
        else:
            kept.append((first, last, pitch))
    return [(start + first, start + last, p) for first, last, p in kept]


def _split_at_dips(span, db, min_frames):
    first, last, pitch = span
    dips, _ = find_peaks(-db[first:last], prominence=DIP_DB)
    if not dips.size:
        return [span]
    _, _, left, right = peak_widths(-db[first:last], dips, rel_height=0.5)
    edges = [first]
    for start, stop in zip(left, right, strict=False):
        edges += [first + int(round(start)), first + int(round(stop))]
    edges.append(last)
    return [
        (begin, end, pitch)
        for begin, end in zip(edges[::2], edges[1::2], strict=False)
        if end - begin >= min_frames
    ]


def _velocity(level, reference, drop):
    return int(
        round(
            np.interp(
                level - reference, [-drop, 0], [MIN_VELOCITY, MAX_VELOCITY]
            )
        )
    )


def _notes_from_pitch(f0, probability, rms, options):
    midi = librosa.hz_to_midi(f0)
    midi = midi - _tuning_offset(midi)
    db = 20 * np.log10(np.maximum(rms[: len(midi)], 1e-7))
    smooth, runs = _smooth_contour(midi, options)
    spans = [span for run in runs for span in _run_spans(smooth, run, options)]
    hop_s = options.hop_length / options.sample_rate
    min_frames = max(int(np.ceil(options.min_note_s / hop_s)), 1)
    spans = [
        part for span in spans for part in _split_at_dips(span, db, min_frames)
    ]
    spans = [
        span
        for span in spans
        if probability[span[0] : span[1]].mean() >= options.min_voiced_prob
    ]
    if options.speech_gate and spans:
        share = singing_share(smooth, hop_s, options)
        spans = [
            span
            for span in spans
            if not share[(span[0] + span[1]) // 2] < options.gate_min_share
        ]
    if not spans:
        return []
    levels = np.array(
        [
            20 * np.log10(max(rms[first:last].mean(), 1e-7))
            for first, last, _ in spans
        ]
    )
    reference = np.percentile(levels, 95)
    notes = []
    for (first, last, pitch), level in zip(spans, levels, strict=False):
        if level < max(reference - options.max_level_drop_db, MIN_LEVEL_DB):
            continue
        notes.append(
            VocalNote(
                float(first * hop_s),
                float(last * hop_s),
                pitch,
                velocity=_velocity(level, reference, options.max_level_drop_db),
            )
        )
    return notes


def transcribe_vocal_melody(
    audio_path, options=None, progress=None, *, should_stop=None
):
    options = options or VocalMelodyOptions()
    _report(progress, 0.0, "Loading vocals...")
    y, _ = librosa.load(audio_path, sr=options.sample_rate, mono=True)
    if len(y) < options.frame_length:
        return []
    f0, probability = _track_pitch(y, options, progress, should_stop)
    rms = librosa.feature.rms(
        y=y, frame_length=LEVEL_FRAME_LENGTH, hop_length=options.hop_length
    )[0]
    notes = _notes_from_pitch(f0, probability, rms, options)
    logger.info("Vocal melody: %d notes from %s", len(notes), audio_path)
    _report(progress, 1.0, f"Vocal notes found: {len(notes)}")
    return notes
