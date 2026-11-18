import librosa
import numpy as np

PULSE_HOP = 256
PULSE_STEPS_PER_BEAT = (1, 2, 4)
CUT_EDGE_SECONDS = 0.05
NOISE_SPREADS = 2.0
SLOWEST_SHARE = 0.75
FASTEST_SHARE = 1.34
SCAN_POINTS_PER_PEAK = 6
MAX_SCAN_STEP_BPM = 0.1
TEMPO_CERTAINTY = 4.5
SAME_TEMPO_DRIFT_S = 0.03
OWN_TEMPO_LEAD = 1.25
RIVAL_DISTANCE_SHARE = 0.02
COARSE_LEVELS = 2


def onsets(mono, sample_rate):
    strength = librosa.onset.onset_strength(
        y=mono, sr=sample_rate, hop_length=PULSE_HOP
    )
    hits = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sample_rate, hop_length=PULSE_HOP
    )
    hits = hits[hits >= CUT_EDGE_SECONDS * sample_rate / PULSE_HOP]
    middle = float(np.median(strength))
    floor = middle + NOISE_SPREADS * float(np.median(np.abs(strength - middle)))
    weights = np.maximum(strength[hits] - floor, 0.0).astype(np.float64)
    return hits * PULSE_HOP / sample_rate, weights


def _levels(seconds, weights, bpms):
    spread = float(np.sqrt(np.sum(weights**2)))
    turns = np.outer(np.atleast_1d(bpms) / 60.0, seconds)
    return np.array(
        [
            np.abs(np.exp(2j * np.pi * steps * turns) @ weights) / spread
            for steps in PULSE_STEPS_PER_BEAT
        ]
    )


def own_tempo(seconds, weights, song_bpm):
    if seconds.shape[0] < 2 or float(np.sum(weights**2)) <= 0.0:
        return None
    length = float(seconds[-1] - seconds[0])
    if length <= 0.0:
        return None
    finest = max(PULSE_STEPS_PER_BEAT)
    step = min(
        MAX_SCAN_STEP_BPM, 60.0 / (length * finest * SCAN_POINTS_PER_PEAK)
    )
    scan = np.arange(
        song_bpm * SLOWEST_SHARE, song_bpm * FASTEST_SHARE + step, step
    )
    fit = _levels(seconds, weights, scan).sum(axis=0)
    near = scan[int(np.argmax(fit))] + np.linspace(-step, step, 41)
    fine = _levels(seconds, weights, near)
    best = int(np.argmax(fine.sum(axis=0)))
    bpm = float(near[best])
    certainty = float(fine[:COARSE_LEVELS, best].max())
    drift = length * abs(bpm - song_bpm) / song_bpm
    if certainty < TEMPO_CERTAINTY or drift < SAME_TEMPO_DRIFT_S:
        return None
    own = float(fine[:, best].sum())
    at_song = float(_levels(seconds, weights, song_bpm).sum())
    others = fit[np.abs(scan - bpm) > RIVAL_DISTANCE_SHARE * bpm]
    rival = max(at_song, float(others.max()) if others.size else 0.0)
    if own < OWN_TEMPO_LEAD * rival:
        return None
    return bpm, certainty
