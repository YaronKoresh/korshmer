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
SAME_TEMPO_DRIFT_S = 0.08
OWN_TEMPO_LEAD = 1.25
RIVAL_DISTANCE_SHARE = 0.02
COARSE_LEVELS = 2
SECTIONS = 3
MIN_SECTION_SECONDS = 30.0
MIN_SECTION_HITS = 8
SECTION_AGREEMENT_SHARE = 0.003
REFINE_SHARE = 0.004
AGREED_CERTAINTY = 3.2


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


def _scan(seconds, song_bpm):
    length = float(seconds[-1] - seconds[0])
    finest = max(PULSE_STEPS_PER_BEAT)
    step = min(
        MAX_SCAN_STEP_BPM, 60.0 / (length * finest * SCAN_POINTS_PER_PEAK)
    )
    return (
        np.arange(
            song_bpm * SLOWEST_SHARE, song_bpm * FASTEST_SHARE + step, step
        ),
        step,
    )


def _differs(bpm, song_bpm, length):
    return length * abs(bpm - song_bpm) / song_bpm >= SAME_TEMPO_DRIFT_S


def _clear_tempo(seconds, weights, song_bpm):
    length = float(seconds[-1] - seconds[0])
    scan, step = _scan(seconds, song_bpm)
    fit = _levels(seconds, weights, scan).sum(axis=0)
    near = scan[int(np.argmax(fit))] + np.linspace(-step, step, 41)
    fine = _levels(seconds, weights, near)
    best = int(np.argmax(fine.sum(axis=0)))
    bpm = float(near[best])
    certainty = float(fine[:COARSE_LEVELS, best].max())
    if certainty < TEMPO_CERTAINTY or not _differs(bpm, song_bpm, length):
        return None
    own = float(fine[:, best].sum())
    at_song = float(_levels(seconds, weights, song_bpm).sum())
    others = fit[np.abs(scan - bpm) > RIVAL_DISTANCE_SHARE * bpm]
    rival = max(at_song, float(others.max()) if others.size else 0.0)
    if own < OWN_TEMPO_LEAD * rival:
        return None
    return bpm, certainty


def _section_tempo(seconds, weights, song_bpm):
    if seconds.shape[0] < MIN_SECTION_HITS or float(np.sum(weights**2)) <= 0.0:
        return None
    scan, _step = _scan(seconds, song_bpm)
    return float(
        scan[int(np.argmax(_levels(seconds, weights, scan).sum(axis=0)))]
    )


def _agreed_tempo(seconds, weights, song_bpm):
    length = float(seconds[-1] - seconds[0])
    if length < SECTIONS * MIN_SECTION_SECONDS:
        return None
    edges = seconds[0] + np.linspace(0.0, length, SECTIONS + 1)
    found = []
    for first, last in zip(edges[:-1], edges[1:], strict=True):
        inside = (seconds >= first) & (seconds <= last)
        tempo = _section_tempo(seconds[inside], weights[inside], song_bpm)
        if tempo is not None:
            found.append(tempo)
    pairs = [
        (abs(one - other) / one, (one + other) / 2.0)
        for index, one in enumerate(found)
        for other in found[index + 1 :]
    ]
    close = [pair for pair in pairs if pair[0] <= SECTION_AGREEMENT_SHARE]
    if not close:
        return None
    centre = min(close)[1]
    near = centre * (1.0 + np.linspace(-REFINE_SHARE, REFINE_SHARE, 81))
    fine = _levels(seconds, weights, near)
    best = int(np.argmax(fine.sum(axis=0)))
    bpm = float(near[best])
    certainty = float(fine[:COARSE_LEVELS, best].max())
    if certainty < AGREED_CERTAINTY or not _differs(bpm, song_bpm, length):
        return None
    return bpm, certainty


def own_tempo(seconds, weights, song_bpm):
    if seconds.shape[0] < 2 or float(np.sum(weights**2)) <= 0.0:
        return None
    if float(seconds[-1] - seconds[0]) <= 0.0:
        return None
    return _clear_tempo(seconds, weights, song_bpm) or _agreed_tempo(
        seconds, weights, song_bpm
    )
