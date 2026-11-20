import numpy as np

from korshmer.audio_processing import clip_tempo
from korshmer.audio_processing.tempo_align import stretch_to_tempo
from korshmer.midi.tempo_map import TempoMap

STRONG_START_SHARE = 0.15
MIN_START_GAP_S = 0.12
SLOWEST_STRETCH = 0.6
FASTEST_STRETCH = 1.67
ON_GRID_S = 0.015
REFERENCE_BPM = 120.0
MIN_MOVED_SHARE = 0.3


def sound_starts(mono, sample_rate):
    seconds, weights = clip_tempo.onsets(mono, sample_rate)
    if seconds.shape[0] == 0 or float(weights.max()) <= 0.0:
        return []
    strong = weights >= STRONG_START_SHARE * float(weights.max())
    kept = []
    for second, weight in sorted(
        zip(seconds[strong], weights[strong], strict=True),
        key=lambda found: -found[1],
    ):
        if all(abs(second - other) >= MIN_START_GAP_S for other, _ in kept):
            kept.append((float(second), float(weight)))
    return sorted(kept)


def grid_moves(starts, beats_at, clip_start, steps_per_beat, bpm):
    moves = []
    for second, weight in starts:
        beat = float(beats_at(clip_start + second))
        wanted = round(beat * steps_per_beat) / steps_per_beat
        moves.append((second, second + (wanted - beat) * 60.0 / bpm, weight))
    return moves


def _fits(anchors, source, target):
    earlier = max(anchor for anchor in anchors if anchor[0] < source)
    later = min(anchor for anchor in anchors if anchor[0] > source)
    for (from_a, to_a), (from_b, to_b) in (
        (earlier, (source, target)),
        ((source, target), later),
    ):
        before, after = from_b - from_a, to_b - to_a
        if before < MIN_START_GAP_S / 2.0 or after <= 0.0:
            return False
        if not SLOWEST_STRETCH <= after / before <= FASTEST_STRETCH:
            return False
    return True


def beat_map(moves, begin, end):
    anchors = [(float(begin), float(begin)), (float(end), float(end))]
    for source, target, _weight in sorted(moves, key=lambda move: -move[2]):
        if not begin < source < end:
            continue
        if any(abs(source - anchor[0]) < 1e-9 for anchor in anchors):
            continue
        if _fits(anchors, source, target):
            anchors.append((float(source), float(target)))
    return sorted(anchors)


def common_move(moves):
    if not moves:
        return 0.0
    return float(np.median([target - source for source, target, _ in moves]))


def worth_stretching(anchors):
    inside = anchors[1:-1]
    if not inside:
        return False
    return moved_sounds(inside) >= MIN_MOVED_SHARE * len(inside)


def largest_move(anchors):
    return max(abs(target - source) for source, target in anchors)


def moved_sounds(anchors):
    return sum(abs(target - source) >= ON_GRID_S for source, target in anchors)


def stretched_onto_beat(data, sample_rate, anchors):
    starts, tempos = [], []
    for (from_a, to_a), (from_b, to_b) in zip(
        anchors, anchors[1:], strict=False
    ):
        starts.append(from_a)
        tempos.append(REFERENCE_BPM * (to_b - to_a) / (from_b - from_a))
    warped = stretch_to_tempo(
        np.asarray(data, dtype=np.float32).T,
        sample_rate,
        0.0,
        TempoMap(tuple(starts), tuple(tempos)),
        REFERENCE_BPM,
    )
    return warped.T
