import math
import numbers
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from korshmer.utils.tempo import MAX_BPM, MIN_BPM

MATCH_LEVELS = ("Weak", "Medium", "Strong")
AUTO_MATCH = "Auto"
DEFAULT_MATCH = AUTO_MATCH
MIN_HEARD_BARS = 2
WEAK_BELOW_SHARE = 0.4
MOVING_CHORDS_SHARE = 0.5
MEDIUM_GROUP_BARS = 2
BLOCK_BARS = 4
MAX_SECTION_BARS = 16
LOW_ENERGY = 0.35
HIGH_ENERGY = 0.7
BASS_PITCH_LIMIT = 55
BASS_WEIGHT = 2.0
ROOT_BONUS = 1.25
MIN_BAR_WEIGHT = 0.05
MIN_SPREAD = 0.25
MIN_PHASE_CONFIDENCE = 0.3
MIN_OFFSET = 0.02
END_TOLERANCE = 0.02
DOWNBEAT_GAIN = 1.3
DOWNBEAT_WINDOW = 0.125
BEATS_PER_BAR = 4
MIN_GRID_ONSETS = 16
MIN_OFFSET_ONSETS = 4
GRID_SEARCH = 0.03
OFFSET_SEARCH = 0.005
MIN_SIGNIFICANCE = 8.0
OFFSET_DIVISIONS = (4, 2, 1)
GRID_TEMPO_STEP = 1e-4
COHERENCE_MARGIN = 0.02
MIN_QUARTER_SIGNIFICANCE = 3.0
VOTE_SIGMAS = 2.5
LOW_SIGMAS = 1.8
BAR_SHARE = 0.3
MAX_BASS_SHARE = 0.5
MIN_STACK = 0.05
MIN_STACKED = 0.3
MIN_PURITY = 0.85
KICK_PITCHES = (35, 36)
MIN_PAIRED_BARS = 8
PAIRED_SIGMAS = 3.5
EARLY_SECONDS = 0.08
MAX_SPAN = 4 * 3600.0


@dataclass
class SongAnalysis:
    bars: int
    degrees: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    sections: list = field(default_factory=list)


def validate_match(level):
    if level != AUTO_MATCH and level not in MATCH_LEVELS:
        raise ValueError(f"Unknown match level: {level}")
    return level


def _check_bpm(bpm):
    if (
        isinstance(bpm, bool)
        or not isinstance(bpm, numbers.Real)
        or not math.isfinite(bpm)
        or not MIN_BPM <= bpm <= MAX_BPM
    ):
        raise ValueError(
            f"bpm must be a number from {MIN_BPM:g} to {MAX_BPM:g}, got {bpm!r}"
        )
    return float(bpm)


def validate_offset(offset):
    if offset is None:
        return None
    if (
        isinstance(offset, bool)
        or not isinstance(offset, numbers.Real)
        or not math.isfinite(offset)
        or offset < 0
    ):
        raise ValueError(
            f"Grid offset must be a number of seconds from 0, got {offset!r}"
        )
    return float(offset)


def _overlap(note, lo, hi):
    return max(0.0, min(note.end, hi) - max(note.start, lo))


def bar_count(existing, bpm):
    bpm = _check_bpm(bpm)
    bar = 240.0 / bpm
    end = existing.get_end_time() if existing is not None else 0.0
    if end <= 0:
        return 0
    bars = math.ceil((end - END_TOLERANCE) / bar)
    return math.ceil(bars / BLOCK_BARS) * BLOCK_BARS


def _bar_degree(weights, root, intervals):
    total = sum(weights)
    if total < MIN_BAR_WEIGHT:
        return None
    size = len(intervals)
    best, best_score = None, 0.0
    for degree in range(size):
        triad = [
            (root + intervals[(degree + step) % size]) % 12
            for step in (0, 2, 4)
        ]
        score = sum(weights[pc] for pc in triad)
        score += (ROOT_BONUS - 1.0) * weights[triad[0]]
        if score > best_score:
            best, best_score = degree, score
    return best


def _fill(degrees):
    out = []
    last = next((d for d in degrees if d is not None), 0)
    for degree in degrees:
        if degree is not None:
            last = degree
        out.append(last)
    return out


def _group_majority(degrees, group):
    out = []
    for start in range(0, len(degrees), group):
        chunk = degrees[start : start + group]
        chosen = Counter(chunk).most_common(1)[0][0]
        out.extend([chosen] * len(chunk))
    return out


def _level(value):
    if value < LOW_ENERGY:
        return "low"
    if value < HIGH_ENERGY:
        return "mid"
    return "high"


def _plan_parts(block_energy):
    low, high = min(block_energy), max(block_energy)
    spread = high - low
    runs = []
    for index, value in enumerate(block_energy):
        level = "mid" if spread < MIN_SPREAD else _level((value - low) / spread)
        if runs and runs[-1][0] == level:
            runs[-1][2] += BLOCK_BARS
        else:
            runs.append([level, index * BLOCK_BARS, BLOCK_BARS])
    sections = []
    for position, (level, start, bars) in enumerate(runs):
        if level == "low":
            if position == 0:
                name = "Intro"
            elif position == len(runs) - 1:
                name = "Outro"
            else:
                name = "Break"
        else:
            name = "Verse" if level == "mid" else "Chorus"
        cursor = start
        remaining = bars
        while remaining > 0:
            part = min(MAX_SECTION_BARS, remaining)
            sections.append((name, cursor, part))
            cursor += part
            remaining -= part
    return sections


def _bar_weights(notes, drums, bars, bar):
    weights = [[0.0] * 12 for _ in range(bars)]
    energy = [0.0] * bars
    for n in notes:
        first = max(0, int(n.start // bar))
        last = min(bars - 1, int(max(n.start, n.end - 1e-9) // bar))
        for b in range(first, last + 1):
            seconds = _overlap(n, b * bar, (b + 1) * bar)
            if seconds <= 0:
                continue
            w = seconds * n.velocity / 127.0
            if n.pitch < BASS_PITCH_LIMIT:
                w *= BASS_WEIGHT
            weights[b][n.pitch % 12] += w
            energy[b] += w
    for n in drums:
        b = int(n.start // bar)
        if 0 <= b < bars:
            energy[b] += n.velocity / 127.0 * 0.25
    return weights, energy


def analyze_song(existing, bpm, root, intervals):
    bars = bar_count(existing, bpm)
    bar = 240.0 / bpm
    notes = [
        n
        for inst in existing.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    drums = [
        n for inst in existing.instruments if inst.is_drum for n in inst.notes
    ]
    weights, energy = _bar_weights(notes, drums, bars, bar)
    degrees = [_bar_degree(w, root, intervals) for w in weights]
    ranked = sorted(energy)
    top = ranked[int(len(ranked) * 0.9)] if ranked else 0.0
    if top > 0:
        energy = [min(1.0, e / top) for e in energy]
    blocks = [
        sum(energy[i : i + BLOCK_BARS]) / BLOCK_BARS
        for i in range(0, bars, BLOCK_BARS)
    ]
    sections = _plan_parts(blocks) if bars else []
    return SongAnalysis(bars, degrees, energy, sections)


def auto_match(analysis):
    heard = [i for i, d in enumerate(analysis.degrees) if d is not None]
    if len(heard) < MIN_HEARD_BARS:
        return "Weak"
    share = len(heard) / (heard[-1] - heard[0] + 1)
    if share < WEAK_BELOW_SHARE:
        return "Weak"
    pairs = [
        (first, second)
        for first, second in zip(
            analysis.degrees[0::MEDIUM_GROUP_BARS],
            analysis.degrees[1::MEDIUM_GROUP_BARS],
            strict=False,
        )
        if first is not None and second is not None
    ]
    moving = sum(first != second for first, second in pairs)
    if pairs and moving / len(pairs) >= MOVING_CHORDS_SHARE:
        return "Strong"
    return "Medium"


def progression_for(analysis, match):
    validate_match(match)
    if match == AUTO_MATCH:
        match = auto_match(analysis)
    if match == "Weak" or analysis.bars == 0:
        return None
    filled = _fill(analysis.degrees)
    if match == "Medium":
        return _group_majority(filled, MEDIUM_GROUP_BARS)
    return filled


def _best_triad(weights):
    if sum(weights) < MIN_BAR_WEIGHT:
        return None
    best, best_score = None, 0.0
    for root in range(12):
        for third in (4, 3):
            score = (
                weights[root] * ROOT_BONUS
                + weights[(root + third) % 12]
                + weights[(root + 7) % 12]
            )
            if score > best_score:
                best = (root, (root + third) % 12, (root + 7) % 12)
                best_score = score
    return best


def bar_triads(existing, bpm):
    bars = bar_count(existing, bpm)
    bar = 240.0 / bpm
    notes = [
        n
        for inst in existing.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    weights, _energy = _bar_weights(notes, [], bars, bar)
    return [_best_triad(w) for w in weights]


@dataclass(frozen=True)
class BeatGrid:
    bpm: float
    beat_phase: float
    bar_offset: float
    confidence: float


def _triad_matrix():
    matrix = np.zeros((12, 24))
    for root in range(12):
        for column, third in enumerate((4, 3)):
            for pc in (root, root + third, root + 7):
                matrix[pc % 12, root * 2 + column] = 1.0
    return matrix


TRIADS = _triad_matrix()
ROOTS = np.eye(12)


def _note_table(existing):
    rows = [
        (n.start, n.end, n.pitch, n.velocity, inst.is_drum)
        for inst in existing.instruments
        for n in inst.notes
    ]
    table = np.array(rows, dtype=float).reshape(-1, 5)
    return table[np.isfinite(table).all(axis=1)]


def _enough(table, minimum):
    if len(table) < max(1, minimum) or table[:, 3].sum() <= 0:
        return False
    return bool(np.ptp(table[:, 0]) <= MAX_SPAN)


def _wrap(value, period):
    return float(round(value, 9) % period)


def _best_tempo(times, weights, bpm, search, div):
    steps = int(round(search / GRID_TEMPO_STEP))
    ratios = 1 + GRID_TEMPO_STEP * np.arange(-steps, steps + 1)
    turn = 2j * np.pi * div * bpm / 60.0 * times
    phasor = np.exp(turn * ratios[0])
    advance = np.exp(turn * GRID_TEMPO_STEP)
    best, best_value = 0, 0j
    for index in range(len(ratios)):
        value = (phasor * weights).sum()
        if index == 0 or abs(value) > abs(best_value):
            best, best_value = index, value
        phasor *= advance
    return bpm * float(ratios[best]), best_value / weights.sum()


def _chord_coherence(table, origin, bar, shapes=TRIADS):
    start, end, pitch, velocity, drum = table.T
    pitched = drum == 0
    if not pitched.any():
        return 0.0
    start = start[pitched]
    end = np.clip(end[pitched], start + 1e-6, start.max() + bar)
    pitch, velocity = pitch[pitched].astype(int), velocity[pitched]
    first = np.floor((start - origin) / bar).astype(int)
    last = np.floor((end - origin - 1e-9) / bar).astype(int)
    low = first.min()
    bars = int(last.max() - low) + 1
    gain = np.where(pitch < BASS_PITCH_LIMIT, BASS_WEIGHT, 1.0)
    weight = velocity / 127.0 * gain

    def seconds(index):
        lo = origin + index * bar
        return np.maximum(np.minimum(end, lo + bar) - np.maximum(start, lo), 0)

    def tally(index, amount, keep=slice(None)):
        cell = (index - low) * 12 + pitch % 12
        return np.bincount(cell[keep], amount[keep], bars * 12)

    cells = tally(first, seconds(first) * weight)
    cells += tally(last, seconds(last) * weight, last > first)
    whole = last - first >= 2
    held = tally(first + 1, bar * weight, whole) - tally(
        last, bar * weight, whole
    )
    cells = cells.reshape(bars, 12) + np.cumsum(held.reshape(bars, 12), axis=0)
    total = cells.sum()
    return float((cells @ shapes).max(axis=1).sum() / total) if total else 0.0


def _stacked_share(table):
    pitched = table[table[:, 4] == 0]
    if len(pitched) < 2:
        return 0.0
    order = np.argsort(pitched[:, 0], kind="stable")
    start, end = pitched[order, 0], pitched[order, 1]
    reach = np.maximum.accumulate(end)[:-1]
    return float(np.mean(start[1:] < reach - MIN_STACK))


def _clearly_more(total, power, k, first, sigmas, gain=DOWNBEAT_GAIN):
    spread = math.sqrt(power[k] + power[first])
    return (
        total[k] > gain * total[first]
        and total[k] - total[first] >= sigmas * spread
    )


def _beat_votes(table, phase, beat, low_only):
    start, _end, pitch, velocity, drum = table.T
    bass = (drum == 0) & (pitch < BASS_PITCH_LIMIT)
    weight = velocity / 127.0
    if low_only:
        kick = (drum == 1) & np.isin(pitch, KICK_PITCHES)
        part = bass.sum() < MAX_BASS_SHARE * (drum == 0).sum()
        weight = np.where(kick | (bass & part), weight * BASS_WEIGHT, 0.0)
    else:
        weight = np.where(bass, weight * BASS_WEIGHT, weight)
    position = (start - phase) / beat
    nearest = np.round(position)
    close = (np.abs(position - nearest) <= DOWNBEAT_WINDOW) & (weight > 0)
    index = (nearest[close] % BEATS_PER_BAR).astype(int)
    bars = np.floor(nearest[close] / BEATS_PER_BAR)
    rows = (bars - bars.min(initial=0)).astype(int)
    pitched = np.where(drum[close] == 0, weight[close], 0.0)
    per_bar = np.bincount(
        rows * BEATS_PER_BAR + index,
        weights=pitched,
        minlength=BEATS_PER_BAR * (rows.max(initial=0) + 1),
    ).reshape(-1, BEATS_PER_BAR)
    return (
        np.bincount(index, weights=weight[close], minlength=BEATS_PER_BAR),
        np.bincount(index, weights=weight[close] ** 2, minlength=BEATS_PER_BAR),
        [len(np.unique(bars[index == k])) for k in range(BEATS_PER_BAR)],
        per_bar,
    )


def _steadily_more(per_bar, k, first):
    voted = per_bar[(per_bar[:, k] > 0) | (per_bar[:, first] > 0)]
    more = per_bar[:, k].sum() > DOWNBEAT_GAIN * per_bar[:, first].sum()
    if len(voted) < MIN_PAIRED_BARS or not more:
        return False
    gap = voted[:, k] - voted[:, first]
    return gap.mean() * math.sqrt(len(voted)) >= PAIRED_SIGMAS * gap.std(ddof=1)


def _bar_beat(table, phase, beat):
    stacked = _stacked_share(table) >= MIN_STACKED
    shapes = TRIADS if stacked else ROOTS

    def coherence(k, beats):
        return _chord_coherence(table, phase + k * beat, beats * beat, shapes)

    scores = [coherence(k, BEATS_PER_BAR) for k in range(BEATS_PER_BAR)]
    tied = list(range(BEATS_PER_BAR))
    if stacked or max(scores) >= MIN_PURITY:
        top = max(scores)
        half = max(coherence(k, BEATS_PER_BAR // 2) for k in range(2))
        if half - top <= COHERENCE_MARGIN:
            tied = [k for k in tied if scores[k] >= top - COHERENCE_MARGIN]
            if len(tied) == 1:
                return tied[0]
    span = np.ptp(table[:, 0]) / (BEATS_PER_BAR * beat) + 1
    for low_only, sigmas in ((True, LOW_SIGMAS), (False, VOTE_SIGMAS)):
        total, power, bars, per_bar = _beat_votes(table, phase, beat, low_only)
        clear = [
            k
            for k in tied[1:]
            if (
                _clearly_more(total, power, k, tied[0], sigmas)
                or (not low_only and _steadily_more(per_bar, k, tied[0]))
            )
            and bars[k] >= BAR_SHARE * span
        ]
        if clear:
            best = max(clear, key=lambda k: total[k])
            return next(
                k
                for k in clear
                if total[best] - total[k] < math.sqrt(power[best] + power[k])
            )
    return tied[0]


def _densest_point(table, phase, beat, div):
    from korshmer.midi.adaptive_snap import BEAT_MARGIN, TOLERANCE_BEATS

    gap = (table[:, 0] - phase) / beat - np.arange(div)[:, None] / div
    near = np.abs(gap - np.round(gap)) <= TOLERANCE_BEATS
    weight = table[:, 3] / 127.0
    total, power = near @ weight, near @ weight**2
    best = int(np.argmax(total))
    if best and _clearly_more(total, power, best, 0, VOTE_SIGMAS, BEAT_MARGIN):
        return phase + best * beat / div
    return phase


def _early(beat):
    return max(DOWNBEAT_WINDOW * beat, EARLY_SECONDS)


def _grid_at(existing, table, tempo, value, div):
    from korshmer.midi.adaptive_snap import find_beat_offset

    beat = 60.0 / tempo
    phase16 = float(np.angle(value)) / (2 * np.pi) * beat / div
    found = find_beat_offset(existing, tempo, phase16, div)
    if div > 1 and found == phase16:
        found = _densest_point(table, phase16, beat, div)
    raw = _wrap(found, beat)
    if beat - raw <= _early(beat):
        raw -= beat
    raw += _bar_beat(table, raw, beat) * beat
    phase = _wrap(raw, beat)
    offset = _wrap(raw, BEATS_PER_BAR * beat)
    return BeatGrid(tempo, phase, offset, float(abs(value)))


def estimate_grid(existing, bpm_hint, *, search=GRID_SEARCH, div=4):
    bpm = _check_bpm(bpm_hint)
    if search < 0 or div < 1:
        raise ValueError(
            f"search must be 0 or more, div at least 1, got {search!r}, {div!r}"
        )
    if existing is None:
        return None
    table = _note_table(existing)
    if not _enough(table, MIN_GRID_ONSETS):
        return None
    tempo, value = _best_tempo(
        table[:, 0], table[:, 3] / 127.0, bpm, search, div
    )
    return _grid_at(existing, table, tempo, value, div)


def fold_offset(offset, bpm, early=MIN_OFFSET):
    bar = BEATS_PER_BAR * 60.0 / bpm
    offset %= bar
    if offset < MIN_OFFSET or bar - offset < early:
        return 0.0
    return offset


def _fits_for(count):
    if count < MIN_GRID_ONSETS:
        return [(0, 1)]
    return [(s, d) for s in (0, OFFSET_SEARCH) for d in OFFSET_DIVISIONS]


def _confident_fit(table, bpm):
    times, weights = table[:, 0], table[:, 3] / 127.0
    count = len(table)
    for search, div in _fits_for(count):
        tempo, value = _best_tempo(times, weights, bpm, search, div)
        size = abs(value)
        needed = (
            MIN_QUARTER_SIGNIFICANCE
            if (search, div) == (0, 1)
            else MIN_SIGNIFICANCE
        )
        if size >= MIN_PHASE_CONFIDENCE and count * size * size >= needed:
            return tempo, value, div
    return None


def estimate_offset(existing, bpm):
    bpm = _check_bpm(bpm)
    if existing is None:
        return 0.0
    table = _note_table(existing)
    if not _enough(table, MIN_OFFSET_ONSETS):
        return 0.0
    fit = _confident_fit(table, bpm)
    if fit is None:
        return 0.0
    tempo, value, div = fit
    grid = _grid_at(existing, table, tempo, value, div)
    middle = float(np.average(table[:, 0], weights=table[:, 3]))
    drift = (middle - grid.bar_offset) * (1 - tempo / bpm)
    return fold_offset(grid.bar_offset + drift, bpm, _early(60.0 / bpm))


def occupied_bars(existing, bpm):
    bpm = _check_bpm(bpm)
    bar = 240.0 / bpm
    notes = [n for inst in existing.instruments for n in inst.notes]
    if not notes:
        return 0, 0
    first = int(min(n.start for n in notes) // bar)
    end = math.ceil((max(n.end for n in notes) - END_TOLERANCE) / bar)
    return first, max(end, first + 1)


def shift_song(existing, offset):
    if offset <= 0:
        return existing
    import pretty_midi

    out = pretty_midi.PrettyMIDI()
    for inst in existing.instruments:
        copy = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name
        )
        for n in inst.notes:
            if n.end - offset <= 0:
                continue
            copy.notes.append(
                pretty_midi.Note(
                    n.velocity,
                    n.pitch,
                    max(0.0, n.start - offset),
                    n.end - offset,
                )
            )
        out.instruments.append(copy)
    return out


LEAD_MIN_NOTES = 8
LEAD_MAX_OVERLAP = 0.2
LEAD_ONSET_WINDOW = 0.03


def _overlap_share(notes):
    ordered = sorted(notes, key=lambda n: n.start)
    stacked = sum(
        1
        for a, b in zip(ordered, ordered[1:], strict=False)
        if b.start < a.end - 1e-6
    )
    return stacked / max(1, len(ordered) - 1)


def _top_line(notes):
    top = []
    group = []
    for note in sorted(notes, key=lambda n: n.start):
        if group and note.start - group[0].start >= LEAD_ONSET_WINDOW:
            top.append(max(group, key=lambda n: n.pitch))
            group = []
        group.append(note)
    if group:
        top.append(max(group, key=lambda n: n.pitch))
    return top


def _named_lead(existing, name):
    lines = [
        _top_line(inst.notes)
        for inst in existing.instruments
        if not inst.is_drum and name in (inst.name or "").lower()
    ]
    lines = [
        line
        for line in lines
        if len(line) >= LEAD_MIN_NOTES
        and _overlap_share(line) <= LEAD_MAX_OVERLAP
    ]
    return max(lines, key=len, default=[])


_NUMBERED = re.compile(r"\s+\d+$")


def _voices(existing):
    voices = {}
    for index, inst in enumerate(existing.instruments):
        if inst.is_drum:
            continue
        family = _NUMBERED.sub("", (inst.name or "").strip().lower())
        voices.setdefault(family or index, []).extend(inst.notes)
    return list(voices.values())


def find_lead(existing, prefer="vocal"):
    if existing is None:
        return []
    named = _named_lead(existing, prefer) if prefer else []
    if named:
        return named
    best, best_pitch = [], -1.0
    for notes in _voices(existing):
        if len(notes) < LEAD_MIN_NOTES:
            continue
        if _overlap_share(notes) > LEAD_MAX_OVERLAP:
            continue
        pitches = sorted(n.pitch for n in notes)
        median = pitches[len(pitches) // 2]
        if median > best_pitch:
            best, best_pitch = notes, median
    return list(best)
