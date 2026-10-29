import pretty_midi

from korshmer.midi_algorithms import validate_beat_grid_args

DIVISIONS = (1, 2, 4)
SHORT_BEATS = 0.4
LONG_BEATS = 1.0
TOLERANCE_BEATS = 0.08
TOLERANCE_SECONDS = 0.06
SWING_SHARE = 0.33
MIN_NOTE = 0.01
OVERLAP_GAP = 0.005
BEAT_EVIDENCE = 4.0
BEAT_MARGIN = 1.5

KICK_AND_SNARE = frozenset({35, 36, 37, 38, 39, 40})
LABELS = {1: "1/4", 2: "1/8", 4: "1/16"}


def supports(grid_div):
    return grid_div in (1, 2, 4)


def finest_for(grid_div):
    return min(int(grid_div), DIVISIONS[-1])


def allowed_divisions(finest):
    steps = tuple(d for d in DIVISIONS if d <= finest)
    return steps or None


def _candidates(note, is_drum, beats, allowed):
    if is_drum:
        wanted = (
            DIVISIONS[1:] if note.pitch in KICK_AND_SNARE else DIVISIONS[:-1]
        )
    elif beats >= LONG_BEATS:
        wanted = DIVISIONS[:-1]
    elif beats >= SHORT_BEATS:
        wanted = DIVISIONS[1:]
    else:
        wanted = DIVISIONS[:-1]
    chosen = [d for d in wanted if d in allowed]
    return chosen or [allowed[-1]]


def choose_division(note, is_drum, tempo, offset, allowed, swing=0.0):
    beat = 60.0 / tempo
    position = (note.start - offset) / beat
    tolerance = min(TOLERANCE_BEATS, TOLERANCE_SECONDS / beat)
    order = _candidates(note, is_drum, (note.end - note.start) / beat, allowed)
    chosen = order[-1]
    for division in order:
        if abs(position - round(position * division) / division) <= tolerance:
            chosen = division
            break
    index = round(position * chosen)
    target = offset + index * beat / chosen
    if chosen > 1 and index % 2 != 0:
        target += beat / chosen * SWING_SHARE * swing
    return chosen, target


def choose_end(note, start, tempo, offset, allowed):
    beat = 60.0 / tempo
    position = (note.end - offset) / beat
    tolerance = min(TOLERANCE_BEATS, TOLERANCE_SECONDS / beat)
    length = (note.end - note.start) / beat
    order = _candidates(note, False, length, allowed)
    chosen = order[-1]
    for division in order:
        if abs(position - round(position * division) / division) <= tolerance:
            chosen = division
            break
    target = offset + round(position * chosen) * beat / chosen
    duration = note.end - note.start
    floor = start + max(MIN_NOTE, min(duration, beat / allowed[-1]))
    return max(target, floor)


def find_beat_offset(midi_obj, tempo, offset, phase_div):
    steps = int(phase_div)
    if steps <= 1:
        return offset
    beat = 60.0 / tempo
    scores = [0.0] * steps
    for inst in midi_obj.instruments:
        for note in inst.notes:
            length = (note.end - note.start) / beat
            if inst.is_drum:
                if note.pitch not in KICK_AND_SNARE:
                    continue
                weight = 1.0
            elif length >= SHORT_BEATS:
                weight = min(length, LONG_BEATS)
            else:
                continue
            position = (note.start - offset) / beat
            for shift in range(steps):
                gap = position - shift / steps
                if abs(gap - round(gap)) <= TOLERANCE_BEATS:
                    scores[shift] += weight
    best = max(range(steps), key=lambda shift: (scores[shift], -shift))
    if scores[best] < BEAT_EVIDENCE or scores[best] < BEAT_MARGIN * scores[0]:
        return offset
    return offset + best * beat / steps


def shared_beat_offset(midis, bpm, grid_offset, grid_divisions):
    merged = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for midi_obj in midis:
        merged.instruments.extend(midi_obj.instruments)
    return find_beat_offset(merged, bpm, grid_offset, grid_divisions)


def snap_adaptive(
    midi_obj,
    tempo,
    offset,
    finest=4,
    strength=1.0,
    swing=0.0,
    phase_div=None,
    snap_drums=True,
):
    allowed = allowed_divisions(finest)
    counts = {LABELS[d]: 0 for d in (allowed or ())}
    if not allowed or not tempo or tempo <= 0:
        return counts
    validate_beat_grid_args(tempo, offset, finest, strength, swing)
    offset = find_beat_offset(midi_obj, tempo, offset, phase_div or finest)
    for inst in midi_obj.instruments:
        # Drums are transcribed at the onsets actually heard in the source.
        # When snap_drums is False keep those onsets instead of moving them onto
        # the grid, so a converted drum part matches the original timing. Their
        # overlapping same-pitch notes are still trimmed, as when snapping.
        if inst.is_drum and not snap_drums:
            _trim_overlaps(inst)
            continue
        for note in inst.notes:
            duration = note.end - note.start
            division, target = choose_division(
                note, inst.is_drum, tempo, offset, allowed, swing
            )
            start = note.start * (1 - strength) + target * strength
            note.start = max(0.0, start)
            note.end = note.start + duration
            if not inst.is_drum:
                end = choose_end(note, note.start, tempo, offset, allowed)
                note.end = note.end * (1 - strength) + end * strength
                note.end = max(note.end, note.start + MIN_NOTE)
            counts[LABELS[division]] += 1
        _trim_overlaps(inst)
    return counts


def _merge_same_start(inst):
    kept = {}
    for note in sorted(inst.notes, key=lambda n: n.start):
        twin = kept.get((note.pitch, note.start))
        if twin is None:
            kept[(note.pitch, note.start)] = note
            continue
        twin.end = max(twin.end, note.end)
        twin.velocity = max(twin.velocity, note.velocity)
    inst.notes = list(kept.values())


def _trim_overlaps(inst):
    _merge_same_start(inst)
    by_pitch = {}
    for note in sorted(inst.notes, key=lambda n: n.start):
        by_pitch.setdefault(note.pitch, []).append(note)
    for notes in by_pitch.values():
        for current, following in zip(notes, notes[1:], strict=False):
            if current.end > following.start:
                current.end = max(
                    current.start + MIN_NOTE, following.start - OVERLAP_GAP
                )
