import copy

import numpy as np
import pretty_midi

from korshmer.analysis import rank_scales
from korshmer.constants import SCALES
from korshmer.midi.bar_lines import bar_lines
from korshmer.midi.polish import classify

STRENGTHS = {"Light": 0.3, "Medium": 0.5, "Strong": 0.7}
LONGEST_SHARE = 0.1
CHORD_WINDOW = 0.05
BASS_PITCH = 52
SYNCOPATION_BEAT_SHARE = 0.125
SPLIT_MIN_BEATS = 1.5
PASSING_MIN_BEATS = 1.0
PASSING_MIN_STEPS = 3
MIN_DURATION = 0.05
VELOCITY_SWING = 8
STEP_CHOICES = (-2, -1, 1, 2)
STEP_WEIGHTS = (0.15, 0.35, 0.35, 0.15)


def _scale_pitch_classes(pm):
    chroma = np.zeros(12)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            chroma[note.pitch % 12] += note.end - note.start
    if not np.any(chroma):
        return list(range(12))
    root, name, _score = rank_scales(chroma / np.max(chroma), 1)[0]
    intervals = SCALES.get(name, SCALES["Major"])
    return sorted((root + step) % 12 for step in intervals)


def _scale_pitches(pitch_classes):
    return [p for p in range(128) if p % 12 in pitch_classes]


def _nearest_index(scale, pitch):
    return int(np.argmin([abs(p - pitch) for p in scale]))


def _move(scale, pitch, steps, low, high):
    index = _nearest_index(scale, pitch) + steps
    index = min(len(scale) - 1, max(0, index))
    moved = scale[index]
    if moved < low or moved > high or moved == pitch:
        return pitch
    return moved


def _bar_starts(pm):
    try:
        downbeats = [float(t) for t in bar_lines(pm)]
    except Exception:
        downbeats = []
    return downbeats or [0.0]


def _first_of_bar_ids(notes, bar_starts):
    firsts = {}
    for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
        bar = int(np.searchsorted(bar_starts, note.start + 1e-6, "right")) - 1
        firsts.setdefault(bar, id(note))
    return {note_id for bar, note_id in firsts.items() if bar % 2 == 0}


def _anchor_ids(notes, bar_starts):
    anchors = _first_of_bar_ids(notes, bar_starts)
    ranked = sorted(notes, key=lambda n: n.end - n.start, reverse=True)
    keep = max(1, int(round(len(ranked) * LONGEST_SHARE))) if ranked else 0
    anchors.update(id(n) for n in ranked[:keep])
    return anchors


def _chord_groups(notes):
    groups = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if groups and note.start - groups[-1][0].start < CHORD_WINDOW:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def _vary_pitches(notes, quota, rng, scale):
    if not notes or quota <= 0:
        return 0
    low = min(n.pitch for n in notes) - 2
    high = max(n.pitch for n in notes) + 2
    changed = 0
    for index in rng.permutation(len(notes))[:quota]:
        note = notes[index]
        for steps in rng.choice(STEP_CHOICES, size=2, p=STEP_WEIGHTS):
            moved = _move(scale, note.pitch, int(steps), low, high)
            if moved != note.pitch:
                note.pitch = moved
                changed += 1
                break
    return changed


def _vary_chords(groups, quota, rng):
    changed = 0
    order = rng.permutation(len(groups))
    for index in order:
        if changed >= quota:
            break
        group = groups[index]
        if len(group) < 2:
            continue
        pitched = sorted(group, key=lambda n: n.pitch)
        if rng.random() < 0.5 and pitched[0].pitch + 12 <= 127:
            pitched[0].pitch += 12
            changed += 1
        elif pitched[-1].pitch - 12 >= 0:
            pitched[-1].pitch -= 12
            changed += 1
    return changed


def _syncopate(inst, notes, quota, rng, beat):
    shifted = 0
    ordered = sorted(inst.notes, key=lambda n: n.start)
    positions = {id(n): i for i, n in enumerate(ordered)}
    shift = beat * SYNCOPATION_BEAT_SHARE
    for index in rng.permutation(len(notes))[:quota]:
        note = notes[index]
        duration = note.end - note.start
        if duration < MIN_DURATION:
            continue
        position = positions[id(note)]
        previous_end = ordered[position - 1].end if position > 0 else 0.0
        following_start = (
            ordered[position + 1].start if position + 1 < len(ordered) else None
        )
        forward_room = (
            following_start - note.end if following_start is not None else shift
        )
        backward_room = note.start - previous_end
        choices = []
        if forward_room >= MIN_DURATION:
            choices.append(min(shift, forward_room))
        if backward_room >= MIN_DURATION:
            choices.append(-min(shift, backward_room))
        if not choices:
            continue
        if len(choices) == 1:
            delta = choices[0]
        else:
            delta = choices[0] if rng.random() < 0.5 else choices[1]
        note.start += delta
        note.end += delta
        shifted += 1
    return shifted


def _split_long(inst, notes, quota, rng, beat):
    added = []
    for index in rng.permutation(len(notes))[:quota]:
        note = notes[index]
        if note.end - note.start < beat * SPLIT_MIN_BEATS:
            continue
        middle = note.start + (note.end - note.start) / 2.0
        added.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=middle,
                end=note.end,
            )
        )
        note.end = middle - 0.01
    inst.notes.extend(added)
    return len(added)


def _add_passing(inst, scale, rng, beat, limit):
    ordered = sorted(inst.notes, key=lambda n: n.start)
    added = []
    for first, second in zip(ordered, ordered[1:], strict=False):
        if len(added) >= limit:
            break
        gap = second.start - first.end
        if gap < beat * PASSING_MIN_BEATS:
            continue
        steps = abs(
            _nearest_index(scale, second.pitch)
            - _nearest_index(scale, first.pitch)
        )
        if steps < PASSING_MIN_STEPS:
            continue
        middle = (first.pitch + second.pitch) // 2
        pitch = scale[_nearest_index(scale, middle)]
        start = first.end + gap / 2.0 - beat / 4.0
        added.append(
            pretty_midi.Note(
                velocity=max(1, min(127, first.velocity - 10)),
                pitch=int(pitch),
                start=start,
                end=start + beat / 2.0,
            )
        )
    inst.notes.extend(added)
    return len(added)


def _humanize(notes, rng):
    for note in notes:
        swing = int(rng.integers(-VELOCITY_SWING, VELOCITY_SWING + 1))
        note.velocity = int(max(1, min(127, note.velocity + swing)))


def create_variation(midi_data, strength="Medium", seed=0):
    if strength not in STRENGTHS:
        raise ValueError(f"Unknown strength: {strength}")
    pm = copy.deepcopy(midi_data)
    rng = np.random.default_rng(int(seed))
    target = STRENGTHS[strength]
    scale = _scale_pitches(_scale_pitch_classes(pm))
    bar_starts = _bar_starts(pm)
    tempos = pm.get_tempo_changes()[1]
    beat = 60.0 / float(tempos[0]) if len(tempos) and tempos[0] > 0 else 0.5
    report = {
        "notes_before": 0,
        "notes_changed": 0,
        "notes_added": 0,
        "anchors_kept": 0,
        "tracks_changed": 0,
    }
    for inst in pm.instruments:
        if inst.is_drum or not inst.notes:
            continue
        original = list(inst.notes)
        report["notes_before"] += len(original)
        anchors = _anchor_ids(original, bar_starts)
        chords = classify(inst) == "chords"
        groups = _chord_groups(original) if chords else []
        free_groups = [
            g for g in groups if not any(id(n) in anchors for n in g)
        ]
        if chords:
            free = [n for g in free_groups for n in g]
        else:
            free = [n for n in original if id(n) not in anchors]
        report["anchors_kept"] += len(original) - len(free)
        quota = min(len(free), int(round(len(original) * target)))
        before = {id(n): (n.pitch, n.start, n.end) for n in original}
        added = 0
        if chords:
            _vary_chords(free_groups, quota, rng)
        else:
            low_voice = np.median([n.pitch for n in original]) < BASS_PITCH
            pitch_quota = quota // 2 if low_voice else quota
            _vary_pitches(free, pitch_quota, rng, scale)
            rhythm_quota = quota - pitch_quota or max(1, quota // 3)
            _syncopate(inst, free, rhythm_quota, rng, beat)
            added += _split_long(inst, free, max(1, quota // 4), rng, beat)
            added += _add_passing(inst, scale, rng, beat, max(1, quota // 4))
        _humanize(inst.notes, rng)
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        moved = sum(
            1 for n in original if (n.pitch, n.start, n.end) != before[id(n)]
        )
        report["notes_changed"] += moved
        report["notes_added"] += added
        report["tracks_changed"] += 1 if moved or added else 0
    return pm, report
