from dataclasses import dataclass

from korshmer.midi_generation import genre_feel, moods

STEPS = 16
STRONG_ONSETS = (0, 8)
HEAD_END = 8
CHORD_STEPS = (-3, 0, 2, 4)
LOW_STEP, HIGH_STEP = -3, 5
LEAP = 3
CANDIDATES = 96
SKELETON_BEAT = 4
SKELETON_LENGTH = 3
ANSWER_HEAD = {"half": 3, "close": 2}
ANSWER_LENGTH = 8
MOOD_NOTES = {name: feel.notes for name, feel in moods.MOOD_FEELS.items()}
MIN_NOTES, MAX_NOTES = 3, 8
VARIATIONS = ("tail_up", "tail_down", "split", "push", "gap")
SHAPE_WEIGHT = 1.5
BROKEN_HOOK = 100.0
PEDAL_SHARE = 0.5

PHRASES = {
    "chorus": (
        ("hook", "hook", "vary", "half"),
        ("hook", "hook", "lift", "close"),
    ),
    "verse": (
        ("skeleton", "skeleton", "vary", "half"),
        ("skeleton", "skeleton", "hook", "half"),
    ),
    "intro": (
        ("hook", "hook", "skeleton", "half"),
        ("hook", "hook", "vary", "half"),
    ),
    "outro": (
        ("hook", "skeleton", "hook", "half"),
        ("hook", "hook", "skeleton", "close"),
    ),
    "break": (
        ("aug", "aug", "inv", "half"),
        ("aug", "inv", "lift", "half"),
    ),
}
PHRASES["bridge"] = PHRASES["break"]
PHRASE_CHOICES = {
    "chorus": (
        PHRASES["chorus"],
        (
            ("hook", "hook", "lift", "half"),
            ("hook", "hook", "vary", "close"),
        ),
        (
            ("hook", "hook", "lift", "half"),
            ("hook", "hook", "inv", "close"),
        ),
        (
            ("hook", "hook", "vary", "half"),
            ("hook", "hook", "inv", "close"),
        ),
    ),
    "verse": (
        PHRASES["verse"],
        (
            ("skeleton", "skeleton", "hook", "half"),
            ("skeleton", "skeleton", "vary", "half"),
        ),
        (
            ("skeleton", "hook", "skeleton", "half"),
            ("skeleton", "hook", "vary", "half"),
        ),
        (
            ("skeleton", "skeleton", "hook", "half"),
            ("skeleton", "skeleton", "hook", "half"),
        ),
    ),
    "intro": (
        PHRASES["intro"],
        (
            ("hook", "hook", "vary", "half"),
            ("hook", "hook", "skeleton", "half"),
        ),
        (
            ("hook", "skeleton", "hook", "half"),
            ("hook", "hook", "vary", "half"),
        ),
    ),
    "outro": (
        PHRASES["outro"],
        (
            ("hook", "hook", "skeleton", "half"),
            ("hook", "skeleton", "hook", "close"),
        ),
    ),
    "break": (
        PHRASES["break"],
        (
            ("aug", "inv", "aug", "half"),
            ("aug", "aug", "lift", "half"),
        ),
        (
            ("aug", "aug", "lift", "half"),
            ("aug", "inv", "aug", "half"),
        ),
    ),
}
PHRASE_BARS = 4
FIXED_ROLES = ("hook", "lift", "inv", "skeleton", "aug")
TAIL_TURNS = ("invert", "step")
TAIL_TURN_COUNT = 1
TAIL_TRIES = 24
TAIL_STEPS = (-2, -1, 1, 2)


@dataclass(frozen=True)
class Hook:
    notes: tuple
    gate: float
    register: int


def strong(onset, length):
    return onset in STRONG_ONSETS or length >= 4


def phrase_plan(rng, own=None):
    plan = {kind: rng.choice(found) for kind, found in PHRASE_CHOICES.items()}
    plan["bridge"] = plan["break"]
    plan.update(own or {})
    return plan


def _shift_rhythm(rng, rhythm, chance):
    if rng.random() >= chance:
        return rhythm
    notes = [list(note) for note in rhythm]
    movable = [
        i
        for i, (onset, _) in enumerate(notes)
        if i and onset % 2 == 0 and onset not in STRONG_ONSETS
    ]
    rng.shuffle(movable)
    for i in movable:
        onset, length = notes[i]
        before = notes[i - 1]
        if rng.random() < 0.5 and length >= 2:
            notes[i] = [onset + 1, length - 1]
        elif onset - 1 > before[0]:
            before[1] = min(before[1], onset - 1 - before[0])
            notes[i] = [onset - 1, length + 1]
        else:
            continue
        break
    return tuple(tuple(note) for note in notes)


def _thin(rng, notes, sustain):
    weak = [
        i for i, (onset, _) in enumerate(notes) if i and onset % SKELETON_BEAT
    ] or [i for i in range(1, len(notes)) if notes[i][0] not in STRONG_ONSETS]
    if not weak:
        return False
    short = min(notes[i][1] for i in weak)
    i = rng.choice([i for i in weak if notes[i][1] == short])
    onset, length = notes.pop(i)
    if sustain:
        notes[i - 1][1] = onset + length - notes[i - 1][0]
    return True


def _split(rng, notes):
    longest = max(length for _, length in notes)
    if longest < 2:
        return False
    i = rng.choice([i for i, n in enumerate(notes) if n[1] == longest])
    onset, length = notes[i]
    first = (length + 1) // 2
    notes[i : i + 1] = [[onset, first], [onset + first, length - first]]
    return True


def fit_density(rng, rhythm, density, spec, sustain=True):
    low = max(MIN_NOTES, min(spec.min_notes, len(rhythm)))
    high = min(MAX_NOTES, max(spec.max_notes, low))
    target = max(low, min(high, round(len(rhythm) * density)))
    notes = [list(note) for note in rhythm]
    while len(notes) > target and _thin(rng, notes, sustain):
        pass
    while len(notes) < target and _split(rng, notes):
        pass
    return tuple(tuple(note) for note in notes)


def _draw(rng, rhythm, spec, mood=None):
    mood = mood or moods.mood_for(None)
    repeat = max(0.0, min(0.6, spec.repeat + mood.repeat))
    steps = []
    current = 0
    for index, (onset, length) in enumerate(rhythm):
        if index == 0:
            current = rng.choice((0, 0, 2, 4))
        else:
            roll = rng.random()
            if roll < repeat:
                move = 0
            elif roll < repeat + mood.leap:
                move = rng.choice(spec.leaps) * rng.choice(mood.leap_turn)
            else:
                move = rng.choice(mood.moves)
            current = max(LOW_STEP, min(HIGH_STEP, current + move))
        if strong(onset, length):
            current = min(
                CHORD_STEPS, key=lambda c: (abs(c - current), rng.random())
            )
        steps.append(current)
    return steps


def _turns(moves):
    signs = [1 if m > 0 else -1 for m in moves if m]
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def shape_fit(shape, steps):
    first, last = steps[0], steps[-1]
    inner = range(1, len(steps) - 1)
    if shape == "rise":
        return (last > first) - (last < first)
    if shape == "fall":
        return (last < first) - (last > first)
    if shape == "arch":
        peak = max(steps)
        inside = steps.index(peak) in inner and peak > max(first, last)
        return 1 if inside else -1
    if shape == "valley":
        dip = min(steps)
        inside = steps.index(dip) in inner and dip < min(first, last)
        return 1 if inside else -1
    moves = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    if shape == "wave":
        return 1 if _turns(moves) >= 2 else -1
    if shape == "pedal":
        share = max(steps.count(s) for s in set(steps)) / len(steps)
        return 1 if share >= PEDAL_SHARE else -1
    return 0


def sticks(steps, spec):
    moves = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    leaps = sum(1 for m in moves if abs(m) >= LEAP)
    return (
        leaps <= 2
        and (0 in moves or leaps == 1)
        and 2 <= len(set(steps)) <= 6
        and max(steps) - min(steps) <= spec.span[1] + 1
    )


def score_hook(rhythm, steps, spec, mood):
    feel = moods.mood_for(mood)
    moves = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    repeats = sum(1 for m in moves if m == 0)
    leaps = [i for i, m in enumerate(moves) if abs(m) >= LEAP]
    span = max(steps) - min(steps)
    turns = _turns(moves)
    wanted = max(0.0, min(0.6, spec.repeat + feel.repeat))
    score = -3.0 * abs(repeats / len(moves) - wanted)
    if len(leaps) == feel.leaps_liked[0]:
        score += 2.0
    elif len(leaps) in feel.leaps_liked:
        score += 1.2
    elif not leaps:
        score += 0.3
    else:
        score -= len(leaps) - 1
    for i in leaps:
        if i + 1 < len(moves) and moves[i] * moves[i + 1] < 0:
            if abs(moves[i + 1]) < LEAP:
                score += 1.0
    score += 1.0 if turns in (1, 2) else 0.2 if turns == 0 else -0.5 * turns
    low, high = spec.span
    score -= 0.8 * max(low - span, span - high, 0)
    score -= 0.5 * abs(len(set(steps)) - 4)
    if strong(*rhythm[steps.index(max(steps))]):
        score += 0.7
    if steps[-1] != steps[0]:
        score += 0.4
    score -= 0.25 * abs(len(rhythm) - feel.notes)
    if mood == "Melancholic" and steps[-1] <= steps[0]:
        score += 0.8
    elif mood == "Uplifting" and steps[-1] >= steps[0]:
        score += 0.6
    elif mood == "Dreamy" and leaps:
        score += 0.5
    return score


def make_hook(rng, style="Pop", mood="Uplifting"):
    spec = genre_feel.feel_for(style).hook
    mood = mood if mood in MOOD_NOTES else moods.DEFAULT_MOOD
    feel = moods.mood_for(mood)
    shape = rng.choice(feel.shapes)
    best, best_score = None, None
    for _ in range(CANDIDATES):
        rhythm = _shift_rhythm(rng, rng.choice(spec.rhythms), spec.shift)
        rhythm = fit_density(rng, rhythm, feel.density, spec, feel.sustain)
        steps = _draw(rng, rhythm, spec, feel)
        score = score_hook(rhythm, steps, spec, mood) + rng.random() * 0.4
        score += SHAPE_WEIGHT * shape_fit(shape, steps)
        if not sticks(steps, spec):
            score -= BROKEN_HOOK
        if best_score is None or score > best_score:
            best, best_score = (rhythm, steps), score
    rhythm, steps = best
    notes = tuple(
        (onset, length, step)
        for (onset, length), step in zip(rhythm, steps, strict=True)
    )
    return Hook(notes, spec.gate, spec.register)


def _tail_start(notes):
    start = next(
        (i for i, note in enumerate(notes) if note[0] >= HEAD_END),
        len(notes) // 2,
    )
    return max(1, min(start, len(notes) - 1))


def _turn_tail(rng, notes, start, turn):
    if turn == "invert":
        pivot = notes[start][2]
        for note in notes[start:]:
            note[2] = max(LOW_STEP, min(HIGH_STEP, 2 * pivot - note[2]))
    else:
        move = rng.choice(TAIL_STEPS)
        for note in notes[start:]:
            note[2] = max(LOW_STEP, min(HIGH_STEP, note[2] + move))


def vary_hook(rng, hook, style="Pop"):
    spec = genre_feel.feel_for(style).hook
    original = [list(note) for note in hook.notes]
    if len(original) < 2:
        return hook
    start = _tail_start(original)
    for _ in range(TAIL_TRIES):
        notes = [list(note) for note in original]
        for turn in rng.sample(TAIL_TURNS, TAIL_TURN_COUNT):
            _turn_tail(rng, notes, start, turn)
        for note in notes[start:]:
            if strong(note[0], note[1]):
                note[2] = min(CHORD_STEPS, key=lambda c: abs(c - note[2]))
        steps = [note[2] for note in notes]
        if notes != original and sticks(steps, spec):
            return Hook(
                tuple(tuple(note) for note in notes), hook.gate, hook.register
            )
    return hook


def _skeleton(notes):
    keep = [
        n for n in notes if n[0] % SKELETON_BEAT == 0 or n[1] >= SKELETON_LENGTH
    ]
    if len(keep) < 2:
        keep = [notes[0], notes[-1]] if len(notes) > 1 else list(notes)
    out = []
    for i, (onset, length, step) in enumerate(keep):
        edge = keep[i + 1][0] if i + 1 < len(keep) else STEPS
        out.append((onset, min(edge - onset, length * 2), step))
    return out


def _augment(notes):
    head = [n for n in notes if n[0] < HEAD_END] or list(notes[:1])
    return [
        (o * 2, min(l * 2, STEPS - o * 2), s)
        for o, l, s in head
        if o * 2 < STEPS
    ]


def _vary(rng, notes):
    out = list(notes)
    last = len(out) - 1
    kind = rng.choice(VARIATIONS)
    if kind == "tail_up" and last >= 1:
        out[last] = (*out[last][:2], out[last][2] + 2)
        out[last - 1] = (*out[last - 1][:2], out[last - 1][2] + 1)
    elif kind == "tail_down":
        out[last] = (*out[last][:2], out[last][2] - 2)
    elif kind == "split" and out[last][1] >= 2:
        onset, length, step = out[last]
        half = length // 2
        out[last:] = [
            (onset, half, step),
            (onset + half, length - half, step + 1),
        ]
    elif kind == "push" and last >= 2:
        index = rng.randrange(1, last)
        onset, length, step = out[index]
        before, before_length, before_step = out[index - 1]
        if onset - before > 1:
            out[index - 1] = (
                before,
                min(before_length, onset - 1 - before),
                before_step,
            )
            out[index] = (onset - 1, length + 1, step)
    elif kind == "gap" and last >= 2:
        onset, length, step = out[0]
        out[0] = (onset, out[1][0] + out[1][1] - onset, step)
        del out[1]
    return out


def _answer(notes, role):
    head = [n for n in notes if n[0] < HEAD_END][: ANSWER_HEAD[role]]
    head = head or list(notes[:1])
    head = [(o, min(l, HEAD_END - o), s) for o, l, s in head]
    return [*head, (HEAD_END, ANSWER_LENGTH, None)]


def plan_bar(hook, role, rng):
    notes = list(hook.notes)
    if role == "hook":
        return notes
    if role == "vary":
        return _vary(rng, notes)
    if role == "lift":
        return [(o, l, s + 2) for o, l, s in notes]
    if role == "inv":
        return [(o, l, -s) for o, l, s in notes]
    if role == "skeleton":
        return _skeleton(notes)
    if role == "aug":
        return _augment(notes)
    return _answer(notes, role)


def bar_roles(kind, bars, final=False, phrases=None):
    if bars < 1:
        return []
    phrases = (phrases or {}).get(kind) or PHRASES.get(kind, PHRASES["chorus"])
    roles = [
        phrases[(b // PHRASE_BARS) % len(phrases)][b % PHRASE_BARS]
        for b in range(bars)
    ]
    ending = phrases[((bars - 1) // PHRASE_BARS) % len(phrases)][-1]
    roles[-1] = "close" if final else ending
    return roles


def stab_hits(hook, limit=4):
    chosen = sorted(
        (n for n in hook.notes if n[1] >= 2),
        key=lambda n: (-n[1], n[0]),
    )[:limit]
    return sorted({onset / STEPS for onset, _, _ in chosen}) or [0.0]
