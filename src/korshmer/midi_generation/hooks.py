from dataclasses import dataclass

from korshmer.midi_generation import genre_feel

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
MOOD_NOTES = {"Uplifting": 6, "Melancholic": 4, "Energetic": 7, "Dreamy": 4}

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
PHRASE_BARS = 4
FIXED_ROLES = ("hook", "lift", "inv", "skeleton", "aug")


@dataclass(frozen=True)
class Hook:
    notes: tuple
    gate: float
    register: int


def strong(onset, length):
    return onset in STRONG_ONSETS or length >= 4


def _draw(rng, rhythm, spec):
    steps = []
    current = 0
    for index, (onset, length) in enumerate(rhythm):
        if index == 0:
            current = rng.choice((0, 0, 2, 4))
        else:
            roll = rng.random()
            if roll < spec.repeat:
                move = 0
            elif roll < spec.repeat + 0.12:
                move = rng.choice(spec.leaps) * rng.choice((-1, 1))
            else:
                move = rng.choice((-2, -1, -1, 1, 1, 2))
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


def score_hook(rhythm, steps, spec, mood):
    moves = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    repeats = sum(1 for m in moves if m == 0)
    leaps = [i for i, m in enumerate(moves) if abs(m) >= LEAP]
    span = max(steps) - min(steps)
    turns = _turns(moves)
    score = -3.0 * abs(repeats / len(moves) - spec.repeat)
    if len(leaps) == 1:
        score += 2.0
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
    score -= 0.25 * abs(len(rhythm) - MOOD_NOTES[mood])
    if mood == "Melancholic" and steps[-1] <= steps[0]:
        score += 0.8
    elif mood == "Uplifting" and steps[-1] >= steps[0]:
        score += 0.6
    elif mood == "Dreamy" and leaps:
        score += 0.5
    return score


def make_hook(rng, style="Pop", mood="Uplifting"):
    spec = genre_feel.feel_for(style).hook
    mood = mood if mood in MOOD_NOTES else "Uplifting"
    best, best_score = None, None
    for _ in range(CANDIDATES):
        rhythm = rng.choice(spec.rhythms)
        steps = _draw(rng, rhythm, spec)
        score = score_hook(rhythm, steps, spec, mood) + rng.random() * 0.4
        if best_score is None or score > best_score:
            best, best_score = (rhythm, steps), score
    rhythm, steps = best
    notes = tuple(
        (onset, length, step)
        for (onset, length), step in zip(rhythm, steps, strict=True)
    )
    return Hook(notes, spec.gate, spec.register)


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
    kind = rng.choice(("tail_up", "tail_down", "split", "push", "gap"))
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


def bar_roles(kind, bars, final=False):
    if bars < 1:
        return []
    phrases = PHRASES.get(kind, PHRASES["chorus"])
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
