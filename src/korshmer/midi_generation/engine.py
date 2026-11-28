import itertools
import random
from dataclasses import dataclass, field

import pretty_midi

from korshmer.constants import GRID_RESOLUTIONS, NOTE_NAMES, SCALES
from korshmer.midi_generation import genre_feel, genres, hooks, moods
from korshmer.utils.tempo import validate_bpm

PROGRESSION_PRESETS = {
    "Pop (I-V-vi-IV)": [0, 4, 5, 3],
    "Jazz (ii-V-I)": [1, 4, 0, 0],
    "Blues (I-IV-V)": [0, 3, 0, 4],
    "Sad (vi-IV-I-V)": [5, 3, 0, 4],
    "Canon (I-V-vi-iii)": [0, 4, 5, 2],
}


SONG_STRUCTURES = {
    "Pop (Verse/Chorus)": [
        ("Intro", 0.3),
        ("Verse", 0.5),
        ("Chorus", 1.0),
        ("Verse", 0.55),
        ("Chorus", 1.0),
        ("Bridge", 0.6),
        ("Chorus", 1.0),
        ("Outro", 0.3),
    ],
    "Simple (A/B)": [("A", 0.5), ("B", 0.9), ("A", 0.5), ("B", 1.0)],
    "Build & Drop": [
        ("Intro", 0.25),
        ("Build", 0.6),
        ("Drop", 1.0),
        ("Break", 0.4),
        ("Drop", 1.0),
        ("Outro", 0.25),
    ],
    "Loop (8 bars)": [("Loop", 0.8)],
}

MELODY_MOODS = list(moods.MOOD_FEELS)

_SECTION_KINDS = {
    "Intro": "intro",
    "Verse": "verse",
    "Chorus": "chorus",
    "Bridge": "bridge",
    "Outro": "outro",
    "A": "verse",
    "B": "chorus",
    "Build": "bridge",
    "Drop": "chorus",
    "Break": "break",
    "Loop": "chorus",
}

DRUM_STYLES = ["Rock", "Techno", "HipHop", "Trap"]


_STYLES = {
    "Pop": ("Pop (I-V-vi-IV)", "Rock", "eighths", (4, 33, 88, 80)),
    "EDM": ("Sad (vi-IV-I-V)", "Techno", "offbeat", (81, 38, 89, 80)),
    "Hip-Hop": ("Sad (vi-IV-I-V)", "HipHop", "sparse", (4, 38, 89, 80)),
    "Lo-Fi": ("Jazz (ii-V-I)", "HipHop", "sparse", (4, 33, 88, 80)),
    "Rock": ("Blues (I-IV-V)", "Rock", "eighths", (29, 34, 88, 80)),
    "Jazz": ("Jazz (ii-V-I)", "Rock", "walking", (0, 32, 89, 65)),
    "Phonk": ("Sad (vi-IV-I-V)", "Trap", "sub", (113, 87, 91, 80)),
}
SONG_STYLES = list(_STYLES)

_SAME_START = 1e-6
CRASH = 49
_KICK, _SNARE, _HAT, _OPEN_HAT, _CLAP = 36, 38, 42, 46, 39
_GHOST_KICK_STEPS = (3, 6, 7, 10, 11, 13, 14)


_DRUM_PATTERNS = {
    "Rock": {
        _KICK: [0, 8, 10],
        _SNARE: [4, 12],
        _HAT: list(range(0, 16, 2)),
    },
    "Techno": {
        _KICK: [0, 4, 8, 12],
        _CLAP: [4, 12],
        _OPEN_HAT: [2, 6, 10, 14],
        _HAT: [1, 3, 5, 7, 9, 11, 13, 15],
    },
    "HipHop": {
        _KICK: [0, 7, 10],
        _SNARE: [4, 12],
        _HAT: list(range(0, 16, 2)),
    },
    "Trap": {
        _KICK: [0, 6, 11],
        _SNARE: [8],
        _CLAP: [8],
        _HAT: list(range(16)),
    },
}


def _rng(seed):
    return random.Random(seed)


@dataclass
class HarmonicContext:
    root: int
    scale: str
    bpm: float
    progression: list = field(default_factory=lambda: [0, 4, 5, 3])
    chord_bars: float = 1.0
    origin: int = 0

    @property
    def intervals(self):
        return SCALES.get(self.scale, SCALES["Major"])

    @property
    def bar(self):
        return 60.0 / self.bpm * 4

    def degree_pitch(self, degree, octave=4):
        n = len(self.intervals)
        octave += degree // n
        return 12 * (octave + 1) + self.root + self.intervals[degree % n]

    def triad(self, degree, octave=4):
        return [self.degree_pitch(degree + step, octave) for step in (0, 2, 4)]

    def chord_at(self, bar_index):
        return self.progression[
            int((bar_index - self.origin) // self.chord_bars)
            % len(self.progression)
        ]

    @classmethod
    def from_midi(cls, pm, bpm, root_idx, scale, progression=None):
        bpm = validate_bpm(bpm)
        ctx = cls(root=root_idx % 12, scale=scale, bpm=bpm)
        if progression:
            ctx.progression = list(progression)
        elif pm is not None and pm.get_end_time() > 0:
            from korshmer.midi_generation import song_match

            analysis = song_match.analyze_song(
                pm, bpm, ctx.root, SCALES.get(scale, SCALES["Major"])
            )
            followed = song_match.progression_for(analysis, "Strong")
            if followed:
                ctx.progression = followed
        return ctx


def _new_inst(name, program, is_drum=False):
    return pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)


def _note(inst, pitch, start, end, velocity):
    pitch = max(0, min(127, int(pitch)))
    velocity = max(1, min(127, int(velocity)))
    if end > start:
        inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))


def trim_overlaps(inst):
    by_pitch = {}
    for note in inst.notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    kept = []
    for notes in by_pitch.values():
        notes.sort(key=lambda n: n.start)
        for note, following in zip(notes, notes[1:], strict=False):
            if following.start - note.start < _SAME_START:
                following.end = max(following.end, note.end)
                note.end = note.start
            elif note.end > following.start:
                note.end = following.start
        kept.extend(n for n in notes if n.end > n.start)
    kept.sort(key=lambda n: (n.start, n.pitch))
    inst.notes = kept


def trim_line(inst):
    ordered = sorted(inst.notes, key=lambda n: n.start)
    for note, following in zip(ordered, ordered[1:], strict=False):
        if note.start < following.start < note.end:
            note.end = following.start


_FILL_START = 8
_FILL_WIPES = {
    "toms": (_KICK, _SNARE, _CLAP, _HAT, _OPEN_HAT),
    "toms_down": (_KICK, _SNARE, _CLAP, _HAT, _OPEN_HAT),
    "roll": (_SNARE, _CLAP),
    "push": (_SNARE, _CLAP),
}
_FILL_TOMS_DOWN = (
    (8, _SNARE),
    (10, genre_feel.HIGH_TOM),
    (11, genre_feel.HIGH_TOM),
    (12, genre_feel.MID_TOM),
    (13, genre_feel.MID_TOM),
    (14, genre_feel.LOW_TOM),
    (15, genre_feel.LOW_TOM),
)
_FILL_PUSH = (10, 13, 14, 15)
_FILL_RIM = (13, 14, 15)
_FILL_TRIPLETS = 6
_FILL_TOMS = (
    (8, genre_feel.HIGH_TOM),
    (10, genre_feel.HIGH_TOM),
    (12, genre_feel.MID_TOM),
    (13, genre_feel.MID_TOM),
    (14, genre_feel.LOW_TOM),
    (15, genre_feel.LOW_TOM),
)
_GHOST_VELOCITY = 36
_ROLL_SOFT = 58
_ROLL_RISE = 40
_CRASH_ACCENT = 108


def _fill_hits(kind):
    if kind == "snare":
        return [(12 + k, _SNARE, 80 + k * 10, 1.0) for k in range(4)]
    if kind == "toms":
        return [(s, p, 82 + i * 6, 1.0) for i, (s, p) in enumerate(_FILL_TOMS)]
    if kind == "roll":
        return [
            (_FILL_START + k, _SNARE, 60 + k * 9, 1.0)
            for k in range(_FILL_START)
        ]
    if kind == "hats":
        return [(12 + k * 0.5, _HAT, 50 + k * 8, 0.5) for k in range(8)]
    if kind == "toms_down":
        return [
            (s, p, 86 + i * 5, 1.0) for i, (s, p) in enumerate(_FILL_TOMS_DOWN)
        ]
    if kind == "push":
        return [(s, _SNARE, 82 + i * 8, 1.0) for i, s in enumerate(_FILL_PUSH)]
    if kind == "rim":
        return [
            (s, genre_feel.SIDE_STICK, 60 + i * 9, 1.0)
            for i, s in enumerate(_FILL_RIM)
        ]
    if kind == "triplets":
        span = 4 / _FILL_TRIPLETS
        return [
            (12 + k * span, _HAT, 56 + k * 8, span)
            for k in range(_FILL_TRIPLETS)
        ]
    return []


def generate_drums(
    ctx,
    style="Rock",
    bpm=None,
    bars=4,
    seed=None,
    energy=1.0,
    start_bar=0,
    pattern=None,
    fill_bars=4,
    velocity_scale=1.0,
    variation=0.0,
    ghosts=None,
    fill="snare",
    crash=False,
    accents=None,
    rolls=(),
    roll_chance=0.0,
):
    rng = _rng(seed)
    if pattern is None:
        pattern = _DRUM_PATTERNS.get(style, _DRUM_PATTERNS["Rock"])
    bar = 60.0 / (bpm or ctx.bpm) * 4
    step = bar / 16
    inst = _new_inst("Drums", 0, is_drum=True)
    for b in range(bars):
        t0 = (start_bar + b) * bar
        filling = fill_bars and b % fill_bars == fill_bars - 1 and energy >= 0.5
        roll = None
        if rolls and energy >= 0.5 and not filling:
            if rng.random() < roll_chance:
                roll = rng.choice(rolls)
        for pitch, steps in pattern.items():
            if energy < 0.5 and pitch in (_OPEN_HAT, _CLAP):
                continue
            wiped = filling and pitch in _FILL_WIPES.get(fill, ())
            for s in steps:
                if wiped and s >= _FILL_START:
                    continue
                if energy < 0.5 and pitch == _HAT and s % 4:
                    continue
                if roll and pitch == _HAT and 0 <= s - roll[0] < roll[1]:
                    continue
                vel = 100 if s % 4 == 0 else 78
                if accents and pitch in accents:
                    vel = accents[pitch]
                vel += rng.randint(-8, 8)
                vel = max(1, min(127, int(vel * velocity_scale)))
                _note(
                    inst, pitch, t0 + s * step, t0 + s * step + step * 0.9, vel
                )
        if roll:
            first, span, count = roll
            for k in range(count):
                at = t0 + (first + k * span / count) * step
                _note(
                    inst,
                    _HAT,
                    at,
                    at + step * span / count * 0.9,
                    (_ROLL_SOFT + k * _ROLL_RISE // count) * velocity_scale,
                )
        if ghosts and energy >= 0.5:
            for pitch, steps in ghosts.items():
                for s in steps:
                    if rng.random() < 0.6:
                        _note(
                            inst,
                            pitch,
                            t0 + s * step,
                            t0 + s * step + step * 0.5,
                            _GHOST_VELOCITY + rng.randint(-6, 6),
                        )
        if variation and energy >= 0.5 and rng.random() < variation:
            kicks = pattern.get(_KICK, [])
            free = [s for s in _GHOST_KICK_STEPS if s not in kicks]
            if free:
                s = rng.choice(free)
                _note(
                    inst,
                    _KICK,
                    t0 + s * step,
                    t0 + s * step + step * 0.9,
                    84 + rng.randint(-6, 6),
                )
        if filling:
            for s, pitch, vel, length in _fill_hits(fill):
                _note(
                    inst,
                    pitch,
                    t0 + s * step,
                    t0 + (s + length * 0.9) * step,
                    int(vel * velocity_scale),
                )
        if crash and energy >= 0.5 and (b == 0 or not b % (fill_bars or 4)):
            _note(inst, CRASH, t0, t0 + bar * 0.5, _CRASH_ACCENT)
    return inst


def _fold(pitch, floor):
    return floor + (pitch - floor) % 12


def _bass_root(ctx, degree, floor):
    return _fold(ctx.degree_pitch(degree, 2), floor)


@dataclass
class _BassBar:
    ctx: HarmonicContext
    degree: int
    following: int
    rng: random.Random
    kicks: list
    riff: list
    floor: int
    jump: float

    def pitch(self, step=0):
        return _bass_root(self.ctx, self.degree, self.floor) + (
            self.ctx.degree_pitch(self.degree + step, 2)
            - self.ctx.degree_pitch(self.degree, 2)
        )

    def approach(self):
        if self.following == self.degree:
            return self.pitch(4)
        side = 1 if self.following < self.degree else -1
        return _bass_root(self.ctx, self.following, self.floor) + (
            self.ctx.degree_pitch(self.following + side, 2)
            - self.ctx.degree_pitch(self.following, 2)
        )


def _bass_eighths(bar):
    root = bar.pitch()
    pitches = [
        root,
        root,
        root + 12,
        root,
        root,
        root,
        bar.pitch(4),
        bar.approach(),
    ]
    velocities = (100, 78, 84, 78, 92, 78, 84, 86)
    return [(2 * k, 2, p, velocities[k]) for k, p in enumerate(pitches)]


def _bass_driving(bar):
    pitches = [bar.pitch()] * 7 + [bar.approach()]
    velocities = (105, 80, 88, 80, 100, 80, 88, 84)
    return [(2 * k, 2, p, velocities[k]) for k, p in enumerate(pitches)]


def _bass_syncopated(bar):
    hits = [
        (0, 3, bar.pitch(), 100),
        (3, 3, bar.pitch(), 88),
        (6, 3, bar.pitch(4), 90),
        (10, 3, bar.pitch(), 92),
    ]
    if bar.rng.random() < 0.5:
        hits.append((13, 3, bar.approach(), 82))
    return hits


def _bass_offbeat(bar):
    root = bar.pitch()
    pitches = (root, root, root + 12, root)
    velocities = (95, 90, 92, 98)
    return [(2 + 4 * k, 2, pitches[k], velocities[k]) for k in range(4)]


def _bass_pump(bar):
    root = bar.pitch()
    hits = []
    for beat in range(4):
        for k, vel in enumerate((78, 88, 96), 1):
            pitch = root + 12 if beat == 3 and k == 3 else root
            hits.append((beat * 4 + k, 1, pitch, vel))
    return hits


def _bass_sparse(bar):
    second = bar.pitch(4) if bar.rng.random() < 0.5 else bar.pitch()
    return [(0, 7, bar.pitch(), 100), (10, 5, second, 85)]


def _bass_808(bar):
    steps = sorted({0, *bar.kicks}) if bar.kicks else [0, 7, 10]
    hits = []
    for index, s in enumerate(steps):
        edge = steps[index + 1] if index + 1 < len(steps) else 16
        pitch = bar.pitch()
        if index == len(steps) - 1 and s >= 10 and bar.following != bar.degree:
            pitch = bar.approach()
        hits.append((s, edge - s, pitch, 108 if s == 0 else 96))
    return hits


def _bass_sub(bar):
    steps = [0]
    for s in sorted(bar.kicks or _SUB_STEPS):
        if s - steps[-1] >= _SUB_SPACING:
            steps.append(s)
    last = len(steps) - 1
    jumping = last > 0 and bar.rng.random() < bar.jump
    jumping = jumping and bar.pitch() + 12 <= bar.floor + _SUB_REACH
    hits = []
    for index, s in enumerate(steps):
        edge = steps[index + 1] if index < last else 16
        pitch = bar.pitch()
        if index == last and index and bar.following != bar.degree:
            pitch = bar.approach()
        elif index == last and jumping:
            pitch += 12
        hits.append((s, edge - s, pitch, 112 if s == 0 else 104))
    return hits


def _bass_octaves(bar):
    root = bar.pitch()
    pitches = [root, root + 12] * 3 + [root, bar.approach()]
    velocities = (100, 82, 92, 82, 98, 82, 92, 86)
    return [(2 * k, 2, p, velocities[k]) for k, p in enumerate(pitches)]


def _bass_push(bar):
    root = bar.pitch()
    return [
        (0, 2, root, 100),
        (3, 1, root, 84),
        (4, 2, root, 92),
        (7, 1, bar.pitch(4), 84),
        (8, 2, root, 98),
        (11, 1, root, 84),
        (12, 2, bar.pitch(4), 90),
        (14, 2, bar.approach(), 86),
    ]


def _bass_gallop(bar):
    root = bar.pitch()
    hits = []
    for beat in range(4):
        last = bar.approach() if beat == 3 else root
        hits.append((beat * 4, 2, root, 104 if beat % 2 == 0 else 96))
        hits.append((beat * 4 + 2, 1, root, 82))
        hits.append((beat * 4 + 3, 1, last, 86))
    return hits


def _bass_chug(bar):
    root = bar.pitch()
    fifth = bar.pitch(4)
    pitches = [root, root, root, fifth, root, root, fifth, bar.approach()]
    velocities = (106, 80, 90, 86, 100, 80, 90, 86)
    return [(2 * k, 2, p, velocities[k]) for k, p in enumerate(pitches)]


def _bass_lazy(bar):
    middle = bar.pitch(4) if bar.rng.random() < 0.5 else bar.pitch()
    return [
        (0, 6, bar.pitch(), 100),
        (6, 6, middle, 86),
        (12, 4, bar.approach(), 84),
    ]


def _bass_drop(bar):
    last = bar.approach() if bar.following != bar.degree else bar.pitch()
    return [
        (0, 7, bar.pitch(), 108),
        (7, 3, bar.pitch() + 12, 92),
        (10, 6, last, 98),
    ]


def _bass_stutter(bar):
    last = bar.approach() if bar.following != bar.degree else bar.pitch()
    return [
        (0, 3, bar.pitch(), 108),
        (3, 5, bar.pitch(), 96),
        (10, 6, last, 100),
    ]


def _bass_two(bar):
    hits = [(0, 8, bar.pitch(), 96)]
    if bar.rng.random() < 0.5:
        hits.append((8, 8, bar.pitch(4), 86))
    else:
        hits.append((8, 4, bar.pitch(4), 86))
        hits.append((12, 4, bar.approach(), 84))
    return hits


def _bass_skip(bar):
    root = bar.pitch()
    return [
        (2, 1, root, 94),
        (3, 1, root, 84),
        (6, 2, root + 12, 92),
        (10, 1, root, 94),
        (11, 1, root, 84),
        (14, 2, root, 98),
    ]


def _bass_bounce(bar):
    root = bar.pitch()
    return [
        (2, 2, root, 95),
        (6, 2, root + 12, 90),
        (10, 2, root, 95),
        (14, 1, root + 12, 92),
        (15, 1, root, 86),
    ]


def _bass_walking(bar):
    pitches = (
        bar.pitch(),
        bar.pitch(bar.rng.choice((1, 2))),
        bar.pitch(bar.rng.choice((2, 4))),
        bar.approach(),
    )
    velocities = (92, 82, 86, 84)
    return [(4 * k, 4, pitches[k], velocities[k]) for k in range(4)]


def _bass_riff(bar):
    last = len(bar.riff) - 1
    return [
        (
            onset,
            length,
            bar.approach()
            if k == last and bar.following != bar.degree
            else bar.pitch(),
            100 if onset in hooks.STRONG_ONSETS else 86,
        )
        for k, (onset, length, _) in enumerate(bar.riff)
    ]


_BASS_BARS = {
    "eighths": _bass_eighths,
    "driving": _bass_driving,
    "syncopated": _bass_syncopated,
    "offbeat": _bass_offbeat,
    "pump": _bass_pump,
    "sparse": _bass_sparse,
    "808": _bass_808,
    "sub": _bass_sub,
    "walking": _bass_walking,
    "riff": _bass_riff,
    "octaves": _bass_octaves,
    "push": _bass_push,
    "gallop": _bass_gallop,
    "chug": _bass_chug,
    "lazy": _bass_lazy,
    "drop": _bass_drop,
    "stutter": _bass_stutter,
    "two": _bass_two,
    "skip": _bass_skip,
    "bounce": _bass_bounce,
}
BASS_BUSYNESS = {
    "sparse": 2,
    "two": 2,
    "lazy": 3,
    "drop": 3,
    "stutter": 3,
    "808": 3,
    "sub": 3,
    "walking": 4,
    "offbeat": 4,
    "syncopated": 5,
    "bounce": 5,
    "skip": 6,
    "eighths": 8,
    "driving": 8,
    "octaves": 8,
    "push": 8,
    "chug": 8,
    "riff": 8,
    "gallop": 12,
    "pump": 12,
}
_SUB_STEPS = (0, 6, 10)
_SUB_SPACING = 4
_SUB_REACH = 20
_POWER_FLOOR = 45
_SHELL_FLOOR = 2
_STACK_DROP = 5
_KICK_LOCKED = ("eighths", "driving", "sparse", "syncopated")


def generate_bass(
    ctx,
    style="Pop",
    drums=None,
    seed=None,
    bars=4,
    start_bar=0,
    energy=1.0,
    rhythm=None,
    riff=None,
    kick_steps=None,
):
    rng = _rng(seed)
    if rhythm is None:
        rhythm = _STYLES.get(genres.primary(style), _STYLES["Pop"])[2]
    if rhythm == "riff" and not riff:
        rhythm = "driving"
    pattern = _BASS_BARS.get(rhythm, _bass_eighths)
    program = _STYLES.get(genres.primary(style), _STYLES["Pop"])[3][1]
    feel = genre_feel.feel_for(style)
    gate = feel.bass_gate
    inst = _new_inst("Bass", program)
    bar = ctx.bar
    unit = bar / 16
    if drums is not None:
        drum_end_bar = int(max((n.end for n in drums.notes), default=0) // bar)
        bars = max(bars, drum_end_bar - start_bar)
    kicks = []
    if drums is not None:
        kicks = sorted(n.start for n in drums.notes if n.pitch == _KICK)
    for b in range(bars):
        t0 = (start_bar + b) * bar
        degree = ctx.chord_at(start_bar + b)
        following = ctx.chord_at(start_bar + b + 1)
        in_bar = [
            round((k - t0) / unit)
            for k in kicks
            if t0 <= k < t0 + bar - unit / 2
        ]
        if kick_steps is not None and not kicks:
            in_bar = list(kick_steps)
        info = _BassBar(
            ctx,
            degree,
            following,
            rng,
            in_bar,
            riff,
            feel.bass_floor,
            feel.bass_jump,
        )
        for s, length, pitch, vel in pattern(info):
            _note(
                inst, pitch, t0 + s * unit, t0 + (s + length * gate) * unit, vel
            )
        if kicks and rhythm in _KICK_LOCKED:
            for n in list(inst.notes):
                if n.start >= t0 and n.start < t0 + bar:
                    nearest = min(kicks, key=lambda x: abs(x - n.start))
                    if abs(nearest - n.start) < bar / 32:
                        n.start = nearest
    if energy < 0.5:
        for n in inst.notes:
            n.velocity = max(1, int(n.velocity * 0.75))
    return inst


def _close_voicing(ctx, degree, octave, seventh, previous):
    chord = ctx.triad(degree, octave)
    if seventh:
        chord = chord + [ctx.degree_pitch(degree + 6, octave)]
    if previous is not None and len(previous) == len(chord):
        best = chord
        for inv in range(3):
            cand = sorted(p + 12 if i < inv else p for i, p in enumerate(chord))
            cand = [p - 12 if p > 72 + 12 * (octave - 4) else p for p in cand]
            if sum(
                abs(a - c) for a, c in zip(previous, cand, strict=True)
            ) < sum(abs(a - c) for a, c in zip(previous, best, strict=True)):
                best = cand
        chord = sorted(best)
    return chord


def _rooted(ctx, degree, octave, steps, floor):
    root = ctx.degree_pitch(degree, octave)
    base = _fold(root, floor)
    return [base + ctx.degree_pitch(degree + s, octave) - root for s in steps]


def _open_voicing(ctx, degree, octave, seventh):
    floor = 12 * octave
    root, fifth, third = _rooted(ctx, degree, octave, (0, 4, 2), floor)
    chord = [root, fifth, third + 12]
    if seventh:
        chord.append(_rooted(ctx, degree, octave, (6,), floor)[0])
    return sorted(chord)


def _shell_voicing(ctx, degree, octave, previous):
    low = 12 * (octave + 1) - _SHELL_FLOOR
    base = sorted(
        _fold(ctx.degree_pitch(degree + step, octave), low)
        for step in (2, 6, 8)
    )
    options = [
        base,
        [base[1], base[2], base[0] + 12],
        [base[2], base[0] + 12, base[1] + 12],
    ]
    if previous is None or len(previous) != len(base):
        return base
    return min(
        options,
        key=lambda c: sum(abs(a - b) for a, b in zip(previous, c, strict=True)),
    )


def _stack_voicing(ctx, degree, octave, seventh):
    floor = 12 * (octave + 1) - _STACK_DROP
    steps = (0, 2, 6) if seventh else (0, 2, 4)
    chord = _rooted(ctx, degree, octave, steps, floor)
    chord.append(chord[0] + 12)
    return sorted(chord)


def generate_chords(
    ctx,
    style="Pop",
    seed=None,
    bars=4,
    start_bar=0,
    energy=1.0,
    hits=None,
    seventh=False,
    power=False,
    octave=4,
):
    rng = _rng(seed)
    program = _STYLES.get(genres.primary(style), _STYLES["Pop"])[3][0]
    feel = genre_feel.feel_for(style)
    inst = _new_inst("Chords", program)
    bar = ctx.bar
    prev = None
    for b in range(bars):
        t0 = (start_bar + b) * bar
        degree = ctx.chord_at(start_bar + b)
        if power:
            low = _fold(ctx.degree_pitch(degree, octave), _POWER_FLOOR)
            chord = [low, low + 7, low + 12]
        elif feel.voicing == "open":
            chord = _open_voicing(ctx, degree, octave, seventh)
        elif feel.voicing == "shell":
            chord = _shell_voicing(ctx, degree, octave, prev)
        elif feel.voicing == "stack":
            chord = _stack_voicing(ctx, degree, octave, seventh)
        else:
            chord = _close_voicing(ctx, degree, octave, seventh, prev)
        prev = chord
        bar_hits = hits or ([0.0, 0.5] if energy >= 0.7 else [0.0])
        edges = [*bar_hits[1:], 1.0]
        for h, edge in zip(bar_hits, edges, strict=True):
            for p in chord:
                _note(
                    inst,
                    p,
                    t0 + h * bar,
                    t0 + (h + (edge - h) * feel.chord_gate) * bar,
                    78 + rng.randint(-5, 5),
                )
    return inst


_PAD_HOLD = 1.02


def _pad_tones(ctx, degree, octave, shape):
    root, third, fifth = ctx.triad(degree, octave)
    if shape == "open":
        return [root, fifth, third + 12]
    if shape == "octave":
        return [root, third, fifth, root + 12]
    if shape == "low":
        return [root - 12, root, third, fifth]
    return [root, third, fifth]


def generate_pad(
    ctx,
    seed=None,
    bars=4,
    start_bar=0,
    energy=1.0,
    octave=3,
    shape="triad",
):
    rng = _rng(seed)
    inst = _new_inst("Pad", 89)
    bar = ctx.bar
    strikes = (0.0, 0.5) if shape == "pulse" else (0.0,)
    for b in range(bars):
        t0 = (start_bar + b) * bar
        tones = _pad_tones(ctx, ctx.chord_at(start_bar + b), octave, shape)
        for at, edge in zip(strikes, [*strikes[1:], _PAD_HOLD], strict=True):
            for p in tones:
                _note(
                    inst,
                    p,
                    t0 + at * bar,
                    t0 + edge * bar,
                    55 + int(25 * energy) + rng.randint(-3, 3),
                )
    return inst


def generate_arp(
    ctx,
    rate="1/16",
    seed=None,
    bars=4,
    start_bar=0,
    energy=1.0,
    shape="updown",
    octave=5,
):
    rng = _rng(seed)
    div = GRID_RESOLUTIONS.get(rate, 4)
    steps = div * 4
    inst = _new_inst("Arp", 81)
    bar = ctx.bar
    for b in range(bars):
        t0 = (start_bar + b) * bar
        tones = ctx.triad(ctx.chord_at(start_bar + b), octave)
        tones.append(tones[0] + 12)
        if shape == "up":
            order = tones
        elif shape == "down":
            order = tones[::-1]
        else:
            order = tones + tones[-2:0:-1]
        for s in range(steps):
            t = t0 + s * bar / steps
            _note(
                inst,
                order[s % len(order)],
                t,
                t + bar / steps * 0.8,
                70 + rng.randint(-6, 6),
            )
    return inst


_CADENCE_GATE = 0.95
_LEAD_OCTAVE = 4
_LEAD_SPAN = 6
_LEAD_CEILING = 84
_KIND_DROP = {"verse": 8, "intro": 3, "outro": 3, "break": 6, "bridge": 6}
_CENTER_PULL = 0.3
_DIATONIC = 7
_DEGREE_REACH = 3
_PLAN_STRETCH = 8
_SNAP_REACH = (0, 1, -1, 2, -2, 3, -3)
_PLACE_SHIFTS = (0, -2, 2)
_SHIFT_COST = 0.5
_STANDSTILL_COST = 100.0
_LEAD_FLOOR = 60
_LEAD_ROOF = 77


@dataclass
class _Lead:
    ctx: HarmonicContext
    mid: int
    snap_inside: bool = False
    last: tuple = ()

    @property
    def low(self):
        return self.mid - _LEAD_SPAN

    @property
    def high(self):
        return self.mid + _LEAD_SPAN

    def clamp(self, degree):
        if degree > self.high:
            degree = 2 * self.high - degree
        if degree < self.low:
            degree = 2 * self.low - degree
        return max(self.low, min(self.high, degree))

    def snap(self, degree, chord):
        n = len(self.ctx.intervals)
        tones = {(chord + step) % n for step in (0, 2, 4)}
        near = [
            degree + delta
            for delta in _SNAP_REACH
            if (degree + delta) % n in tones
        ]
        if not near:
            return degree
        if self.snap_inside or not self.low <= near[0] <= self.high:
            near = [d for d in near if self.low <= d <= self.high] or near
        return self.clamp(near[0])

    def place(self, chord, steps, previous):
        n = len(self.ctx.intervals)
        home = round((self.mid - chord) / n)
        best, best_cost, best_base = None, None, None
        moved = self.last and self.last[0] != chord
        for octave, shift in itertools.product(
            (home - 1, home, home + 1), _PLACE_SHIFTS
        ):
            base = chord + octave * n + shift
            degrees = [base + s for s in steps]
            while max(degrees) > self.high and min(degrees) - n >= self.low:
                degrees = [d - n for d in degrees]
            while min(degrees) < self.low and max(degrees) + n <= self.high:
                degrees = [d + n for d in degrees]
            average = sum(degrees) / len(degrees)
            cost = (
                abs(degrees[0] - previous)
                + _CENTER_PULL * abs(average - self.mid)
                + _SHIFT_COST * (shift != 0)
            )
            if moved and shift and (base - self.last[1]) % n == 0:
                cost += _STANDSTILL_COST
            if moved and [self.clamp(d) for d in degrees] == self.last[2]:
                cost += _STANDSTILL_COST
            if best_cost is None or cost < best_cost:
                best, best_cost, best_base = degrees, cost, base
        placed = [self.clamp(d) for d in best]
        self.last = (chord, best_base, placed)
        return placed

    def cadence(self, chord, role, previous):
        n = len(self.ctx.intervals)
        target = chord if role == "close" else chord + 4
        options = [
            d
            for d in (target + k * n for k in range(-_DEGREE_REACH, 5))
            if self.low <= d <= self.high
        ]
        return min(options, key=lambda d: abs(d - previous))


def _lead_velocity(rng, energy, onset, length, in_phrase, mood=0):
    accent = 8 if onset in hooks.STRONG_ONSETS else 0
    if onset % 4 and length >= 2:
        accent += 5
    arc = 6 if in_phrase in (1, 2) else 0
    return 80 + mood + int(15 * energy) + accent + arc - rng.randint(0, 6)


def mood_register(mood, style):
    low, high = genre_feel.feel_for(style).hook.room
    return max(low, min(high, moods.mood_for(mood).register))


def lead_register(top, drop, lift, shift):
    plain = min(top + lift, _LEAD_CEILING) - drop
    if shift > 0:
        shifted = min(top + shift, max(top, _LEAD_ROOF))
    else:
        shifted = top + shift
    register = min(shifted + lift, _LEAD_CEILING) - drop
    return max(register, min(plain, _LEAD_FLOOR))


def _lead_for(ctx, register, snap_inside=False):
    n = len(ctx.intervals)
    mid = min(
        range(-2 * n, 6 * n),
        key=lambda d: abs(ctx.degree_pitch(d, _LEAD_OCTAVE) - register),
    )
    return _Lead(ctx, mid, snap_inside)


def generate_melody(
    ctx,
    bars=4,
    mood="Uplifting",
    seed=None,
    start_bar=0,
    energy=1.0,
    hook=None,
    kind="chorus",
    style="Pop",
    final=False,
    lift=0,
    phrases=None,
    turn=0,
    register_mood=None,
):
    rng = _rng(seed)
    feel = genre_feel.feel_for(style)
    mood_feel = moods.mood_for(mood)
    if hook is None:
        hook = hooks.make_hook(rng, style, mood)
    gate = min(1.0, hook.gate * mood_feel.gate)
    inst = _new_inst("Melody", 80)
    bar = ctx.bar
    unit = bar / hooks.STEPS
    register = lead_register(
        hook.register,
        _KIND_DROP.get(kind, 0),
        lift,
        turn + mood_register(register_mood or mood, style),
    )
    lead = _lead_for(ctx, register, feel.lead_snaps_inside)
    previous = lead.mid
    settled = {}
    plans = {}
    for b, role in enumerate(
        hooks.bar_roles(kind, bars, final, phrases or feel.phrases)
    ):
        stretch = (role, b // _PLAN_STRETCH)
        if stretch not in plans:
            plans[stretch] = hooks.plan_bar(hook, role, rng)
        plan = plans[stretch]
        t0 = (start_bar + b) * bar
        chord = ctx.chord_at(start_bar + b)
        steps = [s for _, _, s in plan if s is not None]
        degrees = settled.get((chord, role))
        if degrees is None:
            degrees = lead.place(chord, steps, previous)
            if role in hooks.FIXED_ROLES:
                settled[(chord, role)] = degrees
        placed = iter(degrees)
        for onset, length, step in plan:
            legato = gate
            if step is None:
                degree = lead.cadence(chord, role, previous)
                legato = max(gate, _CADENCE_GATE)
            else:
                degree = next(placed)
                if (
                    hooks.strong(onset, length)
                    or len(ctx.intervals) != _DIATONIC
                ):
                    degree = lead.snap(degree, chord)
            previous = degree
            _note(
                inst,
                ctx.degree_pitch(degree, _LEAD_OCTAVE),
                t0 + onset * unit,
                t0 + (onset + length * legato) * unit,
                _lead_velocity(
                    rng,
                    energy,
                    onset,
                    length,
                    b % hooks.PHRASE_BARS,
                    mood_feel.velocity,
                ),
            )
    return inst


def generate_countermelody(ctx, lead, seed=None):
    rng = _rng(seed)
    inst = _new_inst("Counter-melody", 73)
    for n in lead.notes:
        shift = -rng.choice([3, 4]) if n.pitch > 60 else 3
        pitch = n.pitch + shift

        pcs = {(ctx.root + i) % 12 for i in ctx.intervals}
        for d in (0, 1, -1, 2, -2):
            if (pitch + d) % 12 in pcs:
                pitch += d
                break
        _note(
            inst,
            pitch - 12 if pitch > 79 else pitch,
            n.start,
            n.end,
            max(1, n.velocity - 20),
        )
    return inst


def build_song(
    root,
    scale,
    style="Pop",
    bpm=120.0,
    structure="Pop (Verse/Chorus)",
    progression=None,
    seed=None,
    parts=None,
    bars_per_section=8,
):
    bpm = validate_bpm(bpm)
    parts = parts or ["drums", "bass", "chords", "melody", "pad"]
    root_idx = NOTE_NAMES.index(root) if root in NOTE_NAMES else 0
    style_cfg = _STYLES.get(genres.primary(style), _STYLES["Pop"])
    prog_name = progression or style_cfg[0]
    prog = PROGRESSION_PRESETS.get(prog_name, PROGRESSION_PRESETS[style_cfg[0]])
    sections = SONG_STRUCTURES.get(structure, SONG_STRUCTURES["Loop (8 bars)"])
    mood = (
        "Melancholic" if scale in ("Minor", "Harmonic Minor") else "Uplifting"
    )
    ctx = HarmonicContext(
        root=root_idx, scale=scale, bpm=bpm, progression=list(prog)
    )
    base = seed if seed is not None else random.randrange(1 << 30)

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    out = {}
    names = {
        "drums": "Drums",
        "bass": "Bass",
        "chords": "Chords",
        "melody": "Melody",
        "pad": "Pad",
        "arp": "Arp",
        "counter": "Counter-melody",
    }
    hook = hooks.make_hook(random.Random(base), style, mood)
    cursor = 0
    for si, (name, energy) in enumerate(sections):
        bars = int(bars_per_section)
        s_seed = base + si
        drums = None
        if "drums" in parts:
            drums = generate_drums(
                ctx,
                style_cfg[1],
                bars=bars,
                seed=s_seed,
                energy=energy,
                start_bar=cursor,
            )
            out.setdefault("drums", _new_inst("Drums", 0, True)).notes.extend(
                drums.notes
            )
        kw = {
            "seed": s_seed,
            "bars": bars,
            "start_bar": cursor,
            "energy": energy,
        }
        for key in ("bass", "chords", "melody", "pad", "arp"):
            if key not in parts:
                continue
            if energy < 0.5 and key == "arp":
                continue
            if key == "bass":
                inst = generate_bass(ctx, style, drums, **kw)
            elif key == "chords":
                inst = generate_chords(ctx, style, **kw)
            elif key == "melody":
                inst = generate_melody(
                    ctx,
                    mood=mood,
                    hook=hook,
                    kind=_SECTION_KINDS.get(name, "chorus"),
                    style=style,
                    final=si == len(sections) - 1,
                    **kw,
                )
            elif key == "pad":
                inst = generate_pad(ctx, **kw)
            else:
                inst = generate_arp(ctx, "1/16", **kw)
            out.setdefault(
                key, _new_inst(names[key], inst.program)
            ).notes.extend(inst.notes)
        if "counter" in parts and "melody" in out:
            seg = _new_inst("m", 0)
            seg.notes = [
                n for n in out["melody"].notes if n.start >= cursor * ctx.bar
            ]
            c = generate_countermelody(ctx, seg, seed=s_seed)
            out.setdefault(
                "counter", _new_inst(names["counter"], c.program)
            ).notes.extend(c.notes)
        cursor += bars
    for key in parts:
        if key in out:
            trim_overlaps(out[key])
            if out[key].notes:
                pm.instruments.append(out[key])
    return pm
