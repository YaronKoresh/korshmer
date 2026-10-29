from dataclasses import dataclass, field

KICK, SIDE_STICK, SNARE, CLAP = 36, 37, 38, 39
HAT, PEDAL_HAT, OPEN_HAT = 42, 44, 46
RIDE, TAMBOURINE, SHAKER = 51, 54, 70
LOW_TOM, MID_TOM, HIGH_TOM = 45, 47, 50


_TRIPLET = 1 / 3
_EVEN = list(range(0, 16, 2))
_SIXTEENTHS = list(range(16))


@dataclass(frozen=True)
class HookStyle:
    rhythms: tuple
    repeat: float
    span: tuple
    leaps: tuple
    gate: float
    register: int


@dataclass(frozen=True)
class GenreFeel:
    grooves: tuple
    swing: float
    progressions: tuple
    bass: tuple
    chord_hits: tuple
    hook: HookStyle
    seventh: float
    minor_progressions: tuple = ()
    power_chords: bool = False
    drum_vel: float = 1.0
    ghosts: dict = field(default_factory=dict)
    fill: str = "snare"
    crash: bool = False
    bass_gate: float = 0.9
    chord_gate: float = 0.9
    strum: float = 0.0
    voicing: str = "close"
    stab: bool = False
    riff: bool = False
    arp_rates: tuple = ("1/8", "1/16")


FEELS = {
    "Pop": GenreFeel(
        grooves=(
            {
                KICK: [0, 8, 10],
                SNARE: [4, 12],
                CLAP: [4, 12],
                HAT: [0, 2, 4, 6, 8, 10, 12],
                OPEN_HAT: [14],
            },
            {
                KICK: [0, 6, 8, 11],
                SNARE: [4, 12],
                CLAP: [4, 12],
                HAT: _SIXTEENTHS,
            },
            {
                KICK: [0, 8],
                CLAP: [4, 12],
                HAT: [2, 6, 10, 14],
                SHAKER: _SIXTEENTHS,
            },
        ),
        swing=0.0,
        progressions=((0, 4, 5, 3), (0, 5, 3, 4), (0, 3, 5, 4)),
        minor_progressions=((0, 5, 2, 6), (0, 6, 5, 6), (0, 3, 6, 2)),
        bass=("eighths", "syncopated", "eighths"),
        chord_hits=([0.0, 0.5], [0.0, 0.375, 0.5], [0.0, 0.25, 0.5, 0.75]),
        hook=HookStyle(
            rhythms=(
                ((0, 3), (3, 3), (6, 2), (8, 3), (11, 3)),
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)),
                ((2, 2), (4, 2), (6, 2), (8, 3), (11, 2), (13, 3)),
                ((0, 4), (4, 2), (6, 2), (8, 3), (11, 5)),
                ((0, 2), (3, 1), (4, 2), (8, 2), (11, 1), (12, 4)),
                ((0, 3), (3, 3), (6, 3), (9, 3), (12, 4)),
            ),
            repeat=0.2,
            span=(3, 6),
            leaps=(3, 4),
            gate=0.88,
            register=74,
        ),
        arp_rates=("1/8", "1/16"),
        seventh=0.15,
        fill="snare",
        crash=True,
        bass_gate=0.85,
        chord_gate=0.85,
        strum=0.008,
    ),
    "Rock": GenreFeel(
        grooves=(
            {KICK: [0, 8, 10], SNARE: [4, 12], HAT: _EVEN},
            {
                KICK: [0, 2, 8, 10],
                SNARE: [4, 12],
                OPEN_HAT: [6, 14],
                HAT: [0, 2, 4, 8, 10, 12],
            },
            {KICK: [0, 6, 8], SNARE: [4, 12], RIDE: _EVEN},
        ),
        swing=0.0,
        progressions=((0, 3, 4, 3), (0, 3, 0, 4), (0, 4, 3, 4)),
        minor_progressions=((0, 5, 6, 0), (0, 3, 4, 0), (0, 6, 3, 0)),
        bass=("driving", "driving", "syncopated"),
        chord_hits=(
            [0.0, 0.5],
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.375, 0.5, 0.75],
        ),
        hook=HookStyle(
            rhythms=(
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
                ((0, 3), (3, 3), (6, 2), (8, 4), (12, 4)),
                ((0, 2), (3, 1), (4, 2), (8, 2), (11, 1), (12, 4)),
                ((0, 4), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)),
                ((0, 2), (2, 2), (5, 2), (8, 2), (10, 2), (13, 3)),
            ),
            repeat=0.35,
            span=(3, 5),
            leaps=(3, 4),
            gate=0.78,
            register=71,
        ),
        arp_rates=("1/8", "1/8"),
        seventh=0.0,
        power_chords=True,
        drum_vel=1.08,
        fill="toms",
        crash=True,
        bass_gate=0.6,
        chord_gate=0.95,
        stab=True,
        riff=True,
    ),
    "Hip-Hop": GenreFeel(
        grooves=(
            {KICK: [0, 7, 10], SNARE: [4, 12], HAT: _EVEN},
            {
                KICK: [0, 3, 10],
                SNARE: [4, 12],
                HAT: _SIXTEENTHS,
                OPEN_HAT: [14],
            },
            {KICK: [0, 6, 11], SNARE: [8], CLAP: [8], HAT: _SIXTEENTHS},
        ),
        swing=0.35,
        progressions=((5, 4, 5, 3), (5, 3, 4, 5), (3, 4, 2, 5)),
        minor_progressions=((0, 0, 5, 6), (0, 5, 0, 3), (0, 2, 3, 6)),
        bass=("808", "808", "sparse"),
        chord_hits=([0.0], [0.0, 0.625], [0.0, 0.375]),
        hook=HookStyle(
            rhythms=(
                ((0, 2), (3, 2), (6, 3), (10, 2), (13, 3)),
                ((2, 3), (6, 2), (8, 4), (14, 2)),
                ((0, 6), (6, 2), (8, 2), (12, 4)),
                ((0, 2), (2, 1), (4, 3), (8, 2), (10, 1), (12, 3)),
                ((0, 3), (4, 2), (7, 3), (11, 2), (14, 2)),
            ),
            repeat=0.35,
            span=(2, 4),
            leaps=(3,),
            gate=0.62,
            register=70,
        ),
        arp_rates=("1/8", "1/16"),
        seventh=0.4,
        ghosts={SNARE: [7, 15]},
        fill="hats",
        bass_gate=0.97,
        chord_gate=0.95,
        strum=0.018,
        voicing="open",
    ),
    "Lo-Fi": GenreFeel(
        grooves=(
            {
                KICK: [0, 7, 10],
                SIDE_STICK: [4, 12],
                SHAKER: _EVEN,
            },
            {
                KICK: [0, 3, 10],
                SIDE_STICK: [4, 12],
                HAT: [2, 6, 10, 14],
                SHAKER: [1, 5, 9, 13],
            },
            {
                KICK: [0, 10],
                SIDE_STICK: [4, 12],
                SHAKER: _EVEN,
                TAMBOURINE: [4, 12],
            },
        ),
        swing=0.5,
        progressions=((1, 4, 0, 5), (3, 2, 1, 0), (0, 2, 3, 2)),
        minor_progressions=((3, 6, 2, 5), (1, 4, 0, 5), (0, 2, 3, 6)),
        bass=("sparse", "walking", "sparse"),
        chord_hits=([0.0], [0.0, 0.5], [0.0, 0.625]),
        hook=HookStyle(
            rhythms=(
                ((0, 4), (5, 3), (8, 4), (13, 3)),
                ((2, 3), (6, 3), (10, 3), (14, 2)),
                ((0, 3), (3, 2), (6, 4), (11, 5)),
                ((0, 2), (2, 2), (4, 6), (10, 2), (12, 4)),
                ((3, 3), (6, 2), (8, 6), (14, 2)),
            ),
            repeat=0.15,
            span=(3, 5),
            leaps=(3, 4),
            gate=0.97,
            register=70,
        ),
        arp_rates=("1/8 Triplet", "1/8"),
        seventh=0.85,
        drum_vel=0.85,
        ghosts={SIDE_STICK: [7, 15]},
        fill="none",
        bass_gate=0.95,
        chord_gate=1.0,
        strum=0.024,
        voicing="open",
    ),
    "Jazz": GenreFeel(
        grooves=(
            {KICK: [0, 10], RIDE: [0, 4, 6, 8, 12, 14], PEDAL_HAT: [4, 12]},
            {
                KICK: [0],
                RIDE: [0, 4, 6, 8, 12, 14],
                SIDE_STICK: [7, 15],
                PEDAL_HAT: [4, 12],
            },
            {
                KICK: [0, 11],
                RIDE: [0, 4, 6, 8, 12, 14],
                PEDAL_HAT: [4, 12],
                SNARE: [10],
            },
        ),
        swing=0.8,
        progressions=((1, 4, 0, 0), (0, 5, 1, 4), (2, 5, 1, 4)),
        minor_progressions=((1, 4, 0, 0), (1, 4, 0, 5), (0, 5, 1, 4)),
        bass=("walking", "walking", "sparse"),
        chord_hits=([0.0, 0.625], [0.0, 0.375], [0.25, 0.75]),
        hook=HookStyle(
            rhythms=(
                ((2, 2), (4, 2), (6, 2), (8, 2), (10, 3), (13, 3)),
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
                ((4, 2), (6, 2), (8, 4), (14, 2)),
                ((0, 3), (3, 1), (4, 2), (6, 2), (8, 4)),
                ((1, 2), (3, 2), (6, 2), (8, 3), (11, 2), (13, 3)),
            ),
            repeat=0.1,
            span=(4, 7),
            leaps=(3,),
            gate=0.74,
            register=71,
        ),
        arp_rates=("1/8 Triplet", "1/8 Triplet"),
        seventh=1.0,
        drum_vel=0.8,
        ghosts={SNARE: [7, 10, 15]},
        fill="none",
        bass_gate=0.95,
        chord_gate=0.55,
        strum=0.014,
        voicing="shell",
    ),
    "EDM": GenreFeel(
        grooves=(
            {KICK: [0, 4, 8, 12], CLAP: [4, 12], OPEN_HAT: [2, 6, 10, 14]},
            {
                KICK: [0, 4, 8, 12],
                CLAP: [4, 12],
                OPEN_HAT: [2, 6, 10, 14],
                HAT: list(range(1, 16, 2)),
            },
            {
                KICK: [0, 4, 8, 12],
                SNARE: [12],
                CLAP: [4],
                HAT: _SIXTEENTHS,
                TAMBOURINE: [4, 12],
            },
        ),
        swing=0.0,
        progressions=((5, 3, 0, 4), (3, 0, 4, 5), (0, 2, 3, 4)),
        minor_progressions=((0, 6, 5, 4), (5, 6, 0, 0), (0, 2, 6, 5)),
        bass=("offbeat", "pump", "offbeat"),
        chord_hits=(
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.375, 0.5, 0.875],
            [0.0, 0.5],
        ),
        hook=HookStyle(
            rhythms=(
                (
                    (0, 2),
                    (2, 2),
                    (4, 2),
                    (6, 2),
                    (8, 2),
                    (10, 2),
                    (12, 2),
                    (14, 2),
                ),
                ((0, 1), (3, 1), (6, 1), (8, 1), (10, 1), (12, 2)),
                ((0, 3), (3, 3), (6, 2), (8, 3), (11, 3), (14, 2)),
                ((2, 2), (6, 2), (10, 2), (14, 2)),
                ((0, 2), (2, 2), (4, 1), (5, 1), (6, 2), (8, 4)),
            ),
            repeat=0.45,
            span=(2, 5),
            leaps=(3, 4),
            gate=0.5,
            register=79,
        ),
        arp_rates=("1/16", "1/16"),
        seventh=0.2,
        drum_vel=1.05,
        fill="roll",
        crash=True,
        bass_gate=0.45,
        chord_gate=0.4,
        voicing="stack",
        stab=True,
    ),
}
DEFAULT_FEEL = "Pop"


def feel_for(style):
    return FEELS.get(style, FEELS[DEFAULT_FEEL])


def groove_for(rng, style):
    pattern = rng.choice(feel_for(style).grooves)
    return {pitch: list(steps) for pitch, steps in pattern.items()}


def progression_for(rng, style, kind, minor=False):
    feel = feel_for(style)
    pool = feel.minor_progressions if minor else ()
    chords = list(rng.choice(pool or feel.progressions))
    if kind in ("bridge", "break"):
        shift = rng.choice((1, 2))
        chords = chords[shift:] + chords[:shift]
    return chords


def apply_swing(instruments, bpm, swing):
    if swing <= 0:
        return
    beat = 60.0 / bpm
    eps = beat * 0.04
    for inst in instruments:
        for note in inst.notes:
            pos = (note.start / beat) % 1.0
            if abs(pos - 0.5) * beat < eps:
                shift = swing * _TRIPLET * beat
            elif min(abs(pos - 0.25), abs(pos - 0.75)) * beat < eps:
                shift = swing * _TRIPLET * beat / 2
            else:
                continue
            note.start += shift
            note.end += shift


def apply_strum(inst, strum):
    if strum <= 0:
        return
    groups = {}
    for note in inst.notes:
        groups.setdefault(round(note.start, 3), []).append(note)
    for notes in groups.values():
        for index, note in enumerate(sorted(notes, key=lambda n: n.pitch)):
            shift = min(index * strum, (note.end - note.start) / 2)
            note.start += shift
