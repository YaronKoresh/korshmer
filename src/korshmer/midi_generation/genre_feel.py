from dataclasses import dataclass, field, replace
from functools import cache

from korshmer.midi_generation.genres import blended, parts

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
    min_notes: int = 4
    max_notes: int = 8
    shift: float = 0.5
    room: tuple = (-4, 5)


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
    fills: tuple = ()
    crash: bool = False
    bass_gate: float = 0.9
    chord_gate: float = 0.9
    strum: float = 0.0
    voicing: str = "close"
    stab: bool = False
    riff: bool = False
    arp_rates: tuple = ("1/8", "1/16")
    accents: dict = field(default_factory=dict)
    hat_rolls: tuple = ()
    roll_chance: float = 0.0
    bass_floor: int = 33
    bass_on_kicks: bool = False
    bass_jump: float = 0.0
    home_modes: tuple = ()
    phrases: dict = field(default_factory=dict)
    lead_snaps_inside: bool = False
    chord_spans: tuple = (1,)
    hook_holds: bool = False


FEELS = {
    "Pop": GenreFeel(
        chord_spans=(1, 1, 1, 2),
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
            {KICK: [0, 3, 8, 10], SNARE: [4, 12], HAT: _EVEN},
            {KICK: [0, 7, 8], SNARE: [4, 12], CLAP: [12], HAT: _SIXTEENTHS},
            {
                KICK: [0, 8, 14],
                CLAP: [4, 12],
                HAT: _EVEN,
                TAMBOURINE: [2, 6, 10, 14],
            },
            {KICK: [0, 5, 8, 11], SNARE: [4, 12], HAT: _EVEN, OPEN_HAT: [6]},
        ),
        swing=0.0,
        progressions=(
            (0, 4, 5, 3),
            (0, 5, 3, 4),
            (0, 3, 5, 4),
            (0, 2, 5, 3),
            (3, 4, 0, 5),
            (0, 1, 3, 4),
            (5, 4, 0, 3),
            (0, 3, 1, 4),
            (0, 4, 1, 3),
        ),
        minor_progressions=(
            (0, 5, 2, 6),
            (0, 6, 5, 6),
            (0, 3, 6, 2),
            (0, 5, 6, 4),
            (0, 2, 6, 3),
            (5, 6, 2, 0),
            (0, 3, 5, 2),
            (3, 5, 6, 0),
            (0, 6, 2, 5),
        ),
        bass=("eighths", "syncopated", "eighths", "octaves", "push"),
        chord_hits=(
            [0.0, 0.5],
            [0.0, 0.375, 0.5],
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.5, 0.75],
            [0.0, 0.25, 0.625],
            [0.0, 0.375, 0.75],
        ),
        hook=HookStyle(
            rhythms=(
                ((0, 3), (3, 3), (6, 2), (8, 3), (11, 3)),
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)),
                ((2, 2), (4, 2), (6, 2), (8, 3), (11, 2), (13, 3)),
                ((0, 4), (4, 2), (6, 2), (8, 3), (11, 5)),
                ((0, 2), (3, 1), (4, 2), (8, 2), (11, 1), (12, 4)),
                ((0, 3), (3, 3), (6, 3), (9, 3), (12, 4)),
                ((0, 2), (2, 2), (4, 4), (8, 2), (10, 2), (12, 4)),
                ((0, 3), (3, 1), (4, 4), (8, 3), (11, 1), (12, 4)),
                ((2, 2), (4, 4), (8, 2), (10, 2), (12, 2), (14, 2)),
                ((0, 4), (4, 4), (8, 2), (10, 2), (12, 4)),
                (
                    (0, 2),
                    (2, 1),
                    (3, 3),
                    (6, 2),
                    (8, 2),
                    (10, 1),
                    (11, 3),
                    (14, 2),
                ),
                ((0, 6), (6, 2), (8, 6), (14, 2)),
            ),
            repeat=0.2,
            span=(3, 6),
            leaps=(3, 4),
            gate=0.88,
            register=74,
        ),
        arp_rates=("1/8", "1/16"),
        seventh=0.15,
        fills=("snare", "push", "roll"),
        fill="snare",
        crash=True,
        bass_gate=0.85,
        chord_gate=0.85,
        strum=0.008,
    ),
    "Rock": GenreFeel(
        chord_spans=(1, 1, 2),
        grooves=(
            {KICK: [0, 8, 10], SNARE: [4, 12], HAT: _EVEN},
            {
                KICK: [0, 2, 8, 10],
                SNARE: [4, 12],
                OPEN_HAT: [6, 14],
                HAT: [0, 2, 4, 8, 10, 12],
            },
            {KICK: [0, 6, 8], SNARE: [4, 12], RIDE: _EVEN},
            {KICK: [0, 8], SNARE: [4, 12], HAT: _EVEN, OPEN_HAT: [14]},
            {KICK: [0, 3, 8, 11], SNARE: [4, 12], HAT: _EVEN},
            {KICK: [0, 6, 8, 14], SNARE: [4, 12], RIDE: _EVEN},
            {KICK: [0, 7, 8, 10], SNARE: [4, 12], OPEN_HAT: [0, 4, 8, 12]},
        ),
        swing=0.0,
        progressions=(
            (0, 3, 4, 3),
            (0, 3, 0, 4),
            (0, 4, 3, 4),
            (0, 0, 3, 4),
            (0, 3, 4, 0),
            (3, 0, 4, 3),
            (0, 5, 4, 0),
            (0, 4, 3, 0),
            (0, 3, 3, 4),
        ),
        minor_progressions=(
            (0, 5, 6, 0),
            (0, 3, 4, 0),
            (0, 6, 3, 0),
            (0, 6, 5, 0),
            (0, 3, 6, 0),
            (0, 6, 0, 5),
            (0, 5, 3, 0),
            (0, 6, 4, 0),
            (0, 4, 6, 0),
            (0, 2, 5, 6),
            (0, 2, 3, 6),
            (0, 6, 5, 4),
            (0, 3, 6, 5),
        ),
        bass=("driving", "driving", "syncopated", "gallop", "chug"),
        chord_hits=(
            [0.0, 0.5],
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.375, 0.5, 0.75],
            [0.0, 0.375, 0.5],
            [0.0, 0.5, 0.75],
            [0.0, 0.25, 0.5, 0.625, 0.75],
        ),
        hook=HookStyle(
            rhythms=(
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
                ((0, 3), (3, 3), (6, 2), (8, 4), (12, 4)),
                ((0, 2), (3, 1), (4, 2), (8, 2), (11, 1), (12, 4)),
                ((0, 4), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)),
                ((0, 2), (2, 2), (5, 2), (8, 2), (10, 2), (13, 3)),
                ((0, 2), (2, 2), (4, 4), (8, 2), (10, 2), (12, 2), (14, 2)),
                ((0, 3), (3, 3), (6, 2), (8, 2), (10, 2), (12, 4)),
                ((0, 6), (6, 2), (8, 4), (12, 4)),
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 4), (12, 4)),
                ((2, 2), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)),
                ((0, 4), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
            ),
            repeat=0.35,
            span=(3, 5),
            leaps=(3, 4),
            gate=0.78,
            register=71,
            min_notes=5,
        ),
        arp_rates=("1/8", "1/8"),
        seventh=0.0,
        power_chords=True,
        drum_vel=1.08,
        fills=("toms", "toms_down"),
        fill="toms",
        crash=True,
        bass_gate=0.6,
        chord_gate=0.95,
        stab=True,
        riff=True,
    ),
    "Hip-Hop": GenreFeel(
        chord_spans=(1, 2, 2),
        grooves=(
            {KICK: [0, 7, 10], SNARE: [4, 12], HAT: _EVEN},
            {
                KICK: [0, 3, 10],
                SNARE: [4, 12],
                HAT: _SIXTEENTHS,
                OPEN_HAT: [14],
            },
            {KICK: [0, 6, 11], SNARE: [8], CLAP: [8], HAT: _SIXTEENTHS},
            {KICK: [0, 10], SNARE: [4, 12], HAT: _EVEN, OPEN_HAT: [6]},
            {KICK: [0, 7, 8], SNARE: [4, 12], HAT: _EVEN},
            {KICK: [0, 5, 10], SNARE: [4, 12], CLAP: [12], HAT: _SIXTEENTHS},
            {
                KICK: [0, 3, 7, 10],
                SNARE: [4, 12],
                HAT: [0, 2, 4, 6, 8, 10, 12],
                OPEN_HAT: [14],
            },
        ),
        swing=0.35,
        progressions=(
            (5, 4, 5, 3),
            (5, 3, 4, 5),
            (3, 4, 2, 5),
            (5, 3, 5, 0),
            (5, 5, 2, 3),
            (3, 5, 2, 5),
            (5, 2, 3, 5),
            (2, 5, 3, 5),
            (5, 3, 5, 5),
        ),
        minor_progressions=(
            (0, 0, 5, 6),
            (0, 5, 0, 3),
            (0, 2, 3, 6),
            (0, 3, 0, 5),
            (0, 6, 3, 5),
            (0, 0, 2, 6),
            (0, 4, 5, 0),
            (0, 5, 2, 3),
            (0, 5, 3, 2),
        ),
        bass=("808", "808", "stutter", "drop"),
        chord_hits=(
            [0.0],
            [0.0, 0.625],
            [0.0, 0.375],
            [0.0, 0.5],
            [0.0, 0.75],
        ),
        hook=HookStyle(
            rhythms=(
                ((0, 2), (3, 2), (6, 3), (10, 2), (13, 3)),
                ((2, 3), (6, 2), (8, 4), (14, 2)),
                ((0, 6), (6, 2), (8, 2), (12, 4)),
                ((0, 2), (2, 1), (4, 3), (8, 2), (10, 1), (12, 3)),
                ((0, 3), (4, 2), (7, 3), (11, 2), (14, 2)),
                ((0, 3), (3, 3), (6, 2), (8, 4), (13, 3)),
                ((0, 2), (2, 2), (5, 3), (8, 2), (10, 2), (13, 3)),
                ((1, 2), (3, 3), (6, 2), (8, 3), (12, 4)),
                ((0, 4), (6, 2), (8, 2), (10, 2), (14, 2)),
                ((0, 2), (3, 3), (8, 2), (11, 3), (14, 2)),
                (
                    (0, 1),
                    (1, 1),
                    (2, 2),
                    (6, 2),
                    (8, 1),
                    (9, 1),
                    (10, 2),
                    (14, 2),
                ),
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
        fills=("hats", "snare", "triplets", "push"),
        fill="hats",
        bass_gate=0.97,
        chord_gate=0.95,
        strum=0.018,
        voicing="open",
    ),
    "Lo-Fi": GenreFeel(
        chord_spans=(1, 1, 2),
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
            {KICK: [0, 5, 10], SIDE_STICK: [4, 12], HAT: _EVEN},
            {KICK: [0, 8, 11], SIDE_STICK: [4, 12], SHAKER: _EVEN, HAT: [14]},
            {
                KICK: [0, 6, 10],
                SIDE_STICK: [4, 12],
                TAMBOURINE: [2, 6, 10, 14],
                SHAKER: [0, 4, 8, 12],
            },
            {KICK: [0, 7], SIDE_STICK: [4, 12], SHAKER: _SIXTEENTHS},
        ),
        swing=0.5,
        progressions=(
            (1, 4, 0, 5),
            (3, 2, 1, 0),
            (0, 2, 3, 2),
            (1, 4, 2, 5),
            (3, 1, 2, 5),
            (0, 5, 1, 3),
            (3, 2, 0, 4),
            (1, 2, 0, 5),
            (5, 1, 4, 2),
        ),
        minor_progressions=(
            (3, 6, 2, 5),
            (1, 4, 0, 5),
            (0, 2, 3, 6),
            (3, 1, 4, 5),
            (5, 3, 4, 1),
            (3, 0, 4, 6),
            (0, 4, 3, 2),
            (5, 2, 4, 1),
            (3, 6, 4, 0),
        ),
        bass=("sparse", "walking", "sparse", "lazy", "two"),
        chord_hits=(
            [0.0],
            [0.0, 0.5],
            [0.0, 0.625],
            [0.0, 0.375],
            [0.0, 0.75],
        ),
        hook=HookStyle(
            rhythms=(
                ((0, 4), (5, 3), (8, 4), (13, 3)),
                ((2, 3), (6, 3), (10, 3), (14, 2)),
                ((0, 3), (3, 2), (6, 4), (11, 5)),
                ((0, 2), (2, 2), (4, 6), (10, 2), (12, 4)),
                ((3, 3), (6, 2), (8, 6), (14, 2)),
                ((0, 6), (6, 2), (8, 4), (12, 4)),
                ((2, 4), (6, 2), (8, 6), (14, 2)),
                ((0, 3), (4, 4), (8, 3), (12, 4)),
                ((0, 4), (4, 2), (6, 6), (12, 4)),
                ((1, 3), (4, 4), (9, 3), (12, 4)),
                ((0, 2), (3, 3), (6, 4), (10, 2), (13, 3)),
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
        fills=("none", "rim"),
        fill="none",
        bass_gate=0.95,
        chord_gate=1.0,
        strum=0.024,
        voicing="open",
    ),
    "Jazz": GenreFeel(
        chord_spans=(1, 1, 1, 2),
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
            {KICK: [0, 8], RIDE: [0, 4, 6, 8, 12, 14], PEDAL_HAT: [4, 12]},
            {
                KICK: [0, 6],
                RIDE: [0, 4, 6, 8, 12, 14],
                PEDAL_HAT: [4, 12],
                SIDE_STICK: [10],
            },
            {
                KICK: [0],
                RIDE: [0, 4, 8, 10, 12, 14],
                PEDAL_HAT: [4, 12],
                SNARE: [6],
            },
            {
                KICK: [0, 14],
                RIDE: [0, 4, 6, 8, 12],
                PEDAL_HAT: [4, 12],
                SIDE_STICK: [14],
            },
        ),
        swing=0.8,
        progressions=(
            (1, 4, 0, 0),
            (0, 5, 1, 4),
            (2, 5, 1, 4),
            (1, 4, 0, 3),
            (0, 2, 1, 4),
            (3, 2, 1, 4),
            (0, 3, 2, 5),
            (5, 1, 4, 0),
            (1, 4, 5, 0),
        ),
        minor_progressions=(
            (1, 4, 0, 0),
            (1, 4, 0, 5),
            (0, 5, 1, 4),
            (0, 3, 1, 4),
            (1, 4, 0, 3),
            (5, 1, 4, 0),
            (3, 1, 4, 0),
            (0, 2, 1, 4),
            (3, 6, 1, 4),
        ),
        bass=("walking", "walking", "sparse", "two", "walking"),
        chord_hits=(
            [0.0, 0.625],
            [0.0, 0.375],
            [0.25, 0.75],
            [0.0, 0.5],
            [0.0, 0.375, 0.75],
            [0.125, 0.625],
        ),
        hook=HookStyle(
            rhythms=(
                ((2, 2), (4, 2), (6, 2), (8, 2), (10, 3), (13, 3)),
                ((0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
                ((4, 2), (6, 2), (8, 4), (14, 2)),
                ((0, 3), (3, 1), (4, 2), (6, 2), (8, 4)),
                ((1, 2), (3, 2), (6, 2), (8, 3), (11, 2), (13, 3)),
                ((0, 2), (2, 2), (4, 2), (6, 3), (10, 2), (12, 2), (14, 2)),
                ((2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 4)),
                ((0, 3), (3, 3), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)),
                ((0, 4), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)),
                ((1, 1), (2, 2), (4, 2), (6, 2), (9, 3), (12, 4)),
                ((0, 2), (2, 2), (4, 4), (10, 2), (12, 2), (14, 2)),
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
        fills=("none", "rim"),
        fill="none",
        bass_gate=0.95,
        chord_gate=0.55,
        strum=0.014,
        voicing="shell",
    ),
    "EDM": GenreFeel(
        chord_spans=(1, 1, 2),
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
            {KICK: [0, 4, 8, 12], CLAP: [4, 12], HAT: _SIXTEENTHS},
            {
                KICK: [0, 4, 8, 12],
                SNARE: [4, 12],
                OPEN_HAT: [2, 6, 10, 14],
                SHAKER: _SIXTEENTHS,
            },
            {
                KICK: [0, 4, 8, 12],
                CLAP: [4, 12],
                OPEN_HAT: [2, 6, 10, 14],
                HAT: [0, 3, 6, 9, 12, 15],
            },
            {
                KICK: [0, 4, 8, 12],
                CLAP: [4, 12],
                OPEN_HAT: [2, 6, 10, 14],
                TAMBOURINE: [4, 12],
                SHAKER: _EVEN,
            },
        ),
        swing=0.0,
        progressions=(
            (5, 3, 0, 4),
            (3, 0, 4, 5),
            (0, 2, 3, 4),
            (5, 0, 4, 3),
            (3, 5, 0, 4),
            (0, 5, 2, 3),
            (5, 4, 3, 0),
            (3, 4, 5, 0),
            (5, 0, 3, 4),
        ),
        minor_progressions=(
            (0, 6, 5, 4),
            (5, 6, 0, 0),
            (0, 2, 6, 5),
            (0, 5, 6, 6),
            (5, 6, 0, 5),
            (0, 3, 5, 6),
            (3, 0, 5, 6),
            (0, 5, 4, 6),
            (5, 0, 6, 3),
        ),
        bass=("offbeat", "pump", "offbeat", "skip", "bounce"),
        chord_hits=(
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.375, 0.5, 0.875],
            [0.0, 0.5],
            [0.0, 0.375, 0.75],
            [0.125, 0.375, 0.625, 0.875],
            [0.0, 0.1875, 0.375, 0.5, 0.75],
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
                (
                    (0, 2),
                    (2, 1),
                    (3, 2),
                    (6, 2),
                    (8, 2),
                    (10, 1),
                    (11, 2),
                    (14, 2),
                ),
                ((0, 3), (3, 3), (6, 3), (9, 3), (12, 2), (14, 2)),
                ((0, 1), (2, 1), (4, 1), (6, 1), (8, 2), (10, 2), (12, 4)),
                ((0, 4), (4, 2), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)),
                ((0, 2), (3, 2), (6, 2), (8, 2), (11, 2), (14, 2)),
                ((0, 2), (4, 2), (6, 2), (8, 2), (12, 2), (14, 2)),
            ),
            repeat=0.45,
            span=(2, 5),
            leaps=(3, 4),
            gate=0.5,
            register=77,
        ),
        arp_rates=("1/16", "1/16"),
        seventh=0.2,
        drum_vel=1.05,
        fills=("roll", "snare", "push"),
        fill="roll",
        crash=True,
        bass_gate=0.45,
        chord_gate=0.4,
        voicing="stack",
        stab=True,
    ),
    "Phonk": GenreFeel(
        chord_spans=(1, 2, 2),
        grooves=(
            {
                KICK: [0, 6, 10],
                CLAP: [8],
                HAT: [0, 2, 4, 6, 8, 10, 12],
                OPEN_HAT: [14],
            },
            {
                KICK: [0, 3, 10],
                CLAP: [8],
                SNARE: [8],
                HAT: _EVEN,
            },
            {
                KICK: [0, 7, 11],
                CLAP: [8],
                HAT: [0, 2, 3, 4, 6, 8, 10, 11, 12, 14],
                OPEN_HAT: [6],
            },
            {KICK: [0, 6, 12], CLAP: [8], HAT: _EVEN},
            {KICK: [0, 10, 14], CLAP: [8], HAT: _EVEN, OPEN_HAT: [4]},
            {KICK: [0, 3, 11], CLAP: [8], SNARE: [8], HAT: _EVEN},
            {KICK: [0, 7, 10], CLAP: [8], HAT: [0, 2, 4, 6, 7, 8, 10, 12, 14]},
        ),
        swing=0.0,
        progressions=(
            (0, 0, 5, 6),
            (0, 0, 0, 5),
            (0, 5, 0, 6),
            (0, 0, 6, 5),
            (0, 0, 0, 6),
            (0, 0, 5, 3),
            (0, 0, 3, 5),
            (0, 0, 6, 3),
        ),
        minor_progressions=(
            (0, 0, 5, 6),
            (0, 0, 0, 5),
            (0, 0, 3, 0),
            (0, 5, 0, 6),
            (0, 0, 6, 5),
            (0, 0, 0, 6),
            (0, 0, 5, 3),
            (0, 0, 3, 5),
            (0, 0, 6, 3),
            (0, 0, 3, 6),
            (0, 6, 0, 3),
            (0, 3, 0, 6),
            (0, 6, 0, 5),
            (0, 5, 3, 6),
            (0, 6, 5, 3),
            (0, 5, 6, 3),
        ),
        bass=("sub",),
        chord_hits=(
            [0.0],
            [0.0],
            [0.0, 0.75],
            [0.0, 0.625],
            [0.0, 0.5],
        ),
        hook=HookStyle(
            rhythms=(
                (
                    (0, 1),
                    (2, 1),
                    (3, 1),
                    (6, 1),
                    (8, 1),
                    (10, 1),
                    (11, 1),
                    (14, 1),
                ),
                ((0, 2), (3, 2), (6, 2), (8, 2), (11, 2), (14, 1), (15, 1)),
                (
                    (0, 1),
                    (2, 1),
                    (4, 1),
                    (6, 1),
                    (7, 1),
                    (10, 1),
                    (12, 1),
                    (15, 1),
                ),
                (
                    (0, 1),
                    (3, 1),
                    (4, 1),
                    (7, 1),
                    (8, 1),
                    (11, 1),
                    (12, 1),
                    (14, 1),
                ),
                (
                    (0, 1),
                    (1, 1),
                    (3, 1),
                    (6, 1),
                    (8, 1),
                    (9, 1),
                    (11, 1),
                    (14, 1),
                ),
                (
                    (0, 1),
                    (2, 1),
                    (3, 1),
                    (5, 1),
                    (8, 1),
                    (10, 1),
                    (11, 1),
                    (13, 1),
                ),
                (
                    (0, 1),
                    (3, 1),
                    (6, 1),
                    (8, 1),
                    (9, 1),
                    (11, 1),
                    (14, 1),
                    (15, 1),
                ),
                (
                    (0, 2),
                    (2, 1),
                    (3, 1),
                    (6, 2),
                    (8, 2),
                    (10, 1),
                    (11, 1),
                    (14, 2),
                ),
                (
                    (0, 1),
                    (1, 1),
                    (4, 1),
                    (6, 1),
                    (8, 1),
                    (9, 1),
                    (12, 1),
                    (14, 1),
                ),
                (
                    (0, 1),
                    (2, 1),
                    (5, 1),
                    (6, 1),
                    (8, 1),
                    (10, 1),
                    (13, 1),
                    (14, 1),
                ),
            ),
            repeat=0.45,
            span=(2, 4),
            leaps=(2, 3),
            gate=0.8,
            register=64,
            min_notes=8,
            room=(-2, 3),
        ),
        arp_rates=("1/8", "1/8"),
        seventh=0.0,
        power_chords=True,
        drum_vel=1.12,
        fills=("hats", "triplets"),
        fill="hats",
        crash=True,
        bass_gate=0.97,
        chord_gate=0.95,
        accents={KICK: 112, CLAP: 108, SNARE: 104},
        hat_rolls=(
            (14, 2, 4),
            (6, 2, 4),
            (12, 2, 3),
            (10, 2, 3),
            (12, 4, 6),
            (4, 2, 3),
        ),
        roll_chance=0.6,
        bass_floor=32,
        bass_on_kicks=True,
        bass_jump=0.3,
        home_modes=("Minor", "Phrygian"),
        lead_snaps_inside=True,
        hook_holds=True,
        phrases={
            "chorus": (
                ("hook", "hook", "hook", "vary"),
                ("hook", "hook", "lift", "vary"),
            ),
            "verse": (
                ("hook", "hook", "hook", "vary"),
                ("hook", "skeleton", "hook", "vary"),
            ),
            "intro": (
                ("hook", "hook", "hook", "vary"),
                ("hook", "hook", "hook", "vary"),
            ),
            "outro": (
                ("hook", "hook", "hook", "vary"),
                ("hook", "hook", "hook", "vary"),
            ),
        },
    ),
}
DEFAULT_FEEL = "Pop"
HOME_CHORD = 0
DEPARTING_KINDS = ("bridge", "break")
OPENING_KIND = "intro"
OPENS_AT_HOME = 0.4
HOOK_KIND = "chorus"
HOOK_BARS = 3


@cache
def _blend_feel(names):
    lead, *others = (FEELS[name] for name in names)
    rhythms = [
        rhythm
        for other in others
        for rhythm in other.hook.rhythms
        if lead.hook.min_notes <= len(rhythm) <= lead.hook.max_notes
    ]
    return replace(
        lead,
        grooves=blended(lead.grooves, (o.grooves for o in others)),
        progressions=blended(
            lead.progressions, (o.progressions for o in others)
        ),
        minor_progressions=blended(
            lead.minor_progressions or lead.progressions,
            (o.minor_progressions or o.progressions for o in others),
        ),
        bass=blended(lead.bass, (o.bass for o in others)),
        chord_hits=blended(lead.chord_hits, (o.chord_hits for o in others)),
        fills=blended(lead.fills, (o.fills for o in others)),
        hat_rolls=blended(lead.hat_rolls, (o.hat_rolls for o in others)),
        hook=replace(lead.hook, rhythms=blended(lead.hook.rhythms, (rhythms,))),
    )


def feel_for(style):
    if style in FEELS:
        return FEELS[style]
    names = tuple(name for name in parts(style) if name in FEELS)
    if not names:
        return FEELS[DEFAULT_FEEL]
    if len(names) == 1:
        return FEELS[names[0]]
    return _blend_feel(names)


def groove_for(rng, style):
    pattern = rng.choice(feel_for(style).grooves)
    return {pitch: list(steps) for pitch, steps in pattern.items()}


def progression_for(rng, style, kind, minor=False):
    feel = feel_for(style)
    pool = (feel.minor_progressions if minor else ()) or feel.progressions
    if feel.hook_holds and kind == HOOK_KIND:
        pool = [p for p in pool if len(set(p)) < len(p)]
    chords = list(rng.choice(pool))
    if kind in ("bridge", "break"):
        shift = rng.choice((1, 2))
        chords = chords[shift:] + chords[:shift]
    return chords


def _turned(chords, turn):
    return chords[turn:] + chords[:turn]


def _kept(turns, chords, fits):
    return [t for t in turns if fits(_turned(chords, t))] or turns


def lay_out(rng, style, kind, chords, anchored=False):
    feel = feel_for(style)
    chords = list(chords)
    turns = list(range(len(chords)))
    span = 1 if kind == HOOK_KIND else rng.choice(feel.chord_spans)
    if kind in DEPARTING_KINDS:
        turns = _kept(turns, chords, lambda c: c[0] != HOME_CHORD)
    if feel.hook_holds and kind == HOOK_KIND:
        turns = _kept(turns, chords, _held)
    opening = kind == OPENING_KIND and rng.random() < OPENS_AT_HOME
    turn = turns[0] if anchored or opening else rng.choice(turns)
    return settle(style, kind, _turned(chords, turn)), span


def _held(chords):
    return len(set(chords[:HOOK_BARS])) < HOOK_BARS


def settle(style, kind, chords):
    chords = list(chords)
    if kind in DEPARTING_KINDS and chords[0] == HOME_CHORD:
        away = [t for t, chord in enumerate(chords) if chord != HOME_CHORD]
        chords = _turned(chords, away[0]) if away else chords
    if kind == HOOK_KIND and feel_for(style).hook_holds and not _held(chords):
        chords[1] = chords[0]
    return chords


def spread(chords, span):
    return [chord for chord in chords for _ in range(span)]


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
