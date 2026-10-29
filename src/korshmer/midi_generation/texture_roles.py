from dataclasses import dataclass

ENERGY_STEP = 0.08
MAX_ENTRY_ENERGY = 0.95
MIN_OCTAVE = 2
MAX_OCTAVE = 6
BATCH_OCTAVE_SHIFTS = (0, 1, -1)
ENERGY_SHAPE = (0.75, 0.25)
DEFAULT_BRIGHTNESS = 0.5


BRIGHTNESS = {
    42: 0.1,
    32: 0.1,
    33: 0.1,
    19: 0.15,
    16: 0.2,
    17: 0.2,
    18: 0.2,
    45: 0.25,
    24: 0.3,
    25: 0.3,
    26: 0.3,
    27: 0.35,
    28: 0.3,
    29: 0.35,
    0: 0.4,
    1: 0.4,
    2: 0.4,
    3: 0.4,
    7: 0.4,
    41: 0.35,
    40: 0.45,
    48: 0.4,
    49: 0.35,
    50: 0.45,
    52: 0.4,
    53: 0.4,
    61: 0.4,
    62: 0.45,
    4: 0.5,
    5: 0.5,
    89: 0.4,
    88: 0.55,
    94: 0.6,
    95: 0.6,
    91: 0.6,
    99: 0.55,
    81: 0.55,
    80: 0.6,
    83: 0.65,
    12: 0.7,
    11: 0.75,
    46: 0.75,
    8: 0.85,
    9: 0.9,
    10: 0.9,
    98: 0.9,
    100: 0.8,
    108: 0.95,
}


@dataclass(frozen=True)
class TextureRole:
    name: str
    kind: str
    octave: int
    min_energy: float = 0.0
    max_energy: float = 1.0
    root_first: bool = False
    avoid_root: bool = False
    velocity: float = 1.0
    lengths: tuple = ()
    beats: tuple = ()
    drum_pitches: tuple = ()
    entrance: bool = False
    answers: str = ""
    follows_chords: bool = False


DRUM_ROLES = (
    TextureRole(
        "Kick",
        "drums",
        0,
        min_energy=0.4,
        velocity=1.1,
        lengths=(1,),
        beats=(0, 2, 0, 3),
        drum_pitches=(36, 36, 35, 36),
    ),
    TextureRole(
        "Snare",
        "drums",
        0,
        min_energy=0.55,
        lengths=(1,),
        beats=(1, 3, 1, 3),
        drum_pitches=(38, 40, 38, 37),
    ),
    TextureRole(
        "Hats",
        "drums",
        0,
        min_energy=0.7,
        velocity=0.8,
        lengths=(0.5,),
        beats=(0.5, 1.5, 2.5, 3.5),
        drum_pitches=(42, 42, 46, 42),
    ),
)

LONG_ROLES = (
    TextureRole(
        "Drone",
        "long",
        2,
        root_first=True,
        velocity=0.8,
        lengths=(3,),
        follows_chords=True,
    ),
    TextureRole(
        "Foundation",
        "long",
        3,
        root_first=True,
        velocity=0.85,
        lengths=(2, 3),
        follows_chords=True,
    ),
    TextureRole(
        "Harmony",
        "long",
        4,
        min_energy=0.2,
        avoid_root=True,
        velocity=0.9,
        lengths=(2, 3),
        follows_chords=True,
    ),
    TextureRole(
        "Air",
        "long",
        5,
        max_energy=0.75,
        avoid_root=True,
        velocity=0.7,
        lengths=(3,),
    ),
    TextureRole(
        "Swell",
        "long",
        5,
        min_energy=0.75,
        root_first=True,
        velocity=1.0,
        lengths=(3,),
        entrance=True,
        follows_chords=True,
    ),
)

SHORT_ROLES = (
    TextureRole(
        "Bass Pulse",
        "short",
        2,
        min_energy=0.4,
        root_first=True,
        velocity=1.05,
        lengths=(1,),
        beats=(0, 2, 0, 2),
        follows_chords=True,
    ),
    TextureRole(
        "Ostinato",
        "short",
        3,
        min_energy=0.5,
        velocity=0.85,
        lengths=(0.5,),
    ),
    TextureRole(
        "Hook",
        "short",
        5,
        min_energy=0.45,
        root_first=True,
        velocity=1.05,
        lengths=(1, 0.5),
    ),
    TextureRole(
        "Answer",
        "short",
        5,
        min_energy=0.45,
        avoid_root=True,
        velocity=0.95,
        lengths=(1,),
        answers="Hook",
    ),
    TextureRole(
        "Counter",
        "short",
        4,
        min_energy=0.6,
        avoid_root=True,
        velocity=0.9,
        lengths=(1, 2),
        follows_chords=True,
    ),
    TextureRole(
        "Chime",
        "short",
        6,
        min_energy=0.7,
        velocity=0.75,
        lengths=(0.5,),
        entrance=True,
    ),
    TextureRole(
        "Sparkle",
        "short",
        6,
        min_energy=0.85,
        velocity=0.7,
        lengths=(0.5,),
        entrance=True,
    ),
)

BATCH_ROLES = DRUM_ROLES + LONG_ROLES + SHORT_ROLES


def role_for(position):
    return BATCH_ROLES[position % len(BATCH_ROLES)]


def batch_of(position):
    return position // len(BATCH_ROLES)


def role_octave(role, batch):
    shift = BATCH_OCTAVE_SHIFTS[batch % len(BATCH_OCTAVE_SHIFTS)]
    if batch == 0:
        shift = 0
    return max(MIN_OCTAVE, min(MAX_OCTAVE, role.octave + shift))


def entry_energy(role, batch):
    return min(MAX_ENTRY_ENERGY, role.min_energy + ENERGY_STEP * batch)


def slot_energies(bar_energy, total_slots, routine_bars):
    energies = []
    for slot in range(total_slots):
        bars = bar_energy[slot * routine_bars : (slot + 1) * routine_bars]
        energies.append(sum(bars) / len(bars) if bars else 0.0)
    return energies


def gate_slots(rng, slots, role, batch, energies):
    low = entry_energy(role, batch)
    high = role.max_energy
    wanted = [i for i, e in enumerate(energies) if low <= e <= high]
    kept = {s for s in slots if low <= energies[s] <= high}
    if role.entrance:
        kept |= {
            s for s in wanted if s == 0 or not (low <= energies[s - 1] <= high)
        }
    if not kept:
        if wanted:
            kept = {rng.choice(wanted)}
        else:
            middle = (low + min(high, 1.0)) / 2
            kept = {
                min(
                    range(len(energies)),
                    key=lambda i: abs(energies[i] - middle),
                )
            }
    return kept


def velocity_for(role, energy):
    base, span = ENERGY_SHAPE
    return role.velocity * (base + span * energy)


def brightness(program):
    return BRIGHTNESS.get(program, DEFAULT_BRIGHTNESS)


def match_voices(picks, roles, batch):
    ranked = sorted(picks, key=lambda pick: brightness(pick[1]))
    order = sorted(
        range(len(roles)), key=lambda i: role_octave(roles[i], batch)
    )
    matched = [None] * len(roles)
    for slot, pick in zip(order, ranked, strict=True):
        matched[slot] = pick
    return matched


def chord_shift(chord_root, intervals):

    def distance(degree):
        gap = abs(intervals[degree] - chord_root) % 12
        return min(gap, 12 - gap)

    return min(range(len(intervals)), key=lambda d: (distance(d), d))
