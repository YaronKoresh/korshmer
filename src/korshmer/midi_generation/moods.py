from dataclasses import dataclass

DEFAULT_MOOD = "Uplifting"
BRIGHT, NEUTRAL, DARK = 1, 0, -1
SPARSE, PLAIN, BUSY = -1, 0, 1


@dataclass(frozen=True)
class MoodFeel:
    notes: int
    density: float
    sustain: bool
    register: int
    moves: tuple
    leap: float
    leap_turn: tuple
    leaps_liked: tuple
    repeat: float
    shapes: tuple
    gate: float
    velocity: int
    band: float
    drums: float
    harmony: int
    seventh: float
    hits: int
    bass: int
    kick_variation: float
    fills: tuple
    pads: tuple


MOOD_FEELS = {
    "Uplifting": MoodFeel(
        notes=6,
        density=1.0,
        sustain=True,
        register=2,
        moves=(-1, 1, 1, 1, 2, -2),
        leap=0.14,
        leap_turn=(1, 1, 1, -1),
        leaps_liked=(1,),
        repeat=0.0,
        shapes=("rise", "arch", "rise", "wave"),
        gate=1.0,
        velocity=2,
        band=1.0,
        drums=1.0,
        harmony=BRIGHT,
        seventh=0.0,
        hits=PLAIN,
        bass=PLAIN,
        kick_variation=0.15,
        fills=(4, 4, 8),
        pads=("triad", "octave", "triad"),
    ),
    "Melancholic": MoodFeel(
        notes=4,
        density=0.8,
        sustain=False,
        register=-3,
        moves=(-1, -1, -1, 1, -2, 1),
        leap=0.05,
        leap_turn=(-1, -1, -1, 1),
        leaps_liked=(0, 1),
        repeat=0.05,
        shapes=("fall", "valley", "fall", "arch"),
        gate=1.1,
        velocity=-9,
        band=0.88,
        drums=0.92,
        harmony=DARK,
        seventh=0.15,
        hits=SPARSE,
        bass=SPARSE,
        kick_variation=0.0,
        fills=(8,),
        pads=("triad", "low", "low"),
    ),
    "Energetic": MoodFeel(
        notes=7,
        density=1.35,
        sustain=True,
        register=0,
        moves=(-2, -1, 1, 2, 1, -1),
        leap=0.12,
        leap_turn=(1, -1),
        leaps_liked=(1,),
        repeat=0.18,
        shapes=("wave", "pedal", "rise", "wave"),
        gate=0.68,
        velocity=5,
        band=1.1,
        drums=1.06,
        harmony=NEUTRAL,
        seventh=-0.1,
        hits=BUSY,
        bass=BUSY,
        kick_variation=0.5,
        fills=(4,),
        pads=("pulse", "triad", "pulse"),
    ),
    "Dreamy": MoodFeel(
        notes=4,
        density=0.85,
        sustain=True,
        register=5,
        moves=(-1, 1, 1, -1, 2, -2),
        leap=0.12,
        leap_turn=(1, -1, 1),
        leaps_liked=(1, 0),
        repeat=0.08,
        shapes=("arch", "wave", "valley", "arch"),
        gate=1.15,
        velocity=-13,
        band=0.82,
        drums=0.88,
        harmony=NEUTRAL,
        seventh=0.6,
        hits=SPARSE,
        bass=SPARSE,
        kick_variation=0.0,
        fills=(8,),
        pads=("open", "open", "octave"),
    ),
}


def mood_for(name):
    return MOOD_FEELS.get(name, MOOD_FEELS[DEFAULT_MOOD])


def triad_brightness(intervals, degree):
    count = len(intervals)

    def pitch(step):
        return intervals[step % count] + 12 * (step // count)

    third = pitch(degree + 2) - pitch(degree)
    fifth = pitch(degree + 4) - pitch(degree)
    if third == 4 and fifth == 7:
        return 1.0
    if third == 3 and fifth == 7:
        return -1.0
    return -1.5 if third == 3 else 0.0


def progression_brightness(intervals, chords):
    if len(intervals) < 5:
        return 0.0
    weights = [2.0] + [1.0] * (len(chords) - 1)
    total = sum(
        w * triad_brightness(intervals, d)
        for w, d in zip(weights, chords, strict=True)
    )
    return total / sum(weights)


def pick_by_mood(options, key, leaning):
    options = list(options)
    if leaning > 0:
        return max(options, key=key)
    if leaning < 0:
        return min(options, key=key)
    return options[0]
