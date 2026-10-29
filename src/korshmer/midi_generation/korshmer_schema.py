import math
import random

import pretty_midi

from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.midi_generation import texture_roles
from korshmer.utils.tempo import validate_bpm

SCHEMA_SCALES = ("Pentatonic Major", "Pentatonic Minor")

DRUM_TRACKS = 3
LONG_TRACKS = 5
SHORT_TRACKS = 7
TRACKS_PER_ACTIVATION = DRUM_TRACKS + LONG_TRACKS + SHORT_TRACKS

ROUTINE_BARS = 4
BEATS_PER_BAR = 4
DEFAULT_MODULUS = 10
MAX_TRACK_INDEX = DEFAULT_MODULUS
MIN_BATCHES = 2
MAX_TRACKS = 6 * TRACKS_PER_ACTIVATION
MIN_ACTIVE_TRACKS = 2
MIN_MINUTES = 0.5
MAX_MINUTES = 120.0
DEFAULT_MINUTES = 3.0

LONG_PROGRAMS = {
    "Violin": 40,
    "Cello": 42,
    "String Ensemble": 48,
    "Synth Strings": 50,
    "Choir Aahs": 52,
    "Church Organ": 19,
    "Pad Warm": 89,
    "Pad Halo": 94,
    "Saw Lead": 81,
}

SHORT_PROGRAMS = {
    "Acoustic Piano": 0,
    "Electric Piano": 4,
    "Marimba": 12,
    "Acoustic Guitar": 24,
    "Electric Guitar": 27,
    "Pizzicato Strings": 45,
    "Harp": 46,
    "Kalimba": 108,
    "Vibraphone": 11,
}

LONG_LENGTHS = (1, 2, 3)
SHORT_LENGTHS = (2, 1, 0.5)
DRUM_LENGTHS = (1,)
NOTE_GAP = 0.98

SCHEMA_OCTAVES = tuple(range(2, 8))
GM_PERCUSSION_RANGE = (35, 81)


def track_index(zero_based_index, modulus=DEFAULT_MODULUS):
    return int(zero_based_index) % int(modulus) + 1


def period_bars(index):
    return ROUTINE_BARS * (int(index) + 1)


BATCH_BARS = period_bars(DEFAULT_MODULUS)


def validate_minutes(minutes):
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        raise ValueError("The length in minutes must be a number.") from None
    if not math.isfinite(value) or not MIN_MINUTES <= value <= MAX_MINUTES:
        raise ValueError(
            f"The length must be between {MIN_MINUTES:g} and "
            f"{MAX_MINUTES:g} minutes."
        )
    return value


def bars_for_minutes(minutes, bpm):
    seconds_per_bar = 60.0 / validate_bpm(bpm) * BEATS_PER_BAR
    bars = math.ceil(validate_minutes(minutes) * 60.0 / seconds_per_bar)
    return max(ROUTINE_BARS, -(-bars // ROUTINE_BARS) * ROUTINE_BARS)


def plan_schema(total_bars):
    batches = max(MIN_BATCHES, round(total_bars / BATCH_BARS))
    tracks = batches * TRACKS_PER_ACTIVATION
    modulus = DEFAULT_MODULUS
    if tracks > MAX_TRACKS:
        tracks = MAX_TRACKS
        needed = math.ceil(total_bars / (2 * ROUTINE_BARS)) - 1
        modulus = max(DEFAULT_MODULUS, min(needed, tracks))
    return tracks, modulus


def validate_schema_key(root, scale):
    if root not in NOTE_NAMES:
        raise ValueError(
            "The Korshmer Schema needs a defined root note. "
            "Choose a root in the THEORY tab (not Auto)."
        )
    if scale not in SCHEMA_SCALES:
        raise ValueError(
            "The Korshmer Schema works only in Pentatonic Major or "
            "Pentatonic Minor. Choose one of them in the THEORY tab."
        )
    return NOTE_NAMES.index(root), scale


def plan_beats(rng):
    return rng.sample(range(BEATS_PER_BAR), BEATS_PER_BAR)


def plan_pitches(rng, root_idx, scale, is_drum=False):
    intervals = SCALES[scale]
    degrees = rng.sample(range(len(intervals)), ROUTINE_BARS)
    low, high = GM_PERCUSSION_RANGE if is_drum else (0, 127)
    options = []
    for octave in SCHEMA_OCTAVES:
        base = 12 * (octave + 1) + root_idx
        pitches = [base + intervals[d] for d in degrees]
        if all(low <= p <= high for p in pitches):
            options.append(pitches)
    return rng.choice(options)


def plan_role_pitches(rng, root_idx, scale, role, batch):
    intervals = SCALES[scale]
    degrees = list(range(len(intervals)))
    fifth = next((d for d, i in enumerate(intervals) if i == 7), None)
    if role.avoid_root:
        chosen = degrees[1:]
        rng.shuffle(chosen)
    elif role.root_first and role.follows_chords:
        top = fifth if fifth is not None else 0
        chosen = [0, rng.choice((0, top)), 0, top]
    elif role.root_first:
        rest = degrees[1:]
        rng.shuffle(rest)
        chosen = [0, *rest[: ROUTINE_BARS - 1]]
        if fifth is not None and fifth not in chosen:
            chosen[-1] = fifth
        elif fifth is not None:
            chosen.remove(fifth)
            chosen.append(fifth)
    else:
        chosen = rng.sample(degrees, ROUTINE_BARS)
    octave = texture_roles.role_octave(role, batch)
    base = 12 * (octave + 1) + root_idx
    chosen = chosen[:ROUTINE_BARS]
    return [base + intervals[d] for d in chosen], chosen, base


def _followed_pitch(follow, intervals, chord_root, position):
    base, degrees = follow
    shift = texture_roles.chord_shift(chord_root, intervals)
    return base + intervals[(degrees[position] + shift) % len(intervals)]


def _fit_voices(long_picks, short_picks):
    fitted_long = []
    fitted_short = []
    for batch in range(len(long_picks) // LONG_TRACKS):
        fitted_long += texture_roles.match_voices(
            long_picks[batch * LONG_TRACKS : (batch + 1) * LONG_TRACKS],
            texture_roles.LONG_ROLES,
            batch,
        )
        fitted_short += texture_roles.match_voices(
            short_picks[batch * SHORT_TRACKS : (batch + 1) * SHORT_TRACKS],
            texture_roles.SHORT_ROLES,
            batch,
        )
    return fitted_long, fitted_short


def _alternate_answers(slots, specs):
    calls = {
        (spec[6], spec[5].name): index
        for index, spec in enumerate(specs)
        if spec[5] is not None
    }
    for index, spec in enumerate(specs):
        role = spec[5]
        if role is None or not role.answers:
            continue
        call = calls.get((spec[6], role.answers))
        if call is None:
            continue
        apart = slots[index] - slots[call]
        if apart:
            slots[index] = apart


def plan_slots(rng, total_slots, index):
    plays = min(total_slots, max(1, round(total_slots / (int(index) + 1))))
    chosen = []
    for k in range(plays):
        low = k * total_slots // plays
        high = max(low + 1, (k + 1) * total_slots // plays)
        chosen.append(rng.randrange(low, high))
    return chosen


def fill_quiet_slots(
    rng, slots, total_slots, minimum=MIN_ACTIVE_TRACKS, wanted=None
):
    count = len(slots)
    for slot in range(total_slots):
        active = [t for t in range(count) if slot in slots[t]]
        idle = [t for t in range(count) if slot not in slots[t]]
        rng.shuffle(idle)
        if wanted is not None:
            idle.sort(key=lambda t: not wanted(t, slot))
        for t in idle[: max(0, minimum - len(active))]:
            slots[t].add(slot)


def _kinds(track_count):
    batch = (
        ["drums"] * DRUM_TRACKS
        + ["long"] * LONG_TRACKS
        + ["short"] * SHORT_TRACKS
    )
    return batch * (track_count // TRACKS_PER_ACTIVATION)


def _pick_distinct(rng, table, count):
    return rng.sample(list(table.items()), count)


def compose_routine_tracks(
    rng,
    root_idx,
    scale,
    bpm,
    total_bars,
    track_count,
    modulus=DEFAULT_MODULUS,
    existing_tracks=0,
    start_bar=0,
    velocity_scale=1.0,
    long_pool=None,
    short_pool=None,
    kit=0,
    bar_energy=None,
    bar_roots=None,
):
    kinds = _kinds(track_count)
    indexes = [
        track_index(existing_tracks + k, modulus) for k in range(len(kinds))
    ]
    beat = 60.0 / bpm
    bar = beat * BEATS_PER_BAR

    batches = track_count // TRACKS_PER_ACTIVATION
    long_picks = []
    short_picks = []
    for _ in range(batches):
        long_picks += _pick_distinct(
            rng, long_pool or LONG_PROGRAMS, LONG_TRACKS
        )
        short_picks += _pick_distinct(
            rng, short_pool or SHORT_PROGRAMS, SHORT_TRACKS
        )
    arranged = bar_energy is not None
    intervals_of_scale = SCALES[scale]
    if arranged:
        long_picks, short_picks = _fit_voices(long_picks, short_picks)
    longs = iter(long_picks)
    shorts = iter(short_picks)

    specs = []
    for position, (kind, index) in enumerate(zip(kinds, indexes, strict=True)):
        role = texture_roles.role_for(position) if arranged else None
        batch = texture_roles.batch_of(position)
        follow = None
        beats = plan_beats(rng)
        if role is not None and role.beats:
            beats = list(role.beats)
        suffix = f" ({role.name})" if role is not None else ""
        if kind == "drums":
            inst = pretty_midi.Instrument(
                program=kit,
                is_drum=True,
                name=f"Schema {index} Drums{suffix}",
            )
            if role is not None:
                pitches = list(role.drum_pitches)
            else:
                pitches = plan_pitches(rng, root_idx, scale, is_drum=True)
        else:
            label, program = next(longs if kind == "long" else shorts)
            inst = pretty_midi.Instrument(
                program=program, name=f"Schema {index} {label}{suffix}"
            )
            if role is not None:
                pitches, degrees, base = plan_role_pitches(
                    rng, root_idx, scale, role, batch
                )
                if role.follows_chords and bar_roots is not None:
                    follow = (base, degrees)
            else:
                pitches = plan_pitches(rng, root_idx, scale)
        specs.append((inst, index, beats, pitches, kind, role, batch, follow))

    total_slots = total_bars // ROUTINE_BARS
    slots = [set(plan_slots(rng, total_slots, spec[1])) for spec in specs]
    energies = None
    wanted = None
    if arranged:
        energies = texture_roles.slot_energies(
            list(bar_energy), total_slots, ROUTINE_BARS
        )
        slots = [
            texture_roles.gate_slots(rng, slot_set, spec[5], spec[6], energies)
            for slot_set, spec in zip(slots, specs, strict=True)
        ]
        _alternate_answers(slots, specs)

        def wanted(track, slot):
            role, batch = specs[track][5], specs[track][6]
            low = texture_roles.entry_energy(role, batch)
            return low <= energies[slot] <= role.max_energy

    fill_quiet_slots(rng, slots, total_slots, wanted=wanted)

    lengths = {
        "drums": DRUM_LENGTHS,
        "long": LONG_LENGTHS,
        "short": SHORT_LENGTHS,
    }
    for (
        inst,
        _index,
        beats,
        pitches,
        kind,
        role,
        _batch,
        follow,
    ), track_slots in zip(specs, slots, strict=True):
        choices = role.lengths if role is not None else lengths[kind]
        for slot in sorted(track_slots):
            slot_bar = start_bar + slot * ROUTINE_BARS
            routine = []
            for offset in range(ROUTINE_BARS):
                start = (slot_bar + offset) * bar + beats[offset] * beat
                routine.append((start, rng.choice(choices)))
            slot_end = (slot_bar + ROUTINE_BARS) * bar
            level = velocity_scale
            if role is not None and energies is not None:
                level *= texture_roles.velocity_for(role, energies[slot])
            for i, (start, quarters) in enumerate(routine):
                limit = routine[i + 1][0] if i + 1 < len(routine) else slot_end
                velocity = int(rng.randint(80, 100) * level)
                pitch = pitches[i]
                if follow is not None:
                    pitch = _followed_pitch(
                        follow,
                        intervals_of_scale,
                        bar_roots[slot * ROUTINE_BARS + i],
                        i,
                    )
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=max(1, min(127, velocity)),
                        pitch=pitch,
                        start=start,
                        end=min(start + quarters * beat * NOTE_GAP, limit),
                    )
                )
    return [spec[0] for spec in specs]


def pentatonic_for(scale):
    if scale in SCHEMA_SCALES:
        return scale
    intervals = SCALES.get(scale, SCALES["Major"])
    return (
        "Pentatonic Minor"
        if 3 in intervals and 4 not in intervals
        else "Pentatonic Major"
    )


def build_schema_instruments(
    root,
    scale,
    bpm,
    existing_tracks=0,
    seed=None,
    minutes=DEFAULT_MINUTES,
):
    root_idx, scale = validate_schema_key(root, scale)
    bpm = validate_bpm(bpm)
    total_bars = bars_for_minutes(minutes, bpm)
    track_count, modulus = plan_schema(total_bars)
    return compose_routine_tracks(
        random.Random(seed),
        root_idx,
        scale,
        bpm,
        total_bars,
        track_count,
        modulus,
        existing_tracks,
    )
