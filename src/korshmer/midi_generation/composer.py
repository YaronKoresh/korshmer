import math
import random
from dataclasses import dataclass, field

import pretty_midi

from korshmer.constants import GRID_RESOLUTIONS, NOTE_NAMES, SCALES
from korshmer.midi_generation import (
    engine,
    genre_feel,
    genres,
    hooks,
    song_match,
)
from korshmer.midi_generation.korshmer_schema import (
    DEFAULT_MODULUS,
    MAX_TRACKS,
    TRACKS_PER_ACTIVATION,
    bars_for_minutes,
    compose_routine_tracks,
    pentatonic_for,
    plan_schema,
    validate_bars,
    validate_minutes,
    whole_routines,
)
from korshmer.utils.tempo import validate_bpm
from korshmer.utils.track_layout import merge_by_instrument

ROLE_KEYS = (
    "drums",
    "bass",
    "chords",
    "pad",
    "melody",
    "arp",
    "pluck",
    "counter",
)
ROLE_LABELS = {
    "drums": "Drums",
    "bass": "Bass",
    "chords": "Chords",
    "pad": "Pad",
    "melody": "Melody",
    "arp": "Arp",
    "pluck": "Pluck",
    "counter": "Counter-melody",
}
DEFAULT_ROLES = ("drums", "bass", "chords", "pad", "melody")

MODES = ("New song", "Continue editor", "Layer over editor")
MAX_TEXTURE_BATCHES = MAX_TRACKS // TRACKS_PER_ACTIVATION
MIN_ACTIVE_ROLES = 2
MOODS = ["Auto", *engine.MELODY_MOODS]
TEXTURE_VELOCITY = 0.75
ACT_BARS = 48
ON_BEAT_WINDOW = 0.06
ACT_DRUM_VARIATION = 0.35
ACT_CHORD_OCTAVES = (4, 3)
ACT_PAD_OCTAVES = (3, 4)
ACT_ARP_SHAPES = ("updown", "up", "down")
ARC_CLIMAX_AT = 0.7
ARC_SWAP_CHANCE = 0.7
ARC_MIN_SWAPS = 3
ARC_ENERGY = {
    "Statement": 1.0,
    "Development": 1.0,
    "Reflection": 0.6,
    "Climax": 1.15,
    "Return": 0.9,
}
ARC_REFLECTION_MUTED = frozenset({"drums", "bass", "pluck"})
ARC_CLIMAX_LIFT = 12
ARC_BREATH_ROLES = ("drums", "bass", "arp", "pluck")
ARC_CRASH_VELOCITY = 112
ARC_CRASH_LENGTH = 0.5
_ARC_MODES = ("Major", "Minor", "Dorian", "Mixolydian", "Lydian", "Phrygian")

STRUCTURES = {
    "Verse / Chorus": [
        "Intro",
        "Verse",
        "Chorus",
        "Verse",
        "Chorus",
        "Break",
        "Chorus",
        "Outro",
    ],
    "Build & Drop": ["Intro", "Build", "Drop", "Break", "Drop", "Outro"],
    "Journey (A-B-A)": ["Intro", "A", "B", "A", "Break", "B", "Outro"],
    "Loop": ["Loop"],
}

AUTO_STRUCTURE = "Auto"
AUTO_STYLE = "Auto"
STYLE_CHOICES = (AUTO_STYLE, *engine.SONG_STYLES)
AUTO_STYLE_TEMPOS = (
    (84.0, "Lo-Fi", "Jazz"),
    (104.0, "Hip-Hop", "Pop"),
    (124.0, "Pop", "Pop"),
    (138.0, "EDM", "EDM"),
)
AUTO_STYLE_FAST = ("Phonk", "Rock")
SYNTH_PROGRAMS = frozenset({38, 39, *range(80, 104)})
DRIVEN_PROGRAMS = frozenset({29, 30})
SYNTH_SONG_SHARE = 0.35
DRIVEN_SONG_SHARE = 0.2
ANSWER_MIN_BEATS = 2.0
ANSWER_JOIN_BEATS = 1.0
ANSWER_NAME = "Melody answer"
HARMONY_MIN_BEATS = 1.0
SUNG_PHRASE_SHARE = 0.4
RHYTHM_ROLES = ("drums", "bass")
LEAD_ROLES = ("melody", "counter")
RHYTHM_LEAD = 6
MAX_VELOCITY = 127
MAX_PITCH = 127
SONG_GAIN_RANGE = (0.8, 1.25)
AUDIBLE_VELOCITY = 24
SONG_GAIN_DEADBAND = 0.15
SONG_QUIET_GAIN = 1.0
LAYER_ENERGY_FLOOR = 0.35
PHRASE_BARS = 4
PHRASE_TURNS = (0, 1, 0, 2)
LAYER_RHYTHM_SECTIONS = ("Break",)
LAYER_BASS_RHYTHMS = ("syncopated", "sparse", "offbeat")
LAYER_CHORD_HITS = ([0.0, 0.75], [0.0, 0.25, 0.5, 0.75], [0.0, 0.625])
PHRASE_RESTS = {"arp": 1, "pluck": 2}
PICKUP_LENGTH = 0.125
PICKUP_GAP = 0.03125
PICKUP_TOP = 55
PICKUP_SOFTER = 8
PATTERN_ATTEMPTS = 8
LAYER_FILLS = ("snare", "toms", "hats", "roll")
DEFAULT_STRUCTURE = "Verse / Chorus"
STYLE_STRUCTURES = {
    "EDM": "Build & Drop",
    "Lo-Fi": "Journey (A-B-A)",
    "Jazz": "Journey (A-B-A)",
    "Phonk": "Build & Drop",
}

_ALL = set(ROLE_KEYS)
SECTION_PROFILES = {
    "Intro": (0.3, (4, 8), {"pad", "chords", "arp", "melody"}, "intro"),
    "Verse": (
        0.55,
        (8, 16),
        {"drums", "bass", "chords", "pad", "melody"},
        "verse",
    ),
    "Chorus": (1.0, (8, 16), _ALL, "chorus"),
    "Break": (0.35, (4, 8), {"pad", "chords", "melody", "arp"}, "break"),
    "Build": (0.7, (4, 8), {"drums", "bass", "chords", "arp", "pad"}, "bridge"),
    "Drop": (1.0, (8, 16), _ALL, "chorus"),
    "A": (0.6, (8, 16), {"drums", "bass", "chords", "pad", "melody"}, "verse"),
    "B": (0.9, (8, 16), _ALL, "chorus"),
    "Outro": (0.3, (4, 8), {"pad", "chords", "melody"}, "intro"),
    "Loop": (0.8, (8,), _ALL, "verse"),
}
_BODY_NAMES = ("Verse", "Chorus", "A", "B", "Build", "Drop")

_TRANSITIONS = {
    0: (3, 4, 5, 1, 2),
    1: (4, 6, 3),
    2: (5, 3, 1),
    3: (4, 0, 1, 5),
    4: (0, 5, 3),
    5: (3, 1, 4, 2),
    6: (0, 2),
}
_PROGRESSION_STARTS = {
    "verse": (0, 5, 0),
    "chorus": (3, 0, 5),
    "break": (5, 3),
    "bridge": (5, 1, 3),
    "intro": (0, 5),
}
_FIXED_GROOVES = ("EDM", "Jazz")
_BASS_RHYTHMS = ("eighths", "offbeat", "sparse", "syncopated", "walking")
_CHORD_HITS = {
    "verse": ([0.0], [0.0, 0.5], [0.0, 0.75]),
    "chorus": ([0.0, 0.25, 0.5, 0.75], [0.0, 0.5], [0.0, 0.375, 0.5]),
    "break": ([0.0],),
    "bridge": ([0.0, 0.5], [0.25, 0.75]),
    "intro": ([0.0],),
}
_PRIORITY = (
    "drums",
    "bass",
    "chords",
    "pad",
    "melody",
    "arp",
    "pluck",
    "counter",
)

_HARMONY_SCALES = {
    "Pentatonic Major": "Major",
    "Pentatonic Minor": "Minor",
    "Blues": "Minor",
    "Chromatic": "Major",
}


@dataclass
class ComposeOptions:
    root: str
    scale: str
    bpm: float = 120.0
    minutes: float = 3.0
    bars: int | None = None
    style: str = AUTO_STYLE
    structure: str = AUTO_STRUCTURE
    mood: str = "Auto"
    seed: int | None = None
    roles: tuple = DEFAULT_ROLES
    texture_batches: int = 0
    mode: str = "New song"
    existing_tracks: int = 0
    match: str = song_match.DEFAULT_MATCH
    grid_offset: float | None = None
    skip_empty_edges: bool = True


@dataclass
class Section:
    name: str
    start_bar: int
    bars: int
    energy: float
    roles: tuple = ()
    progression: tuple = ()
    act: int = 0
    act_kind: str = "Statement"
    key: str = ""


@dataclass
class Act:
    index: int
    kind: str
    theme: int = 0
    shift: int = 0
    scale: str = ""
    programs: dict = field(default_factory=dict)
    kit: int = 0


@dataclass
class Composition:
    instruments: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    total_bars: int = 0
    start_bar: int = 0
    bar_chords: list = field(default_factory=list)
    grid_offset: float = 0.0
    hook: object = None
    tracks: list = field(default_factory=list)
    style: str = ""


def validate_options(options):
    if options.root not in NOTE_NAMES:
        raise ValueError(
            "The generator needs a defined root note. "
            "Choose a root in the THEORY tab (not Auto)."
        )
    if options.scale not in SCALES:
        raise ValueError(
            "The generator needs a defined scale. "
            "Choose a scale in the THEORY tab (not Auto)."
        )
    if options.bars is None:
        validate_minutes(options.minutes)
    else:
        validate_bars(options.bars)
    validate_bpm(options.bpm)
    if options.style not in STYLE_CHOICES:
        raise ValueError(f"Unknown style: {options.style}")
    if options.structure not in (AUTO_STRUCTURE, *STRUCTURES):
        raise ValueError(f"Unknown structure: {options.structure}")
    if options.mood not in MOODS:
        raise ValueError(f"Unknown mood: {options.mood}")
    if options.mode not in MODES:
        raise ValueError(f"Unknown mode: {options.mode}")
    song_match.validate_match(options.match)
    song_match.validate_offset(options.grid_offset)
    unknown = [r for r in options.roles if r not in ROLE_KEYS]
    if unknown:
        raise ValueError(f"Unknown role: {unknown[0]}")
    batches = options.texture_batches
    if (
        not isinstance(batches, int)
        or isinstance(batches, bool)
        or not 0 <= batches <= MAX_TEXTURE_BATCHES
    ):
        raise ValueError(
            f"The texture size must be between 0 and {MAX_TEXTURE_BATCHES}."
        )
    if not options.roles and batches == 0:
        raise ValueError("Choose at least one role.")
    return NOTE_NAMES.index(options.root)


def _program_shares(existing):
    counts = {}
    for inst in existing.instruments:
        if not inst.is_drum:
            counts[inst.program] = counts.get(inst.program, 0) + len(inst.notes)
    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()} if total else {}


def song_style(existing, bpm, scale):
    shares = _program_shares(existing)
    synth = sum(v for p, v in shares.items() if p in SYNTH_PROGRAMS)
    if not shares or synth >= SYNTH_SONG_SHARE:
        return resolve_style(AUTO_STYLE, bpm, scale)
    driven = sum(v for p, v in shares.items() if p in DRIVEN_PROGRAMS)
    if driven >= DRIVEN_SONG_SHARE:
        return "Rock"
    if bpm < AUTO_STYLE_TEMPOS[0][0]:
        return "Lo-Fi" if _is_minor_like(scale) else "Jazz"
    return "Pop"


def match_palette(rng, programs, existing):
    families = {p // 8 for p in _program_shares(existing)}
    matched = {}
    for role, options in programs.items():
        close = [p for p in options if p // 8 in families]
        matched[role] = rng.choice(close or list(options))
    return matched


def resolve_style(style, bpm, scale, existing=None):
    if style != AUTO_STYLE:
        return style
    if existing is not None:
        return song_style(existing, bpm, scale)
    minor = _is_minor_like(scale)
    for limit, minor_style, major_style in AUTO_STYLE_TEMPOS:
        if bpm < limit:
            return minor_style if minor else major_style
    return AUTO_STYLE_FAST[0] if minor else AUTO_STYLE_FAST[1]


def _mean_velocity(notes):
    notes = list(notes)
    if not notes:
        return None
    return sum(n.velocity for n in notes) / len(notes)


def _scale_velocity(note, gain):
    note.velocity = int(max(1, min(MAX_VELOCITY, round(note.velocity * gain))))


def keep_audible(voices):
    for _role, inst in voices:
        for note in inst.notes:
            note.velocity = max(AUDIBLE_VELOCITY, note.velocity)


def _steady_gain(ratio):
    low, high = SONG_GAIN_RANGE
    if abs(ratio - 1.0) <= SONG_GAIN_DEADBAND:
        return 1.0
    edge = 1.0 + math.copysign(SONG_GAIN_DEADBAND, ratio - 1.0)
    return min(high, max(low, ratio / edge))


def song_bar_gains(existing, bar, total_bars, sections=()):
    sums = [0.0] * total_bars
    counts = [0] * total_bars
    for inst in existing.instruments:
        for note in inst.notes:
            index = int(note.start // bar)
            if 0 <= index < total_bars:
                sums[index] += note.velocity
                counts[index] += 1
    heard = sorted(s / c for s, c in zip(sums, counts, strict=True) if c)
    if not heard:
        return [1.0] * total_bars
    middle = heard[len(heard) // 2]
    spans = [(s.start_bar, s.start_bar + s.bars) for s in sections] or [
        (index, index + 1) for index in range(total_bars)
    ]
    gains = [SONG_QUIET_GAIN] * total_bars
    for first, last in spans:
        last = min(last, total_bars)
        count = sum(counts[first:last])
        if count:
            level = sum(sums[first:last]) / count
            gains[first:last] = [_steady_gain(level / middle)] * (last - first)
    return gains


def follow_song_dynamics(voices, gains, bar):
    for _role, inst in voices:
        for note in inst.notes:
            index = int(note.start // bar)
            if 0 <= index < len(gains):
                _scale_velocity(note, gains[index])


def match_song_level(voices, existing):
    level = _mean_velocity(
        n
        for inst in existing.instruments
        if not inst.is_drum
        for n in inst.notes
    )
    if level is None:
        return
    for role, inst in voices:
        own = _mean_velocity(inst.notes)
        if own is None:
            continue
        target = level + (RHYTHM_LEAD if role in RHYTHM_ROLES else 0)
        for note in inst.notes:
            _scale_velocity(note, target / own)


def _fresh(rng, options, used):
    unused = [o for o in options if o not in used]
    choice = rng.choice(unused or list(options))
    used.append(choice)
    return choice


def lift_rhythm_section(voices, lead_notes=()):
    leads = [
        n for role, inst in voices if role in LEAD_ROLES for n in inst.notes
    ]
    lead = _mean_velocity(leads or lead_notes)
    if lead is None:
        return
    target = min(MAX_VELOCITY, lead + RHYTHM_LEAD)
    for role, inst in voices:
        level = _mean_velocity(inst.notes) if role in RHYTHM_ROLES else None
        if level is None or level >= target:
            continue
        for note in inst.notes:
            _scale_velocity(note, target / level)


def harmony_scale(scale):
    if scale in _HARMONY_SCALES:
        return _HARMONY_SCALES[scale]
    if len(SCALES[scale]) == 7:
        return scale
    return "Major"


def _is_minor_like(scale):
    intervals = SCALES[scale]
    return 3 in intervals and 4 not in intervals


def _split_loop(total_bars):
    energy = SECTION_PROFILES["Loop"][0]
    sections = []
    cursor = 0
    while cursor < total_bars:
        bars = min(ACT_BARS, total_bars - cursor)
        sections.append(Section("Loop", cursor, bars, energy))
        cursor += bars
    return sections


def structure_for(style):
    return STYLE_STRUCTURES.get(style, DEFAULT_STRUCTURE)


def plan_sections(rng, total_bars, structure):
    names = list(STRUCTURES[structure])
    if len(names) == 1 or total_bars < 16:
        names = ["Loop"]
    if names == ["Loop"] and total_bars > ACT_BARS:
        return _split_loop(total_bars)
    lens = [rng.choice(SECTION_PROFILES[n][1]) for n in names]
    if len(names) > 1:
        middle = [n for n in names[1:-1] if n in _BODY_NAMES] or ["Verse"]
        has_break = "Break" in names
        cycle = 0
        since_break = 0
        while sum(lens) < total_bars:
            if has_break and since_break >= ACT_BARS * 0.75:
                name = "Break"
                since_break = 0
            else:
                name = middle[cycle % len(middle)]
                cycle += 1
            names.insert(len(names) - 1, name)
            bars = rng.choice(SECTION_PROFILES[name][1])
            lens.insert(len(lens) - 1, bars)
            if name != "Break":
                since_break += bars
    while sum(lens) > total_bars:
        candidates = [i for i, n in enumerate(lens) if n > 4]
        if candidates:
            longest = max(candidates, key=lambda i: (lens[i], -abs(i - 1)))
            lens[longest] -= 4
            continue
        if len(names) <= 3:
            break
        drop = max(1, len(names) // 2)
        del names[drop], lens[drop]
    grow = 0
    while sum(lens) < total_bars:
        lens[grow % len(lens)] += 4
        grow += 1
    if sum(lens) != total_bars:
        names, lens = ["Loop"], [total_bars]
    sections = []
    cursor = 0
    for name, bars in zip(names, lens, strict=True):
        energy = SECTION_PROFILES[name][0]
        sections.append(Section(name, cursor, bars, energy))
        cursor += bars
    return sections


def act_kinds(count):
    if count <= 1:
        return ["Statement"]
    if count == 2:
        return ["Statement", "Climax"]
    climax = min(count - 2, max(1, round(ARC_CLIMAX_AT * (count - 1))))
    kinds = ["Development"] * count
    kinds[0] = "Statement"
    kinds[-1] = "Return"
    kinds[climax] = "Climax"
    if climax - 1 >= 2:
        kinds[climax - 1] = "Reflection"
    return kinds


def mode_options(scale):
    intervals = SCALES[scale]
    count = len(intervals)
    found = []
    for degree in range(1, count):
        rotated = tuple(
            (intervals[(degree + k) % count] - intervals[degree]) % 12
            for k in range(count)
        )
        for name in _ARC_MODES:
            if tuple(SCALES[name]) == rotated:
                found.append((intervals[degree], name))
                break
    return found


def vary_programs(rng, profile, previous):
    alternatives = {
        role: [p for p in profile.programs[role] if p != previous[role]]
        for role in genres.PITCHED_ROLES
    }
    swappable = [role for role, found in alternatives.items() if found]
    chosen = {r for r in swappable if rng.random() < ARC_SWAP_CHANCE}
    spare = [r for r in swappable if r not in chosen]
    rng.shuffle(spare)
    while len(chosen) < min(ARC_MIN_SWAPS, len(swappable)):
        chosen.add(spare.pop())
    return {
        role: rng.choice(alternatives[role]) if role in chosen else program
        for role, program in previous.items()
    }


def home_mode(scale, modes):
    if not modes or scale in modes or _is_minor_like(scale):
        return 0, scale
    found = {name: shift for shift, name in mode_options(scale)}
    for name in modes:
        if name in found:
            return found[name], name
    return 0, scale


def plan_acts(rng, count, scale, base_programs, profile, modes=()):
    options = mode_options(scale)
    if modes:
        options = [o for o in options if o[1] in modes]
    acts = []
    for index, kind in enumerate(act_kinds(count)):
        if kind in ("Statement", "Return"):
            acts.append(
                Act(index, kind, 0, 0, scale, dict(base_programs), profile.kit)
            )
            continue
        before = acts[-1]
        fresh = [o for o in options if o[0] != before.shift]
        shift, name = rng.choice(fresh) if fresh else (0, scale)
        programs = vary_programs(rng, profile, before.programs)
        kits = [k for k in (profile.kit, *profile.kits) if k != before.kit]
        kit = rng.choice(kits) if kits else before.kit
        acts.append(Act(index, kind, index, shift, name, programs, kit))
    return acts


def vary_progression(rng, chords):
    chords = list(chords)
    shift = rng.randrange(len(chords))
    chords = chords[shift:] + chords[:shift]
    if rng.random() < 0.6:
        i = rng.randrange(1, len(chords))
        options = [d for d in _TRANSITIONS[chords[i - 1]] if d != chords[i]]
        chords[i] = rng.choice(options)
    return chords


def make_progression(rng, kind):
    chords = [rng.choice(_PROGRESSION_STARTS.get(kind, (0,)))]
    while len(chords) < 4:
        options = [d for d in _TRANSITIONS[chords[-1]] if d != chords[-1]]
        chords.append(rng.choice(options))
    if kind in ("verse", "bridge", "break") and rng.random() < 0.6:
        chords[-1] = 4
    elif kind == "chorus" and rng.random() < 0.5:
        chords[-1] = rng.choice((0, 4))
    return chords


def make_drum_pattern(rng, style):
    pattern = genre_feel.groove_for(rng, style)
    if style in _FIXED_GROOVES:
        return pattern
    kick = pattern[engine._KICK]
    if rng.random() < 0.6:
        extra = rng.choice([3, 6, 7, 10, 11, 14])
        if extra not in kick:
            kick.append(extra)
    pattern[engine._KICK] = sorted(set(kick))
    return pattern


def _new_instrument(role, program=0):
    return pretty_midi.Instrument(
        program=program,
        is_drum=role == "drums",
        name=ROLE_LABELS[role],
    )


def _breathe(out, lo, hi):
    others = [
        n
        for role, voices in out.items()
        if role not in ARC_BREATH_ROLES
        for voice in voices.values()
        for n in voice.notes
        if n.start < hi and n.end > lo
    ]
    if not others:
        return
    for role in ARC_BREATH_ROLES:
        for voice in out.get(role, {}).values():
            kept = []
            for note in voice.notes:
                if lo <= note.start < hi:
                    continue
                if note.start < lo < note.end:
                    note.end = lo
                kept.append(note)
            voice.notes = kept


def _mark_acts(out, starts, bar, start_bar):
    for act, (first_bar, roles, chapter) in starts.items():
        if act == 0:
            continue
        at = (start_bar + first_bar) * bar
        if "drums" in roles:
            _voice(out, "drums", chapter.kit).notes.append(
                pretty_midi.Note(
                    ARC_CRASH_VELOCITY,
                    engine.CRASH,
                    at,
                    at + bar * ARC_CRASH_LENGTH,
                )
            )
        if chapter.kind == "Climax":
            _breathe(out, max(0.0, at - bar), at)


def _voice(out, role, program):
    voices = out.setdefault(role, {})
    if program not in voices:
        voice = _new_instrument(role, program)
        if voices:
            voice.name = f"{ROLE_LABELS[role]} {len(voices) + 1}"
        voices[program] = voice
    return voices[program]


def active_roles(profile_roles, user_roles, texture_on):
    chosen = [r for r in _PRIORITY if r in user_roles and r in profile_roles]
    needed = MIN_ACTIVE_ROLES - (1 if texture_on and chosen else 0)
    if texture_on and not chosen:
        needed = MIN_ACTIVE_ROLES - 1
    for role in _PRIORITY:
        if len(chosen) >= needed:
            break
        if role in user_roles and role not in chosen:
            chosen.append(role)
    return tuple(r for r in _PRIORITY if r in chosen)


def _markov_continuation(rng, notes, start, end, scale_pcs):
    source = sorted(notes, key=lambda n: n.start)
    transitions = {}
    for a, b, c in zip(source, source[1:], source[2:], strict=False):
        key = (b.pitch - a.pitch, round(a.end - a.start, 2))
        transitions.setdefault(key, []).append(
            (c.pitch - b.pitch, b.end - b.start, b.start - a.start)
        )
    if not transitions:
        return []
    last = source[-1]
    state = (
        last.pitch - source[-2].pitch if len(source) > 1 else 0,
        round(source[-2].end - source[-2].start, 2) if len(source) > 1 else 0.5,
    )
    pitch = last.pitch
    clock = start
    out = []
    while clock < end:
        options = transitions.get(state)
        if not options:
            state = rng.choice(list(transitions))
            options = transitions[state]
        interval, duration, gap = rng.choice(options)
        pitch = max(36, min(96, pitch + interval))
        for delta in (0, 1, -1, 2, -2):
            if (pitch + delta) % 12 in scale_pcs:
                pitch += delta
                break
        duration = max(0.1, min(duration, end - clock))
        out.append(
            pretty_midi.Note(
                rng.randint(80, 105), pitch, clock, clock + duration
            )
        )
        clock += max(duration, min(max(gap, 0.1), 2.0))
        state = (interval, round(duration, 2))
    return out


def _fit_to_chords(notes, bar, start_bar, bar_chords):
    beat = bar / 4
    for note in notes:
        index = int(note.start // bar) - start_bar
        if not 0 <= index < len(bar_chords) or not bar_chords[index]:
            continue
        offset = note.start % beat
        if min(offset, beat - offset) > ON_BEAT_WINDOW:
            continue
        chord = bar_chords[index]
        for delta in (0, 1, -1, 2, -2, 3, -3):
            if (note.pitch + delta) % 12 in chord:
                note.pitch += delta
                break


HARMONY_STEPS = (3, 4, 8, 9, 5)
HARMONY_VELOCITY_DROP = 20
HARMONY_LOW_PITCH = 48


def _harmonize(rng, notes, bar, bar_chords, scale_pcs):
    out = []
    for note in notes:
        index = int(note.start // bar)
        chord = (
            bar_chords[index]
            if 0 <= index < len(bar_chords) and bar_chords[index]
            else set()
        )
        down = note.pitch - HARMONY_STEPS[0] >= HARMONY_LOW_PITCH
        best, best_score = None, None
        for rank, step in enumerate(HARMONY_STEPS):
            pitch = note.pitch - step if down else note.pitch + step
            if pitch % 12 not in scale_pcs or not 0 <= pitch <= 127:
                continue
            score = (pitch % 12 in chord, -rank, rng.random())
            if best_score is None or score > best_score:
                best, best_score = pitch, score
        if best is None:
            continue
        out.append(
            pretty_midi.Note(
                max(1, note.velocity - HARMONY_VELOCITY_DROP),
                best,
                note.start,
                note.end,
            )
        )
    return out


def _lead_notes(existing):
    if existing is None:
        return []
    leads = [i for i in existing.instruments if not i.is_drum and i.notes]
    if not leads:
        return []
    return max(leads, key=lambda i: len(i.notes)).notes


def _section_drums(existing, layer_prog, bar, first_bar, bars):
    if layer_prog is None or existing is None:
        return None
    lo = first_bar * bar
    hi = lo + bars * bar
    hits = [
        n
        for inst in existing.instruments
        if inst.is_drum
        for n in inst.notes
        if lo <= n.start < hi
    ]
    if not hits:
        return None
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes = hits
    return drums


def _phrases(bars, split):
    if not split:
        return [(0, bars)]
    return [
        (offset, min(PHRASE_BARS, bars - offset))
        for offset in range(0, bars, PHRASE_BARS)
    ]


def _revoice(notes, turn):
    chords = {}
    for note in notes:
        chords.setdefault(round(note.start, 4), []).append(note)
    for chord in chords.values():
        chord.sort(key=lambda n: n.pitch)
        if len(chord) < 2:
            continue
        if turn == 1 and chord[0].pitch + 12 <= MAX_PITCH:
            chord[0].pitch += 12
        elif turn == 2 and chord[-1].pitch - 12 >= 0:
            chord[-1].pitch -= 12


def _bass_pickup(notes, end, bar):
    if not notes:
        return
    start = end - bar * PICKUP_LENGTH
    last = max(notes, key=lambda n: n.start)
    for note in notes:
        if note.start < start < note.end:
            note.end = start
    if last.start >= start:
        return
    pitch = last.pitch + 12 if last.pitch + 12 <= PICKUP_TOP else last.pitch
    notes.append(
        pretty_midi.Note(
            max(AUDIBLE_VELOCITY, last.velocity - PICKUP_SOFTER),
            pitch,
            start,
            end - bar * PICKUP_GAP,
        )
    )


def _restrike(notes, start, end):
    middle = 0.5 * (start + end)
    again = []
    for note in notes:
        if note.start < middle < note.end and note.start >= start:
            again.append(
                pretty_midi.Note(note.velocity, note.pitch, middle, note.end)
            )
            note.end = middle
    notes.extend(again)


def vary_phrase(inst, role, turn, end, bar):
    if not turn:
        return
    if role in ("chords", "pad"):
        _revoice(inst.notes, turn)
    if role == "pad" and turn == 2:
        _restrike(inst.notes, end - bar, end)
    elif role == "bass":
        _bass_pickup(inst.notes, end, bar)


def _widened(pool, extra):
    return tuple(pool) + tuple(x for x in extra if x not in pool)


def _turn(phrase_count, layering):
    return PHRASE_TURNS[phrase_count % len(PHRASE_TURNS)] if layering else 0


def _calm_rate(rates):
    return min(rates, key=lambda rate: GRID_RESOLUTIONS.get(rate, 4))


def lead_spans(lead, lo, hi, join):
    spans = []
    for note in sorted(lead, key=lambda n: n.start):
        if note.end <= lo or note.start >= hi:
            continue
        if spans and note.start - spans[-1][1] <= join:
            spans[-1][1] = max(spans[-1][1], note.end)
        else:
            spans.append([note.start, note.end])
    return [(max(lo, a), min(hi, b)) for a, b in spans]


def sung_share(lead, lo, hi, join):
    if hi <= lo:
        return 0.0
    return sum(b - a for a, b in lead_spans(lead, lo, hi, join)) / (hi - lo)


def answer_phrases(notes, lead, lo, hi, beat):
    edges = [lo]
    for begin, end in lead_spans(lead, lo, hi, ANSWER_JOIN_BEATS * beat):
        edges += [begin, end]
    edges.append(hi)
    gaps = [
        (a, b)
        for a, b in zip(edges[::2], edges[1::2], strict=True)
        if b - a >= ANSWER_MIN_BEATS * beat
    ]
    answers = []
    for note in notes:
        gap = next((g for g in gaps if g[0] <= note.start < g[1]), None)
        if gap is None:
            continue
        note.end = min(note.end, gap[1])
        if note.end > note.start:
            answers.append(note)
    return answers


def _layer_pattern(rng, style, made, key):
    if key not in made:
        seen = [sorted(p.items()) for p in made.values()]
        pattern = make_drum_pattern(rng, style)
        for _attempt in range(PATTERN_ATTEMPTS):
            if sorted(pattern.items()) not in seen:
                break
            pattern = make_drum_pattern(rng, style)
        made[key] = pattern
    return made[key]


def _heard_energy(energy, first, bars, name):
    heard = energy[first : first + bars]
    if not heard:
        return SECTION_PROFILES[name][0]
    level = sum(heard) / len(heard)
    return min(1.0, LAYER_ENERGY_FLOOR + (1 - LAYER_ENERGY_FLOOR) * level)


def _clip_notes(inst, start, end):
    inst.notes = [n for n in inst.notes if start <= n.start < end]
    for note in inst.notes:
        note.end = min(note.end, end)


def _grid_offset(options, existing, bpm):
    if options.grid_offset is None:
        return song_match.estimate_offset(existing, bpm)
    return song_match.fold_offset(options.grid_offset, bpm)


def compose(options, existing=None):
    root_idx = validate_options(options)
    bpm = validate_bpm(options.bpm)
    layering = options.mode == "Layer over editor"
    style = resolve_style(
        options.style, bpm, options.scale, existing if layering else None
    )
    song_gains = None
    rng = random.Random(options.seed)
    bar = 240.0 / bpm
    total_bars = (
        bars_for_minutes(options.minutes, bpm)
        if options.bars is None
        else whole_routines(options.bars)
    )
    user_roles = tuple(r for r in _PRIORITY if r in options.roles)
    texture_on = options.texture_batches > 0

    start_bar = 0
    layer_offset = 0.0
    if options.mode == "Continue editor" and existing is not None:
        end = existing.get_end_time()
        if end > 0:
            layer_offset = _grid_offset(options, existing, bpm)
            start_bar = math.ceil((end - layer_offset) / bar - 1e-9)

    harm_scale = harmony_scale(options.scale)
    layer_prog = None
    song_has_drums = False
    first_bar = 0
    last_bar = None
    if options.mode == "Layer over editor":
        if existing is None or existing.get_end_time() <= 0:
            raise ValueError(
                "Layer over editor needs notes in the editor to follow."
            )
        layer_offset = _grid_offset(options, existing, bpm)
        existing = song_match.shift_song(existing, layer_offset)
        analysis = song_match.analyze_song(
            existing, bpm, root_idx, SCALES[harm_scale]
        )
        total_bars = analysis.bars
        layer_prog = song_match.progression_for(analysis, options.match)
        song_has_drums = any(
            i.is_drum and i.notes for i in existing.instruments
        )
        if song_has_drums:
            user_roles = tuple(r for r in user_roles if r != "drums")
        if options.skip_empty_edges:
            first_bar, last_bar = song_match.occupied_bars(existing, bpm)
        sections = [
            Section(
                name,
                first,
                bars,
                _heard_energy(analysis.energy, first, bars, name),
            )
            for name, first, bars in analysis.sections
        ]
        song_gains = song_bar_gains(existing, bar, total_bars, sections)
    else:
        structure = options.structure
        if structure == AUTO_STRUCTURE:
            structure = structure_for(style)
        sections = plan_sections(rng, total_bars, structure)
    minor_like = _is_minor_like(options.scale)
    feel = genre_feel.feel_for(style)
    home_shift = 0
    if not layering:
        home_shift, harm_scale = home_mode(harm_scale, feel.home_modes)
    if home_shift:
        minor_like = _is_minor_like(harm_scale)
    home_root = (root_idx + home_shift) % 12
    profile = genres.profile_for(style)
    role_programs = {
        role: rng.choice(profile.programs[role])
        for role in genres.PITCHED_ROLES
    }
    if layering:
        role_programs = match_palette(rng, profile.programs, existing)
    role_programs["drums"] = profile.kit
    act_count = (total_bars - 1) // ACT_BARS + 1
    arc = plan_acts(
        random.Random(rng.randrange(1 << 30)),
        act_count,
        harm_scale,
        {role: role_programs[role] for role in genres.PITCHED_ROLES},
        profile,
        feel.home_modes,
    )
    if options.mode == "Layer over editor":
        for chapter in arc:
            chapter.kind = "Statement"
            chapter.shift = 0
            chapter.scale = harm_scale
    mood_pool = (
        ("Melancholic", "Dreamy") if minor_like else ("Uplifting", "Energetic")
    )
    base_mood = (
        rng.choice(mood_pool) if options.mood == "Auto" else options.mood
    )

    progressions = {}
    hook = hooks.make_hook(
        random.Random(rng.randrange(1 << 30)), style, base_mood
    )
    bass_rhythm = {}
    used_bass = []
    used_hits = []
    layer_patterns = {}
    phrase_count = 0
    sung_lead = layering and song_match.has_vocal_lead(existing)
    answers = _new_instrument("counter", role_programs["counter"])
    answers.name = ANSWER_NAME
    chord_hits = {}
    melody_seed = {}
    acts = {
        0: {
            "pattern": make_drum_pattern(rng, style),
            "seventh": rng.random() < feel.seventh,
            "fill": rng.choice((4, 4, 8)),
            "mood": base_mood,
        }
    }
    out = {}
    bar_roots = [0] * total_bars
    bar_chords = [None] * total_bars
    act_starts = {}
    seen = {}
    lead_existing = _lead_notes(existing) if start_bar else []
    layer_lead = (
        song_match.find_lead(existing)
        if options.mode == "Layer over editor"
        else []
    )
    scale_pcs = {(root_idx + i) % 12 for i in SCALES[options.scale]}

    for section in sections:
        _energy, _bars_choice, profile_roles, kind = SECTION_PROFILES[
            section.name
        ]
        act = section.start_bar // ACT_BARS
        chapter = arc[act]
        if chapter.kind == "Climax":
            profile_roles = _ALL
        elif chapter.kind == "Reflection":
            profile_roles = set(profile_roles) - ARC_REFLECTION_MUTED
        if layering and section.name in LAYER_RHYTHM_SECTIONS:
            profile_roles = set(profile_roles) | set(RHYTHM_ROLES)
        section.roles = active_roles(profile_roles, user_roles, texture_on)
        section.act = act
        act_starts.setdefault(act, (section.start_bar, section.roles, chapter))
        section.act_kind = chapter.kind
        section.energy = min(1.0, section.energy * ARC_ENERGY[chapter.kind])
        section_root = (home_root + chapter.shift) % 12
        section.key = f"{NOTE_NAMES[section_root]} {chapter.scale}"
        occurrence = seen.get(section.name, 0)
        seen[section.name] = occurrence + 1
        variant = occurrence if layering else 0
        act = chapter.theme
        if act not in acts:
            acts[act] = {
                "pattern": make_drum_pattern(rng, style),
                "seventh": rng.random() < feel.seventh,
                "fill": rng.choice((4, 4, 8)),
                "mood": (
                    rng.choice(mood_pool)
                    if options.mood == "Auto"
                    else options.mood
                ),
            }
        state = acts[act]
        if (kind, act) not in progressions:
            drawn = genre_feel.progression_for(
                rng, style, kind, minor=minor_like
            )
            progressions[(kind, act)] = (
                vary_progression(rng, drawn) if act else drawn
            )
        prog = list(progressions[(kind, act)])
        if occurrence and rng.random() < 0.4:
            prog[-1] = rng.choice(_TRANSITIONS[prog[-2]])
        elif occurrence and act and rng.random() < 0.5:
            prog = vary_progression(rng, prog)
        closing = section is sections[-1] and layer_prog is None
        if closing:
            last = (start_bar + section.start_bar + section.bars - 1) % len(
                prog
            )
            prog[last] = 0
            prog[last - 1] = 4
        if layer_prog is not None:
            prog = list(layer_prog)
            section.progression = tuple(
                layer_prog[section.start_bar : section.start_bar + section.bars]
            )
        else:
            section.progression = tuple(prog)
        ctx = engine.HarmonicContext(
            root=section_root,
            scale=chapter.scale,
            bpm=bpm,
            progression=prog,
        )
        for b in range(min(section.bars, total_bars - section.start_bar)):
            chord = ctx.chord_at(start_bar + section.start_bar + b)
            root_pitch = ctx.degree_pitch(chord, 4)
            bar_roots[section.start_bar + b] = (root_pitch - root_idx) % 12
            bar_chords[section.start_bar + b] = {
                p % 12 for p in ctx.triad(chord, 4)
            }
        riff = hook.notes if feel.riff and kind == "chorus" else None
        stab = feel.stab and kind == "chorus"
        hits_pool = (
            feel.chord_hits
            if kind not in ("break", "intro")
            else _CHORD_HITS[kind]
        )
        bass_pool = feel.bass
        if layering:
            hits_pool = _widened(hits_pool, LAYER_CHORD_HITS)
            bass_pool = _widened(feel.bass, LAYER_BASS_RHYTHMS)
        hits_variant = variant or occurrence % 2
        opening = _turn(phrase_count, layering)
        bass_rhythm.setdefault(
            (kind, act, variant, opening),
            _fresh(rng, bass_pool, used_bass)
            if layering
            else rng.choice(feel.bass),
        )
        chord_hits.setdefault(
            (kind, act, hits_variant, opening),
            _fresh(rng, hits_pool, used_hits)
            if layering
            else rng.choice(hits_pool),
        )
        melody_seed.setdefault((kind, act), rng.randrange(1 << 30))
        sect_seed = rng.randrange(1 << 30)
        for index, (offset, length) in enumerate(
            _phrases(section.bars, layering)
        ):
            turn = _turn(phrase_count, layering)
            phrase_bar = section.start_bar + offset
            bass_key = (kind, act, variant, turn)
            if bass_key not in bass_rhythm:
                bass_rhythm[bass_key] = _fresh(rng, bass_pool, used_bass)
            hits_key = (kind, act, hits_variant, turn)
            if hits_key not in chord_hits:
                chord_hits[hits_key] = _fresh(rng, hits_pool, used_hits)
            look = act + variant + turn
            kw = {
                "seed": sect_seed + index,
                "bars": length,
                "start_bar": start_bar + phrase_bar,
                "energy": section.energy,
            }
            rested = layering and kind != "chorus"
            phrase_lo = phrase_bar * bar
            sung = sung_lead and (
                sung_share(
                    layer_lead,
                    phrase_lo,
                    phrase_lo + length * bar,
                    ANSWER_JOIN_BEATS * bar / 4,
                )
                >= SUNG_PHRASE_SHARE
            )
            for role in section.roles:
                if rested and PHRASE_RESTS.get(role) == turn:
                    continue
                if sung and role == "pluck":
                    continue
                if role == "drums":
                    pattern = state["pattern"]
                    fill_bars = state["fill"] if variant % 2 == 0 else 4
                    fill = feel.fill
                    if layering:
                        pattern = _layer_pattern(
                            rng, style, layer_patterns, (act, kind, turn)
                        )
                        fill_bars = length
                        fill = LAYER_FILLS[phrase_count % len(LAYER_FILLS)]
                    inst = engine.generate_drums(
                        ctx,
                        engine._STYLES[style][1],
                        bars=length,
                        seed=sect_seed + index,
                        energy=section.energy,
                        start_bar=start_bar + phrase_bar,
                        pattern=pattern,
                        fill_bars=fill_bars,
                        velocity_scale=feel.drum_vel,
                        variation=(
                            ACT_DRUM_VARIATION
                            if act or variant or layering
                            else 0.0
                        ),
                        ghosts=feel.ghosts,
                        fill=fill,
                        crash=feel.crash and kind == "chorus" and not index,
                        accents=feel.accents,
                        rolls=feel.hat_rolls,
                        roll_chance=feel.roll_chance,
                    )
                elif role == "bass":
                    inst = engine.generate_bass(
                        ctx,
                        style,
                        _section_drums(
                            existing, layer_prog, bar, phrase_bar, length
                        ),
                        rhythm="riff" if riff else bass_rhythm[bass_key],
                        riff=riff,
                        kick_steps=(
                            state["pattern"].get(engine._KICK)
                            if feel.bass_on_kicks and not layering
                            else None
                        ),
                        **kw,
                    )
                elif role == "chords":
                    inst = engine.generate_chords(
                        ctx,
                        style,
                        hits=(
                            hooks.stab_hits(hook)
                            if stab
                            else chord_hits[hits_key]
                        ),
                        seventh=state["seventh"] and kind != "break",
                        power=feel.power_chords,
                        octave=ACT_CHORD_OCTAVES[
                            (act + variant) % len(ACT_CHORD_OCTAVES)
                        ],
                        **kw,
                    )
                elif role == "pad":
                    inst = engine.generate_pad(
                        ctx,
                        octave=ACT_PAD_OCTAVES[
                            (act + variant) % len(ACT_PAD_OCTAVES)
                        ],
                        **kw,
                    )
                elif role == "arp":
                    inst = engine.generate_arp(
                        ctx,
                        _calm_rate(feel.arp_rates)
                        if sung
                        else feel.arp_rates[
                            (act + variant + turn) % len(feel.arp_rates)
                        ],
                        shape=ACT_ARP_SHAPES[look % len(ACT_ARP_SHAPES)],
                        **kw,
                    )
                    inst.program = 46
                elif role == "pluck":
                    inst = engine.generate_arp(
                        ctx,
                        feel.arp_rates[
                            (act + variant + 1 + turn // 2)
                            % len(feel.arp_rates)
                        ],
                        shape=ACT_ARP_SHAPES[(look + 1) % len(ACT_ARP_SHAPES)],
                        **kw,
                    )
                    inst.program = 80
                elif role == "melody":
                    if lead_existing or index:
                        continue
                    if layer_lead:
                        lo = section.start_bar * bar
                        hi = lo + section.bars * bar
                        heard = [n for n in layer_lead if lo <= n.start < hi]
                        held = [
                            n
                            for n in heard
                            if not sung_lead
                            or n.end - n.start >= HARMONY_MIN_BEATS * bar / 4
                        ]
                        harmony = _harmonize(
                            rng, held, bar, bar_chords, scale_pcs
                        )
                        program = chapter.programs.get(
                            role, role_programs[role]
                        )
                        _voice(out, role, program).notes.extend(harmony)
                        if not sung_lead:
                            continue
                        tune = engine.generate_melody(
                            ctx,
                            mood=state["mood"],
                            hook=hook,
                            kind=kind,
                            seed=melody_seed[(kind, act)],
                            bars=section.bars,
                            start_bar=start_bar + section.start_bar,
                            energy=section.energy,
                        )
                        answers.notes.extend(
                            answer_phrases(tune.notes, heard, lo, hi, bar / 4)
                        )
                        continue
                    inst = engine.generate_melody(
                        ctx,
                        mood=state["mood"],
                        lift=ARC_CLIMAX_LIFT if chapter.kind == "Climax" else 0,
                        hook=hook,
                        kind="chorus" if section.name == "Loop" else kind,
                        final=closing,
                        style=style,
                        **{**kw, "seed": melody_seed[(kind, act)]},
                    )
                else:
                    continue
                if layering:
                    vary_phrase(
                        inst,
                        role,
                        turn,
                        (start_bar + phrase_bar + length) * bar,
                        bar,
                    )
                program = (
                    chapter.kit
                    if role == "drums"
                    else chapter.programs.get(role, role_programs[role])
                )
                _voice(out, role, program).notes.extend(inst.notes)
            phrase_count += 1
        if "counter" in section.roles and "melody" in out:
            seg = pretty_midi.Instrument(program=0)
            lo = (start_bar + section.start_bar) * bar
            hi = lo + section.bars * bar
            seg.notes = [
                n
                for voice in out["melody"].values()
                for n in voice.notes
                if lo <= n.start < hi
            ]
            counter = engine.generate_countermelody(ctx, seg, seed=sect_seed)
            _voice(out, "counter", chapter.programs["counter"]).notes.extend(
                counter.notes
            )

    if lead_existing and "melody" in user_roles:
        lo = start_bar * bar
        hi = lo + total_bars * bar
        notes = _markov_continuation(rng, lead_existing, lo, hi, scale_pcs)
        _fit_to_chords(notes, bar, start_bar, bar_chords)
        leads = {}
        for note in notes:
            section = next(
                (
                    s
                    for s in sections
                    if (start_bar + s.start_bar) * bar
                    <= note.start
                    < (start_bar + s.start_bar + s.bars) * bar
                ),
                None,
            )
            if section is None or "melody" not in section.roles:
                continue
            program = arc[section.act].programs["melody"]
            if program not in leads:
                lead = _new_instrument("melody", program)
                lead.name = "Melody (continued)"
                if leads:
                    lead.name += f" {len(leads) + 1}"
                leads[program] = lead
            leads[program].notes.append(note)
        if leads:
            out["melody"] = leads

    if layer_lead:
        for count, voice in enumerate(out.get("melody", {}).values(), 1):
            voice.name = "Melody harmony" + (f" {count}" if count > 1 else "")
    _mark_acts(out, act_starts, bar, start_bar)
    song_end = (start_bar + total_bars) * bar
    song_start = first_bar * bar
    if last_bar is not None:
        song_end = min(song_end, last_bar * bar)
    voices = [
        (role, inst)
        for role in _PRIORITY
        for inst in out.get(role, {}).values()
    ]
    if answers.notes:
        voices.append(("melody", answers))
    genre_feel.apply_swing(
        [
            inst
            for role, inst in voices
            if role != "melody" or not (lead_existing or layer_lead)
        ],
        bpm,
        feel.swing,
    )
    if options.mode == "New song":
        for role, inst in voices:
            if role == "chords":
                genre_feel.apply_strum(inst, feel.strum)
    for role, inst in voices:
        _clip_notes(inst, song_start, song_end)
        engine.trim_overlaps(inst)
        if role == "melody":
            engine.trim_line(inst)
    if song_gains is not None:
        match_song_level(voices, existing)
        follow_song_dynamics(voices, song_gains, bar)
        keep_audible(voices)
    lift_rhythm_section(voices, layer_lead)
    instruments = [inst for _role, inst in voices if inst.notes]
    if texture_on:
        bar_energy = [0.0] * total_bars
        for section in sections:
            for b in range(section.start_bar, section.start_bar + section.bars):
                if b < total_bars:
                    bar_energy[b] = section.energy
        pent = pentatonic_for(options.scale)
        tracks = options.texture_batches * TRACKS_PER_ACTIVATION
        modulus = DEFAULT_MODULUS
        if tracks >= MAX_TRACKS:
            _count, modulus = plan_schema(total_bars)
        texture = compose_routine_tracks(
            rng,
            root_idx,
            pent,
            bpm,
            total_bars,
            tracks,
            modulus,
            existing_tracks=options.existing_tracks + len(instruments),
            start_bar=start_bar,
            velocity_scale=TEXTURE_VELOCITY,
            long_pool=profile.texture_long,
            short_pool=profile.texture_short,
            kit=profile.kit,
            bar_energy=bar_energy,
            bar_roots=bar_roots,
        )
        texture = [i for i in texture if not (song_has_drums and i.is_drum)]
        if last_bar is not None:
            for inst in texture:
                _clip_notes(inst, song_start, song_end)
            texture = [i for i in texture if i.notes]
        instruments += texture
    if layer_offset:
        for inst in instruments:
            for note in inst.notes:
                note.start += layer_offset
                note.end += layer_offset
    return Composition(
        instruments,
        sections,
        total_bars,
        start_bar,
        bar_chords,
        layer_offset,
        hook,
        merge_by_instrument(instruments),
        style,
    )
