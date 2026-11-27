import random

import pretty_midi

from korshmer.midi_generation.genres import primary
from korshmer.utils.bend_range import bend_for, set_bend_range

SLIDE_STYLES = frozenset({"Phonk"})
SLIDE_RANGE = 12
PHRASE_BARS = 4
DROP_ENERGY = 0.9
CALM_ENERGY = 0.5
DROP_SLIDES = 3
PHRASE_SLIDES = 2
CALM_SLIDES = 1
SLIDE_SHARE = 3
GLIDE_REACH = 12
GLIDE_WEIGHT = 3
GLIDE_LENGTHS = (1 / 16, 1 / 16, 1 / 8)
SCOOP_LENGTH = 1 / 16
SCOOP_FROM = (-12, -12, 12, -7, -5)
FALL_LENGTH = 1 / 8
FALL_TO = (-12, -12, -7)
LONG_NOTE = 3 / 16
JOINED_GAP = 1 / 16
NOTE_SHARE = 0.5
RESET_AFTER = 1 / 4
EARLIEST = 0.05
STEP_SECONDS = 0.002
STEPS_PER_BEAT = 220
MOST_STEPS = 64
FEWEST_STEPS = 4


def _ramp(start, end, first, last, bar):
    step = max(STEP_SECONDS, bar / 4 / STEPS_PER_BEAT)
    count = max(FEWEST_STEPS, min(MOST_STEPS, int((end - start) / step)))
    return [
        (
            start + (end - start) * k / count,
            first + (last - first) * k / count,
        )
        for k in range(count + 1)
    ]


def _glide(rng, note, earlier, bar):
    length = min(rng.choice(GLIDE_LENGTHS) * bar, NOTE_SHARE * _length(note))
    return _ramp(
        note.start, note.start + length, earlier.pitch - note.pitch, 0, bar
    )


def _scoop(rng, note, bar):
    length = min(SCOOP_LENGTH * bar, NOTE_SHARE * _length(note))
    return _ramp(
        note.start, note.start + length, rng.choice(SCOOP_FROM), 0, bar
    )


def _fall(rng, note, later, bar):
    length = min(FALL_LENGTH * bar, NOTE_SHARE * _length(note))
    reset = note.end + RESET_AFTER * bar
    if later is not None:
        reset = min(reset, later.start)
    return [
        *_ramp(note.end - length, note.end, 0, rng.choice(FALL_TO), bar),
        (reset, 0),
    ]


def _length(note):
    return note.end - note.start


def _glides_in(note, earlier, bar):
    return (
        earlier is not None
        and earlier.pitch != note.pitch
        and abs(earlier.pitch - note.pitch) <= GLIDE_REACH
        and note.start - earlier.end <= JOINED_GAP * bar
    )


def _stands_alone(note, earlier, later):
    return (earlier is None or earlier.end <= note.start) and (
        later is None or later.start >= note.end
    )


def _phrase_slides(energy, notes):
    if energy >= DROP_ENERGY:
        wanted = DROP_SLIDES
    elif energy >= CALM_ENERGY:
        wanted = PHRASE_SLIDES
    else:
        wanted = CALM_SLIDES
    return min(wanted, max(1, notes // SLIDE_SHARE))


def _energy_at(sections, bar_index):
    for section in sections:
        if section.start_bar <= bar_index < section.start_bar + section.bars:
            return section.energy
    return CALM_ENERGY


def _slide(rng, note, earlier, later, bar):
    kinds = ["scoop"]
    if _glides_in(note, earlier, bar):
        kinds += ["glide"] * GLIDE_WEIGHT
    if _length(note) >= LONG_NOTE * bar:
        kinds += ["fall"] * 2
    kind = rng.choice(kinds)
    if kind == "glide":
        return _glide(rng, note, earlier, bar)
    if kind == "fall":
        return _fall(rng, note, later, bar)
    return _scoop(rng, note, bar)


def _slides_for(rng, notes, sections, bar, origin):
    phrases = {}
    for index, note in enumerate(notes):
        earlier = notes[index - 1] if index else None
        later = notes[index + 1] if index + 1 < len(notes) else None
        if note.start < EARLIEST or not _stands_alone(note, earlier, later):
            continue
        bar_index = int((note.start - origin) / bar + 1e-6)
        phrases.setdefault(bar_index // PHRASE_BARS, []).append(index)
    points = []
    taken = set()
    for phrase in sorted(phrases):
        members = phrases[phrase]
        order = sorted(
            members,
            key=lambda i: (
                rng.random()
                ** (
                    1.0
                    / (
                        GLIDE_WEIGHT
                        if i and _glides_in(notes[i], notes[i - 1], bar)
                        else 1
                    )
                )
            ),
            reverse=True,
        )
        left = _phrase_slides(
            _energy_at(sections, phrase * PHRASE_BARS), len(members)
        )
        for index in order:
            if not left:
                break
            if index - 1 in taken or index + 1 in taken:
                continue
            taken.add(index)
            left -= 1
    for index in sorted(taken):
        points += _slide(
            rng,
            notes[index],
            notes[index - 1] if index else None,
            notes[index + 1] if index + 1 < len(notes) else None,
            bar,
        )
    return points


def add_bass_slides(basses, style, seed, sections, bar, origin=0.0):
    if primary(style) not in SLIDE_STYLES:
        return
    rng = random.Random() if seed is None else random.Random(f"slides:{seed}")
    for inst in basses:
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        points = _slides_for(rng, notes, sections, bar, origin)
        if not points:
            continue
        set_bend_range(inst, SLIDE_RANGE)
        inst.pitch_bends = [
            pretty_midi.PitchBend(bend_for(semitones, SLIDE_RANGE), time)
            for time, semitones in sorted(points, key=lambda p: p[0])
        ]
