import pretty_midi

from korshmer.utils.instrument_names import sanitize_instrument_name

DRUMS_NAME = "Drums"
FALLBACK_NAME = "Instrument"
SAME_START = 0.01
VARIANT_SUFFIXES = (" (low)", " (high)")

PROGRAM_NAMES = (
    "Grand Piano",
    "Bright Piano",
    "Studio Piano",
    "Saloon Piano",
    "Mellow Keys",
    "Glass Keys",
    "Harpsichord",
    "Funk Clav",
    "Celesta",
    "Sparkle Bells",
    "Music Box",
    "Vibes",
    "Marimba",
    "Xylophone",
    "Church Bells",
    "Dulcimer",
    "Jazz Organ",
    "Club Organ",
    "Rock Organ",
    "Pipe Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Bandoneon",
    "Nylon Guitar",
    "Steel Guitar",
    "Jazz Guitar",
    "Clean Guitar",
    "Muted Guitar",
    "Crunch Guitar",
    "Heavy Guitar",
    "Guitar Chimes",
    "Upright Bass",
    "Bass Guitar",
    "Pick Bass",
    "Fretless Bass",
    "Slap Bass",
    "Pop Bass",
    "Synth Bass",
    "Sub Bass",
    "Violin",
    "Viola",
    "Cello",
    "Double Bass",
    "Shimmer Strings",
    "Plucked Strings",
    "Harp",
    "Timpani",
    "Strings",
    "Slow Strings",
    "Synth Strings",
    "Airy Strings",
    "Choir",
    "Voice Ooh",
    "Synth Voice",
    "Big Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "Horn",
    "Brass",
    "Synth Brass",
    "Soft Brass",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Bottle Blow",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Square Lead",
    "Saw Lead",
    "Breath Lead",
    "Chiff Lead",
    "Gritty Lead",
    "Vocal Lead",
    "Power Lead",
    "Bass Lead",
    "Dream Pad",
    "Warm Pad",
    "Poly Pad",
    "Choir Pad",
    "Bowed Pad",
    "Metal Pad",
    "Halo Pad",
    "Sweep Pad",
    "Rain",
    "Cinematic",
    "Crystal",
    "Atmosphere",
    "Shine",
    "Eerie",
    "Echo Drops",
    "Sci-Fi",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bagpipes",
    "Fiddle",
    "Shanai",
    "Tiny Bells",
    "Agogo",
    "Steel Drums",
    "Wood Block",
    "Taiko",
    "Tuned Toms",
    "Synth Drum",
    "Reverse Crash",
    "Fret Noise",
    "Breath",
    "Ocean",
    "Birds",
    "Phone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
)


_FRIENDLY_NAMES = frozenset(name.lower() for name in PROGRAM_NAMES)
_GENERAL_MIDI_NAMES = (
    frozenset(name.lower() for name in pretty_midi.INSTRUMENT_MAP)
    - _FRIENDLY_NAMES
)


def instrument_name(program, is_drum=False):
    if is_drum:
        return DRUMS_NAME
    try:
        index = int(program)
    except (TypeError, ValueError):
        return FALLBACK_NAME
    if 0 <= index < len(PROGRAM_NAMES):
        return PROGRAM_NAMES[index]
    return FALLBACK_NAME


def is_general_midi_name(name):
    text = sanitize_instrument_name(name).lower()
    for suffix in VARIANT_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return any(
        part in _GENERAL_MIDI_NAMES
        for part in (text, text.rsplit(" - ", 1)[-1])
    )


def _instrument_key(inst):
    if inst.is_drum:
        return (True, 0)
    return (False, int(inst.program))


def _tidy_notes(notes):
    by_pitch = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    kept = []
    for same in by_pitch.values():
        same.sort(key=lambda n: n.start)
        current = same[0]
        for note in same[1:]:
            if note.start - current.start < SAME_START:
                current.end = max(current.end, note.end)
                current.velocity = max(current.velocity, note.velocity)
                continue
            if current.end > note.start:
                current.end = note.start
            kept.append(current)
            current = note
        kept.append(current)
    kept.sort(key=lambda n: (n.start, n.pitch))
    return kept


def _merged_controls(sources):
    owner = {}
    merged = []
    for index, source in enumerate(sources):
        for change in source.control_changes:
            if owner.setdefault(change.number, index) == index:
                merged.append(
                    pretty_midi.ControlChange(
                        change.number, change.value, change.time
                    )
                )
    merged.sort(key=lambda c: c.time)
    return merged


def _merged_bends(sources):
    for source in sources:
        if source.pitch_bends:
            return [
                pretty_midi.PitchBend(bend.pitch, bend.time)
                for bend in source.pitch_bends
            ]
    return []


def _track_name(sources, is_drum, program, keep_names, keep_shared_names):
    names = {sanitize_instrument_name(source.name) for source in sources}
    if len(names) == 1:
        (only,) = names
        if only in keep_names or (
            keep_shared_names and only and not is_general_midi_name(only)
        ):
            return only
    return instrument_name(program, is_drum)


def merge_by_instrument(instruments, keep_names=(), keep_shared_names=False):
    groups = {}
    for inst in instruments:
        if inst.notes:
            groups.setdefault(_instrument_key(inst), []).append(inst)
    tracks = []
    for (is_drum, program), sources in groups.items():
        if is_drum:
            program = max(sources, key=lambda s: len(s.notes)).program
        track = pretty_midi.Instrument(
            program=int(program),
            is_drum=is_drum,
            name=_track_name(
                sources, is_drum, program, keep_names, keep_shared_names
            ),
        )
        notes = [
            pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
            for source in sources
            for n in source.notes
        ]
        track.notes = _tidy_notes(notes) if len(sources) > 1 else notes
        track.control_changes = _merged_controls(sources)
        track.pitch_bends = _merged_bends(sources)
        tracks.append(track)
    return tracks
