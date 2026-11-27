from dataclasses import dataclass, field
from functools import cache

DEFAULT_GENRE = "Pop"
BLEND_JOIN = " + "
FLAVOR_JOIN = "/"

PITCHED_ROLES = (
    "melody",
    "chords",
    "bass",
    "pad",
    "arp",
    "pluck",
    "counter",
)


KIT_STANDARD = 0
KIT_ROOM = 8
KIT_POWER = 16
KIT_ELECTRONIC = 24
KIT_808 = 25
KIT_DANCE = 26
KIT_JAZZ = 32
KIT_BRUSH = 40
KNOWN_KITS = (
    KIT_STANDARD,
    KIT_ROOM,
    KIT_POWER,
    KIT_ELECTRONIC,
    KIT_808,
    KIT_DANCE,
    KIT_JAZZ,
    KIT_BRUSH,
)


CHIPTUNE_PROGRAMS = (80, 81)


@dataclass(frozen=True)
class GenreProfile:
    name: str
    programs: dict
    kit: int
    texture_long: dict
    texture_short: dict
    extras: dict = field(default_factory=dict)
    kits: tuple = ()


def _p(name, programs, kit, texture_long, texture_short, kits=()):
    return GenreProfile(
        name, programs, kit, texture_long, texture_short, kits=kits
    )


GENRES = {
    "Pop": _p(
        "Pop",
        {
            "melody": (0, 4, 11, 27, 25, 5, 73, 22),
            "chords": (0, 4, 25, 27, 5, 24),
            "bass": (33, 34, 35, 38),
            "pad": (89, 88, 50, 49, 48, 52),
            "arp": (46, 12, 24, 8, 108),
            "pluck": (24, 45, 12, 46, 108),
            "counter": (40, 73, 41, 71, 68, 56),
        },
        KIT_STANDARD,
        {
            "Violin": 40,
            "Cello": 42,
            "String Ensemble": 48,
            "Slow Strings": 49,
            "Synth Strings": 50,
            "Choir Aahs": 52,
            "Warm Pad": 89,
        },
        {
            "Acoustic Piano": 0,
            "Electric Piano": 4,
            "Steel Guitar": 25,
            "Nylon Guitar": 24,
            "Marimba": 12,
            "Harp": 46,
            "Vibraphone": 11,
            "Pizzicato Strings": 45,
            "Kalimba": 108,
        },
        (KIT_ROOM, KIT_POWER),
    ),
    "Rock": _p(
        "Rock",
        {
            "melody": (29, 30, 27, 28, 18, 25),
            "chords": (29, 30, 27, 25, 18),
            "bass": (34, 33, 35),
            "pad": (18, 16, 17, 48, 49, 61),
            "arp": (27, 25, 28, 24),
            "pluck": (28, 29, 25, 27),
            "counter": (18, 16, 40, 61, 42),
        },
        KIT_POWER,
        {
            "Rock Organ": 18,
            "Tonewheel Organ": 16,
            "String Ensemble": 48,
            "Slow Strings": 49,
            "Violin": 40,
            "Cello": 42,
            "Brass Section": 61,
        },
        {
            "Steel Guitar": 25,
            "Clean Guitar": 27,
            "Muted Guitar": 28,
            "Overdrive Guitar": 29,
            "Acoustic Piano": 0,
            "Honky-Tonk Piano": 3,
            "Clavinet": 7,
        },
        (KIT_STANDARD, KIT_ROOM),
    ),
    "Hip-Hop": _p(
        "Hip-Hop",
        {
            "melody": (4, 5, 11, 8, 14, 9, 73, 75, 0),
            "chords": (4, 5, 2, 0, 11, 48),
            "bass": (38, 39),
            "pad": (89, 94, 91, 99, 92, 52),
            "arp": (8, 9, 10, 108, 11),
            "pluck": (9, 10, 112, 108, 45),
            "counter": (53, 54, 41, 73, 52),
        },
        KIT_808,
        {
            "Warm Pad": 89,
            "Halo Pad": 94,
            "Space Voice": 91,
            "Synth Strings": 50,
            "Choir Aahs": 52,
            "Voice Oohs": 53,
            "Bowed Glass": 92,
            "Atmosphere": 99,
        },
        {
            "Electric Piano": 4,
            "FM Electric Piano": 5,
            "Vibraphone": 11,
            "Music Box": 10,
            "Celeste": 8,
            "Kalimba": 108,
            "Glockenspiel": 9,
            "Clavinet": 7,
        },
        (KIT_ELECTRONIC, KIT_ROOM),
    ),
    "Lo-Fi": _p(
        "Lo-Fi",
        {
            "melody": (4, 5, 11, 24, 0, 8, 73, 10),
            "chords": (4, 2, 0, 24, 5, 11),
            "bass": (35, 33, 32),
            "pad": (88, 89, 49, 52, 92),
            "arp": (46, 10, 24, 8, 108),
            "pluck": (24, 10, 11, 46, 108),
            "counter": (11, 71, 53, 73, 65),
        },
        KIT_ROOM,
        {
            "Warm Pad": 89,
            "Fantasia": 88,
            "Slow Strings": 49,
            "Voice Oohs": 53,
            "Viola": 41,
            "Cello": 42,
        },
        {
            "Electric Piano": 4,
            "Nylon Guitar": 24,
            "Vibraphone": 11,
            "Music Box": 10,
            "Kalimba": 108,
            "Harp": 46,
            "Acoustic Piano": 0,
            "Celeste": 8,
        },
        (KIT_BRUSH, KIT_STANDARD),
    ),
    "Jazz": _p(
        "Jazz",
        {
            "melody": (65, 66, 56, 59, 11, 26, 71),
            "chords": (0, 26, 4, 11, 5),
            "bass": (32, 33, 35),
            "pad": (49, 89, 52, 48, 60),
            "arp": (11, 26, 0, 46),
            "pluck": (26, 28, 11, 45),
            "counter": (71, 73, 66, 59, 56, 65),
        },
        KIT_JAZZ,
        {
            "Slow Strings": 49,
            "Cello": 42,
            "Viola": 41,
            "Clarinet": 71,
            "Flute": 73,
            "Voice Oohs": 53,
            "French Horns": 60,
        },
        {
            "Acoustic Piano": 0,
            "Jazz Guitar": 26,
            "Vibraphone": 11,
            "Marimba": 12,
            "Muted Trumpet": 59,
            "Harp": 46,
            "Pizzicato Strings": 45,
            "Electric Piano": 4,
        },
        (KIT_BRUSH, KIT_ROOM),
    ),
    "EDM": _p(
        "EDM",
        {
            "melody": (81, 80, 87, 62, 90, 84),
            "chords": (62, 81, 90, 86, 50, 88),
            "bass": (38, 39),
            "pad": (89, 95, 91, 94, 50, 88),
            "arp": (98, 100, 9, 112, 84),
            "pluck": (80, 83, 98, 45),
            "counter": (81, 54, 84, 85),
        },
        KIT_DANCE,
        {
            "Warm Pad": 89,
            "Halo Pad": 94,
            "Sweep Pad": 95,
            "Space Voice": 91,
            "Synth Strings": 50,
            "Saw Lead": 81,
            "Polysynth": 90,
            "Choir Aahs": 52,
        },
        {
            "Square Lead": 80,
            "Chiffer Lead": 83,
            "Crystal": 98,
            "Brightness": 100,
            "Glockenspiel": 9,
            "Marimba": 12,
            "Electric Piano": 4,
            "Synth Brass": 62,
        },
        (KIT_ELECTRONIC, KIT_808),
    ),
    "Phonk": _p(
        "Phonk",
        {
            "melody": (113,),
            "chords": (19, 52, 48, 0, 49, 20),
            "bass": (87,),
            "pad": (91, 53, 94, 92, 95, 52),
            "arp": (115, 113, 117, 47),
            "pluck": (115, 113, 117, 45),
            "counter": (53, 54, 52, 85, 91),
        },
        KIT_808,
        {
            "Space Voice": 91,
            "Halo Pad": 94,
            "Choir Aahs": 52,
            "Voice Oohs": 53,
            "Synth Voice": 54,
            "Church Organ": 19,
            "Slow Strings": 49,
        },
        {
            "Agogo": 113,
            "Woodblock": 115,
            "Melodic Tom": 117,
            "Synth Drum": 118,
            "Timpani": 47,
            "Pizzicato Strings": 45,
            "Muted Guitar": 28,
        },
        (KIT_ELECTRONIC, KIT_DANCE),
    ),
}


GENRES["Phonk/Memphis"] = _p(
    "Phonk/Memphis",
    {
        **GENRES["Phonk"].programs,
        "melody": (0, 19, 14),
        "chords": (48, 49, 52, 19),
        "arp": (113, 115, 117),
        "pluck": (113, 117, 45),
    },
    KIT_808,
    GENRES["Phonk"].texture_long,
    GENRES["Phonk"].texture_short,
    kits=GENRES["Phonk"].kits,
)
GENRES["Phonk/House"] = _p(
    "Phonk/House",
    {
        **GENRES["Phonk"].programs,
        "chords": (90, 89, 50, 19),
        "pad": (89, 91, 95, 50),
    },
    KIT_808,
    GENRES["Phonk"].texture_long,
    GENRES["Phonk"].texture_short,
    kits=(KIT_DANCE, KIT_ELECTRONIC),
)
FLAVORS = {"Phonk": ("Phonk", "Phonk", "Phonk/Memphis", "Phonk/House")}


def parts(style):
    return tuple(style.split(BLEND_JOIN)) if style else (style,)


def primary(style):
    return parts(style)[0].split(FLAVOR_JOIN)[0]


def flavored(style, rng):
    lead, *others = parts(style)
    choices = FLAVORS.get(lead)
    if not choices:
        return style
    return blend_name((rng.choice(choices), *others))


def blend_name(styles):
    return BLEND_JOIN.join(styles)


def blended(lead, others):
    lead = tuple(lead)
    extra = []
    for pool in others:
        for item in pool:
            if item not in lead and item not in extra:
                extra.append(item)
    return lead + lead + tuple(extra) if extra else lead


@cache
def _blend_profile(names):
    lead, *others = (GENRES[name] for name in names)
    programs = {
        role: blended(lead.programs[role], (o.programs[role] for o in others))
        for role in lead.programs
    }
    kits = []
    for kit in (*lead.kits, *(k for o in others for k in (o.kit, *o.kits))):
        if kit != lead.kit and kit not in kits:
            kits.append(kit)
    return _p(
        blend_name(names),
        programs,
        lead.kit,
        lead.texture_long,
        lead.texture_short,
        kits=tuple(kits),
    )


def profile_for(style):
    if style in GENRES:
        return GENRES[style]
    names = tuple(name for name in parts(style) if name in GENRES)
    if not names:
        return GENRES[DEFAULT_GENRE]
    if len(names) == 1:
        return GENRES[names[0]]
    return _blend_profile(names)
