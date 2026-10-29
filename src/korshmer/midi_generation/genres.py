from dataclasses import dataclass, field

DEFAULT_GENRE = "Pop"

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
            "melody": (0, 4, 11, 27, 25),
            "chords": (0, 4, 25, 27),
            "bass": (33, 34, 35),
            "pad": (89, 88, 50, 49),
            "arp": (46, 12, 24),
            "pluck": (24, 45, 12),
            "counter": (40, 73, 41, 71),
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
            "melody": (29, 30, 27, 28),
            "chords": (29, 30, 27),
            "bass": (34, 33),
            "pad": (18, 16, 17, 48),
            "arp": (27, 25, 28),
            "pluck": (28, 29, 25),
            "counter": (18, 16, 40),
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
            "melody": (4, 5, 11, 8, 14, 9),
            "chords": (4, 5, 2, 0),
            "bass": (38, 39),
            "pad": (89, 94, 91, 99),
            "arp": (8, 9, 10),
            "pluck": (9, 10, 112),
            "counter": (53, 54, 41),
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
            "melody": (4, 5, 11, 24, 0),
            "chords": (4, 2, 0, 24),
            "bass": (35, 33, 32),
            "pad": (88, 89, 49),
            "arp": (46, 10, 24),
            "pluck": (24, 10, 11),
            "counter": (11, 71, 53),
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
            "chords": (0, 26, 4),
            "bass": (32, 33, 35),
            "pad": (49, 89, 52),
            "arp": (11, 26, 0),
            "pluck": (26, 28, 11),
            "counter": (71, 73, 66, 59),
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
            "chords": (62, 81, 90, 86),
            "bass": (38, 39),
            "pad": (89, 95, 91, 94),
            "arp": (98, 100, 9, 112),
            "pluck": (80, 83, 98),
            "counter": (81, 54, 84),
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
}


def profile_for(style):
    return GENRES.get(style, GENRES[DEFAULT_GENRE])
