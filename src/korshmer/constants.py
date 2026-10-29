import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent.parent.parent

BASE_DIR = BASE_DIR.resolve()

FLUID_DIR = BASE_DIR / "fluidsynth"

# The bundled SoundFont lives in fluidsynth/soundfont.sf3. To try a different
# (e.g. higher quality) GM/GS bank without overwriting the bundled file, point
# the KORSHMER_SOUNDFONT environment variable at any .sf3/.sf3 file.
_SF3_OVERRIDE = os.environ.get("KORSHMER_SOUNDFONT")
if _SF3_OVERRIDE:
    SF3_PATH = Path(_SF3_OVERRIDE).expanduser()
else:
    SF3_PATH = FLUID_DIR / "soundfont.sf3"

DATA_DIR = BASE_DIR / "saved_data"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
CRASH_REPORTS_DIR = LOGS_DIR / "crash_reports"
INPUT_LIST_FILE = DATA_DIR / "inputs.json"
AUDIO_HOST_FILE = DATA_DIR / "audio_host.json"
PRESETS_FILE = DATA_DIR / "presets.json"

RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

MIN_NOTES_FOR_SPLIT = 60
SPLIT_CLUSTERS = 2
SPECTRUM_MAX_DURATION = 30.0

MAX_AUDIO_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROJECT_MANIFEST_BYTES = 256 * 1024 * 1024
MAX_PROJECT_ARCHIVE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROJECT_ARCHIVE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024

STEM_SEPARATION_TIMEOUT_SECONDS = None

MIN_PLAYBACK_BUFFER_SIZE = 512
MAX_PLAYBACK_BUFFER_SIZE = 8192
DEFAULT_PLAYBACK_BUFFER_SIZE = 2048
AUDIO_DEVICE_OPEN_ATTEMPTS = 3
AUDIO_DEVICE_RETRY_DELAY_SECONDS = 0.25
AUDIO_FALLBACK_DEVICE_TRIES = 4
BUFFER_UNDERRUN_THRESHOLD = 3
BUFFER_UNDERRUN_WINDOW_SECONDS = 10.0
BUFFER_CPU_HIGH_PERCENT = 85.0
PERFORMANCE_CPU_SPIKE_PERCENT = BUFFER_CPU_HIGH_PERCENT
PERFORMANCE_LOGGER_NAME = "korshmer.performance"
BUFFER_CPU_LOW_PERCENT = 30.0
BUFFER_LOW_LOAD_SECONDS = 30.0

MAX_AUDIO_CHANNELS = 8
MAX_EFFECT_CHAIN_DEPTH = 16
MAX_DECODED_AUDIO_BYTES = 4 * 1024 * 1024 * 1024
DECODED_AUDIO_BYTES_PER_SAMPLE = 4
SEGMENT_CACHE_MAX_BYTES = 100 * 1024 * 1024
STREAMING_AUDIO_THRESHOLD_BYTES = 100 * 1024 * 1024

STEM_FILTER_PROGRESS = 0.35

STEM_TYPE_PATTERNS = {
    "drums": ["drums"],
    "bass": ["bass"],
    "vocals": ["vocals", "vocal"],
    "guitar": ["guitar"],
    "piano": ["piano"],
    "other": ["other"],
}

SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Locrian": [0, 1, 3, 5, 6, 8, 10],
    "Chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Pentatonic Major": [0, 2, 4, 7, 9],
    "Pentatonic Minor": [0, 3, 5, 7, 10],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Spanish (Phrygian Dominant)": [0, 1, 4, 5, 7, 8, 10],
    "Gypsy Minor": [0, 2, 3, 6, 7, 8, 11],
    "Arabian (Maqam Hijaz)": [0, 1, 4, 5, 7, 8, 10],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

GRID_RESOLUTIONS = {
    "1/4": 1,
    "1/8": 2,
    "1/16": 4,
    "1/32": 8,
    "1/64": 16,
    "1/8 Triplet": 3,
    "1/16 Triplet": 6,
}

MICROTONAL_DIVISIONS = {
    "Semitones (12-TET)": 12,
    "Quartertones (24-TET)": 24,
    "Eighthtones (48-TET)": 48,
    "31-TET": 31,
    "53-TET": 53,
}

GM_DRUM_MAP = {
    "Kick": 36,
    "Snare": 38,
    "HiHat Closed": 42,
    "HiHat Open": 46,
    "Tom Low": 45,
    "Tom Mid": 47,
    "Tom High": 50,
    "Crash": 49,
    "Ride": 51,
}

INSTRUMENT_DB = {
    "keys": [
        ("Acoustic Grand Piano", 21, 108, 0),
        ("Bright Acoustic Piano", 21, 108, 1),
        ("Electric Piano 1", 28, 103, 4),
        ("Harpsichord", 29, 89, 6),
        ("Clavinet", 29, 88, 7),
    ],
    "chromatic": [
        ("Vibraphone", 53, 89, 11),
        ("Marimba", 45, 96, 12),
        ("Celesta", 60, 108, 8),
        ("Glockenspiel", 72, 108, 9),
        ("Xylophone", 65, 108, 13),
    ],
    "organ": [
        ("Rock Organ", 36, 96, 18),
        ("Drawbar Organ", 36, 96, 16),
        ("Percussive Organ", 36, 96, 17),
        ("Church Organ", 21, 108, 19),
        ("Accordion", 53, 93, 21),
        ("Harmonica", 60, 96, 22),
    ],
    "guitar": [
        ("Acoustic Guitar (Steel)", 40, 88, 25),
        ("Electric Guitar (Clean)", 40, 88, 27),
        ("Acoustic Guitar (Nylon)", 40, 84, 24),
        ("Electric Guitar (Jazz)", 40, 88, 26),
        ("Overdrive Guitar", 40, 91, 29),
        ("Distortion Guitar", 40, 91, 30),
    ],
    "bass": [
        ("Electric Bass (Finger)", 28, 67, 33),
        ("Pick Bass", 28, 67, 34),
        ("Fretless Bass", 28, 67, 35),
        ("Acoustic Bass", 28, 67, 32),
        ("Slap Bass 1", 28, 67, 36),
        ("Synth Bass 1", 21, 80, 38),
        ("Synth Bass 2", 21, 80, 39),
    ],
    "strings": [
        ("String Ensemble 1", 36, 96, 48),
        ("Cello", 36, 88, 42),
        ("Violin", 55, 103, 40),
        ("Viola", 48, 93, 41),
        ("Contrabass", 24, 67, 43),
        ("Tremolo Strings", 36, 96, 44),
        ("Pizzicato Strings", 36, 96, 45),
        ("Orchestral Harp", 24, 103, 46),
    ],
    "vocals": [
        ("Choir Aahs", 36, 96, 52),
        ("Voice Oohs", 36, 96, 53),
        ("Synth Voice", 36, 96, 54),
        ("Orchestra Hit", 36, 84, 55),
    ],
    "brass": [
        ("Brass Section", 36, 96, 61),
        ("Trumpet", 54, 86, 56),
        ("Trombone", 40, 77, 57),
        ("French Horn", 34, 77, 60),
        ("Tuba", 26, 65, 58),
        ("Muted Trumpet", 52, 88, 59),
        ("Synth Brass 1", 36, 96, 62),
    ],
    "woodwind": [
        ("Flute", 60, 98, 73),
        ("Tenor Sax", 44, 76, 66),
        ("Alto Sax", 49, 81, 65),
        ("Soprano Sax", 56, 88, 64),
        ("Clarinet", 50, 94, 71),
        ("Oboe", 58, 91, 68),
        ("Pan Flute", 60, 96, 75),
    ],
    "synth_lead": [
        ("Lead 1 (Square)", 24, 108, 80),
        ("Lead 2 (Sawtooth)", 24, 108, 81),
        ("Lead 3 (Calliope)", 36, 96, 82),
        ("Lead 8 (Bass+Lead)", 21, 96, 87),
    ],
    "synth_pad": [
        ("Pad 2 (Warm)", 36, 96, 89),
        ("Pad 1 (New Age)", 36, 96, 88),
        ("Pad 3 (Polysynth)", 36, 96, 90),
        ("Pad 4 (Choir)", 36, 96, 91),
        ("Pad 8 (Sweep)", 24, 96, 95),
    ],
    "ethnic": [
        ("Sitar", 43, 89, 104),
        ("Banjo", 48, 84, 105),
        ("Shamisen", 50, 81, 106),
        ("Koto", 43, 84, 107),
        ("Kalimba", 60, 84, 108),
        ("Bag pipe", 58, 79, 109),
    ],
}

DRUM_MAPPINGS = {
    "Addictive Drums 2": {
        36: 36,
        38: 38,
        42: 42,
        46: 46,
        49: 49,
        51: 51,
        45: 65,
        47: 67,
        50: 69,
    },
    "EZDrummer / Superior": {
        36: 36,
        38: 38,
        42: 46,
        46: 25,
        49: 49,
        51: 51,
        45: 47,
        47: 45,
        50: 50,
    },
    "Steven Slate": {
        36: 36,
        38: 38,
        42: 46,
        46: 49,
        49: 57,
        51: 59,
        45: 41,
        47: 43,
        50: 45,
    },
}

HELP_TEXTS = {
    "arp": "Arpeggiates chords into rhythmic patterns.",
    "bass": "Generates a mixed-style dynamic bassline.",
    "beat": "Generates a basic drum beat (Rock, Techno, etc).",
    "bpm": "Beats per minute (BPM), higher BPM is faster.",
    "osc": (
        "Listen for OSC messages on UDP port 8000 (this computer only): "
        "/transport/play, /transport/pause, /transport/stop and /tempo."
    ),
    "obs": (
        "Start and stop an OBS Studio recording together with playback "
        "(obs-websocket on ws://127.0.0.1:4455; set KORSHMER_OBS_URL and "
        "KORSHMER_OBS_PASSWORD or the system keychain to change it)."
    ),
    "link": (
        "Sync the tempo with other Ableton Link devices on the local network."
    ),
    "cc": "Generates CC11 expression from audio volume.",
    "clean": "Removes extremely short noise notes.",
    "clip": "Audio clip (e.g. separated vocals) placed on the editor timeline. Drag to move, double-click to mute.",
    "midi_generator": "One generator for whole songs, layers and continuations. Sections, roles and the length in minutes all come from the same seed.",
    "generator_mode": "New song builds from nothing. Continue editor starts after the end of the editor and extends its melody. Layer over editor adds new tracks on top of the song in the editor: its length, key, chords and parts (intro, verse, chorus, break) are read from the notes already there. A song that already has drums gets no new drum track. The form of the song and how closely a layer follows the chords of the editor are chosen automatically.",
    "generator_minutes": "Wanted length. The sections fill it exactly, so a longer length means more sections, not a slower tempo.",
    "generator_roles": "Which instruments are generated. Each role lands on its own editor lane, and every section keeps at least two of them playing.",
    "seed": "Same seed gives the same result. 0 picks a fresh random seed every time.",
    "lane": "Each lane is one instrument track. Click a lane to make it the target for new notes.",
    "snap_grid": "Snaps note edits to the current grid. Hold Alt while dragging to bypass.",
    "stems_editor": "Loads separated stems into the editor: vocals as a draggable audio clip, instruments as MIDI lanes.",
    "separate": "Splits the selected song into stems (vocals, drums, bass, guitar, piano, other) without MIDI conversion.",
    "click": "Adds a synchronized metronome click track.",
    "comp": "Compresses velocity range (boosts quiet notes).",
    "cresc": "Creates gradual volume change over time.",
    "delay": "Adds MIDI echo notes with fading velocity.",
    "detune": "Adds subtle pitch drift for an analog/Lo-Fi feel.",
    "double": "Duplicates track an octave up or down.",
    "drop2": "Jazz voicing: Drops 2nd highest note an octave down.",
    "euc": "Generates mathematical Euclidean rhythms.",
    "fill": "Adds drum fills at the end of bars.",
    "filter": "High/Low pass frequency filter.",
    "gate": "Silences audio below threshold.",
    "ghost": "Adds quiet ghost notes between main drum hits.",
    "glitch": "Merges very short adjacent notes of same pitch.",
    "gm": "Maps drums to General MIDI standard.",
    "grid": "Rhythmic grid resolution (quantization target).",
    "groove": "Applies micro-timing shifts for rhythmic feel.",
    "harm": "Generates harmony voices based on scale.",
    "human": "Adds subtle human imperfections to timing/velocity.",
    "inv_vel": "Inverts velocity dynamics (loud becomes quiet).",
    "inv": "Smart chord inversions for smoother voice leading.",
    "json": "Exports MIDI data to JSON format.",
    "len": "Adjusts note lengths: Legato connects, Staccato shortens.",
    "lfo": "Applies low-frequency modulation to parameters like velocity or CC.",
    "limit": "Confines notes to a specific pitch range.",
    "lofi": "Reduces velocity resolution (Bitcrush effect).",
    "loop": "Stretches/Compresses the track to fit the exact bar count.",
    "melody": "Generates a random melody in the selected scale.",
    "mirror": "Inverts pitches around a center axis.",
    "mono_mix": "Mixes stereo audio to mono.",
    "mono": "Forces polyphonic track to monophonic melody.",
    "music_only": "Detects speech, jingles and effects (for example a music-video intro or a spoken break) and cuts them out before separation, so they never become MIDI notes and leave no gap: the music after a cut moves earlier, kept in step with the beat. The stems, the MIDI and the A/B source copy all use the shortened timeline.",
    "nogap": "Removes empty bars (silence) from MIDI.",
    "norm": "Maximizes audio volume to 0dB.",
    "octave": "Global pitch transposition by full octaves.",
    "open": "Spreads notes over a wider range (Open Voicing).",
    "pad": "Generates a long pad layer from chords.",
    "pedal": "Adds Sustain Pedal (CC64) messages.",
    "pitch": "Shifts the pitch of the audio by semitones.",
    "prog": "Generates a chord progression in the selected scale.",
    "quantize": "Aligns notes to grid. 100% is robotic accuracy.",
    "ramp": "Creates a velocity ramp from start to end.",
    "randvel": "Randomizes velocities within range.",
    "ratchet": "Splits notes into rapid repeats (rolls). Great for hi-hats.",
    "remap": "Remaps drums to plugin standards (Addictive, EZDrummer).",
    "retro": "Reverses MIDI note order.",
    "rev_aud": "Reverses audio playback.",
    "rhythm": "Flattens pitch to extract pure rhythm.",
    "root": "Base musical key anchor for harmonic calculations.",
    "scale": "Musical scale pattern (Major, Minor, etc).",
    "separation": "Fast: one Demucs pass. Balanced: a dedicated vocal model first, then Demucs on the rest, for cleaner vocals and instruments (about 3x slower on CPU; the first run downloads a 67 MB vocal model). Best: more passes, slowest.",
    "slice": "Slices audio into samples based on amplitude/transients.",
    "snap": "Forces out-of-scale notes to the nearest scale note.",
    "speed": "Time stretches the entire file.",
    "split": "Splits one track into High/Low (Left/Right hand).",
    "stats": "Generates a text report with scale/note statistics.",
    "stems": "Select instruments to extract/convert to MIDI.",
    "strum": "Staggers note start times to simulate strumming.",
    "swing": "Delays weak beats for a groovy feel.",
    "thin": "Randomly removes notes to reduce density.",
    "transient": "Sharpens attack transients. Good for drum detection.",
    "transpose": "Shifts all notes up or down by a specific number of semitones.",
    "vel_ramp": "Creates a gradual linear change in velocity over time.",
    "velocity": "Dynamics control: Compressed reduces range, Fixed flattens it.",
    "voice_lead": "Optimizes note movement between chords for smoother transitions.",
    "wav32": "Saves separated audio stems in 32-bit Float.",
}
