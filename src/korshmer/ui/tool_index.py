from __future__ import annotations

from dataclasses import dataclass

from korshmer.ui.widgets.material.mode_tool import ModeToolCard

NAME_START = 0
NAME_WORD = 1
NAME_PART = 2
KEYWORD_PART = 3
GROUP_PART = 4
HELP_PART = 5
DEFAULT_LIMIT = 12
KEYWORDS = {
    "Polish": "clean tidy fix smooth",
    "Remove short notes": "clean noise glitch tiny",
    "Remove empty bars": "silence gap shorten",
    "Transpose": "pitch key higher lower semitone",
    "Snap to scale": "wrong notes key fix",
    "Invert chords": "inversion voicing",
    "Remap drums": "drum kit map",
    "Create variation": "different version change remix",
    "Detect chords": "what chords name",
    "Detect scales": "what key",
    "Smart loop": "repeat",
    "Speed x2 (faster)": "tempo double time quick",
    "Speed x0.5 (slower)": "tempo half time slow",
    "Groove": "swing shuffle feel",
    "Advanced quantize": "timing tighten beat grid quantise",
    "Strum chords": "guitar",
    "MIDI delay": "echo repeat",
    "Thin out": "fewer notes simplify less busy",
    "Arpeggiate": "arp broken chord",
    "Advanced arpeggio": "arp broken chord",
    "Harmonize": "harmony chords thirds",
    "Force mono": "one note melody monophonic",
    "Double octave": "thicker fuller",
    "MIDI Generator": "create compose build generate song drums bass melody",
    "Humanize": "natural human feel loosen",
    "Velocity ramp": "louder quieter fade volume",
    "Crescendo": "louder quieter fade volume build",
    "Random velocity": "louder quieter natural",
    "Parametric EQ": "bass treble tone equalizer brighter",
    "Compressor": "punch squash loud even",
    "Reverb": "room hall space",
    "Delay": "echo repeat",
    "Limiter": "louder loud peak clip",
    "Filter": "low pass high pass muffled",
    "Gate": "noise silence remove hiss",
    "Normalize": "louder volume level loud",
    "Pitch shift": "transpose key higher lower",
    "Mono mix": "one channel",
    "Reverse": "backwards",
    "Slice by silence": "cut split chop",
    "Smart mastering": "master finish final loud",
    "Range EQ": "bass treble tone equalizer",
}


@dataclass(frozen=True)
class ToolEntry:
    category: str
    group: str
    name: str
    help: str
    page: int
    card: object = None
    keywords: str = ""

    @property
    def label(self) -> str:
        if self.group and self.group != self.name:
            return f"{self.name}  ·  {self.group}"
        return self.name

    @property
    def place(self) -> str:
        return self.category


def collect_tools(categories) -> list[ToolEntry]:
    entries = []
    for page, (category, widget) in enumerate(categories):
        cards = widget.findChildren(ModeToolCard)
        if not cards:
            entries.append(
                ToolEntry(
                    category,
                    category,
                    category,
                    "",
                    page,
                    keywords=KEYWORDS.get(category, ""),
                )
            )
            continue
        for card in cards:
            for mode in card.modes:
                entries.append(
                    ToolEntry(
                        category,
                        card.title,
                        mode.label,
                        mode.help or "",
                        page,
                        card,
                        getattr(mode, "keywords", "")
                        or KEYWORDS.get(mode.label, ""),
                    )
                )
    return entries


def _rank(entry, words):
    name = entry.name.lower()
    around = f"{entry.group} {entry.category}".lower()
    helped = entry.help.lower()
    worst = NAME_START
    for word in words:
        if name.startswith(word):
            rank = NAME_START
        elif any(part.startswith(word) for part in name.split()):
            rank = NAME_WORD
        elif word in name:
            rank = NAME_PART
        elif any(key.startswith(word) for key in entry.keywords.split()):
            rank = KEYWORD_PART
        elif word in around:
            rank = GROUP_PART
        elif word in helped:
            rank = HELP_PART
        else:
            return None
        worst = max(worst, rank)
    return worst


def search_tools(entries, query, limit=DEFAULT_LIMIT) -> list[ToolEntry]:
    words = str(query or "").lower().split()
    if not words:
        return []
    ranked = []
    for order, entry in enumerate(entries):
        rank = _rank(entry, words)
        if rank is not None:
            ranked.append((rank, order, entry))
    ranked.sort(key=lambda item: item[:2])
    if ranked and ranked[0][0] < HELP_PART:
        ranked = [item for item in ranked if item[0] < HELP_PART]
    return [entry for _, _, entry in ranked[: max(1, int(limit))]]
