import re
import threading
from typing import Any

MIDI_CONVERTIBLE_INSTRUMENTS: frozenset[str] = frozenset(
    {
        "piano",
        "guitar",
        "bass",
        "drums",
        "keys",
        "strings",
        "brass",
        "woodwind",
        "synth",
        "organ",
        "percussion",
        "bell",
        "marimba",
        "vibraphone",
        "xylophone",
    }
)

NON_MIDI_INSTRUMENTS: frozenset[str] = frozenset(
    {
        "vocals",
        "speech",
        "effects",
        "ambient",
        "noise",
        "applause",
        "crowd",
        "rain",
        "thunder",
        "wind",
        "water",
    }
)

_whitelist_lock = threading.Lock()


class InstrumentFilter:
    _normalized_cache: dict[str, str] = {}
    _cache_lock = threading.Lock()
    _whitelist: set[str] = set()
    _blacklist: set[str] = set()

    @classmethod
    def set_whitelist(cls, instruments: list[str] | None) -> None:
        with _whitelist_lock:
            cls._whitelist = (
                {cls._normalize_name(i) for i in instruments}
                if instruments
                else set()
            )

    @classmethod
    def set_blacklist(cls, instruments: list[str] | None) -> None:
        with _whitelist_lock:
            cls._blacklist = (
                {cls._normalize_name(i) for i in instruments}
                if instruments
                else set()
            )

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        normalized = cls._normalized_cache.get(name)
        if normalized is not None:
            return normalized

        with cls._cache_lock:
            normalized = cls._normalized_cache.get(name)
            if normalized is not None:
                return normalized
            normalized = re.sub(r"[_\-\s]+", " ", name.lower()).strip()
            cls._normalized_cache[name] = normalized
            return normalized

    @classmethod
    def _fuzzy_match(
        cls, instrument_name: str, target_set: frozenset[str]
    ) -> bool:
        normalized = cls._normalize_name(instrument_name)

        if normalized in target_set:
            return True

        words = normalized.split()
        for target in target_set:
            target_words = target.split() if " " in target else [target]
            if any(word in target_words for word in words):
                return True

        return False

    @classmethod
    def should_convert(cls, instrument_name: str | None) -> bool:
        if not instrument_name:
            return False

        normalized = cls._normalize_name(instrument_name)

        with _whitelist_lock:
            if normalized in cls._blacklist:
                return False
            if normalized in cls._whitelist:
                return True

        if cls._fuzzy_match(instrument_name, NON_MIDI_INSTRUMENTS):
            return False

        if cls._fuzzy_match(instrument_name, MIDI_CONVERTIBLE_INSTRUMENTS):
            return True

        return False

    @classmethod
    def filter_stems(cls, stems_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            name: audio
            for name, audio in stems_dict.items()
            if cls.should_convert(name)
        }

    @classmethod
    def get_convertible_stems(cls, stem_names: list[str]) -> list[str]:
        return [name for name in stem_names if cls.should_convert(name)]


def should_convert_to_midi(instrument_name: str | None) -> bool:
    return InstrumentFilter.should_convert(instrument_name)
