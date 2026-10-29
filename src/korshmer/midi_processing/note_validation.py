from __future__ import annotations

import math
import numbers

MIDI_NOTE_MIN: int = 0
MIDI_NOTE_MAX: int = 127


MIDI_CC_MIN: int = 0

MIDI_CC_MAX: int = 127


def validate_midi_note(value, name: str = "pitch") -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"MIDI {name} must be an integer from {MIDI_NOTE_MIN} to "
            f"{MIDI_NOTE_MAX}, got {value!r}"
        )
    if not isinstance(value, numbers.Integral):
        if math.isnan(value) or not float(value).is_integer():
            raise ValueError(
                f"MIDI {name} must be a whole number from {MIDI_NOTE_MIN} to "
                f"{MIDI_NOTE_MAX}, got {value!r}"
            )
    note = int(value)
    if not MIDI_NOTE_MIN <= note <= MIDI_NOTE_MAX:
        raise ValueError(
            f"MIDI {name} must be between {MIDI_NOTE_MIN} and "
            f"{MIDI_NOTE_MAX}, got {note}"
        )
    return note


def validate_midi_cc_number(value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"MIDI CC controller number must be a whole number from "
            f"{MIDI_CC_MIN} to {MIDI_CC_MAX}, got {value!r}"
        )
    if not isinstance(value, numbers.Integral):
        if math.isnan(value) or not float(value).is_integer():
            raise ValueError(
                f"MIDI CC controller number must be a whole number from "
                f"{MIDI_CC_MIN} to {MIDI_CC_MAX}, got {value!r}"
            )
    cc_number = int(value)
    if not MIDI_CC_MIN <= cc_number <= MIDI_CC_MAX:
        raise ValueError(
            f"MIDI CC controller number must be between {MIDI_CC_MIN} and "
            f"{MIDI_CC_MAX}, got {cc_number}"
        )
    return cc_number


def is_valid_midi_note(value) -> bool:
    try:
        validate_midi_note(value)
    except ValueError:
        return False
    return True


MAX_CHORD_PROGRESSION_LENGTH: int = 256


def validate_chord_progression(chords, name: str = "chord progression"):
    if not isinstance(chords, (list, tuple)):
        raise ValueError(
            f"{name} must be a list or tuple, got {type(chords).__name__}"
        )
    count = len(chords)
    if count < 1:
        raise ValueError(f"{name} must contain at least 1 chord, got {count}")
    if count > MAX_CHORD_PROGRESSION_LENGTH:
        raise ValueError(
            f"{name} must contain at most {MAX_CHORD_PROGRESSION_LENGTH} "
            f"chords, got {count}"
        )
    return chords


def fold_into_midi_range(pitch: int) -> int:
    pitch = int(pitch)
    while pitch > MIDI_NOTE_MAX:
        pitch -= 12
    while pitch < MIDI_NOTE_MIN:
        pitch += 12
    return pitch
