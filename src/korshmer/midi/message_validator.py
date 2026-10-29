from __future__ import annotations

import math
import numbers

from korshmer.midi_processing.note_validation import (
    validate_midi_cc_number,
    validate_midi_note,
)

SYSEX_START: int = 0xF0
SYSEX_END: int = 0xF7

MAX_SYSEX_LENGTH: int = 65536

MIDI_CHANNEL_MIN: int = 0
MIDI_CHANNEL_MAX: int = 15
MIDI_DATA_MAX: int = 127
PITCH_BEND_MIN: int = 0
PITCH_BEND_MAX: int = 16383


def _whole_int(value, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"MIDI {label} must be a whole number from {low} to {high}, "
            f"got {value!r}"
        )
    if not isinstance(value, numbers.Integral):
        if math.isnan(value) or not float(value).is_integer():
            raise ValueError(
                f"MIDI {label} must be a whole number from {low} to {high}, "
                f"got {value!r}"
            )
    number = int(value)
    if not low <= number <= high:
        raise ValueError(
            f"MIDI {label} must be between {low} and {high}, got {number}"
        )
    return number


def validate_sysex(data) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"SysEx data must be bytes, got {type(data).__name__}")
    size = len(data)
    if size > MAX_SYSEX_LENGTH:
        raise ValueError(
            f"SysEx message must be at most {MAX_SYSEX_LENGTH} bytes, "
            f"got {size}"
        )
    if size < 2 or data[0] != SYSEX_START or data[-1] != SYSEX_END:
        raise ValueError("SysEx message must start with 0xF0 and end with 0xF7")
    if any(b > MIDI_DATA_MAX for b in data[1:-1]):
        raise ValueError("SysEx data bytes must be in the range 0 to 127")
    return bytes(data)


def validate_sysex_time(time) -> float:
    if isinstance(time, bool) or not isinstance(time, numbers.Real):
        raise TypeError(
            f"SysEx time must be a number, got {type(time).__name__}"
        )
    value = float(time)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"SysEx time must be finite and not negative, got {time!r}"
        )
    return value


def validate_channel(channel) -> int:
    return _whole_int(channel, "channel", MIDI_CHANNEL_MIN, MIDI_CHANNEL_MAX)


def validate_velocity(velocity) -> int:
    return _whole_int(velocity, "velocity", 0, MIDI_DATA_MAX)


def validate_cc(controller, value) -> tuple[int, int]:
    return (
        validate_midi_cc_number(controller),
        _whole_int(value, "CC value", 0, MIDI_DATA_MAX),
    )


def validate_pitch_bend(value) -> int:
    return _whole_int(value, "pitch bend", PITCH_BEND_MIN, PITCH_BEND_MAX)


def validate_note_message(channel, key, velocity=0) -> tuple[int, int, int]:
    return (
        validate_channel(channel),
        validate_midi_note(key, "key"),
        validate_velocity(velocity),
    )
