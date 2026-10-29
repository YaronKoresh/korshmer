import inspect
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

from korshmer.audio_processing.validation import validate_time_signature
from korshmer.utils.tempo import validate_bpm
from korshmer.utils.timeline import validate_timeline_time


class MIDIFileFormat(Enum):
    TYPE_0 = 0
    TYPE_1 = 1
    TYPE_2 = 2


_LYRIC_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

_LYRIC_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_lyrics_text(text) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _LYRIC_CONTROL_CHARS_RE.sub("", text)
    return _LYRIC_LONE_SURROGATE_RE.sub("", cleaned)


KEY_SIGNATURE_MIN: int = -7
KEY_SIGNATURE_MAX: int = 7


def validate_key_signature(key, is_major=True) -> tuple[int, bool]:
    if isinstance(key, (bool, np.bool_)) or not isinstance(
        key, (int, np.integer)
    ):
        raise ValueError(
            "Key signature must be an integer count of sharps (positive) or "
            f"flats (negative), got {key!r}"
        )
    key = int(key)
    if not KEY_SIGNATURE_MIN <= key <= KEY_SIGNATURE_MAX:
        raise ValueError(
            f"Key signature must be between {KEY_SIGNATURE_MIN} and "
            f"{KEY_SIGNATURE_MAX}, got {key}"
        )
    if not isinstance(is_major, (bool, np.bool_)):
        raise ValueError(
            f"Key signature mode (is_major) must be a bool, got {is_major!r}"
        )
    return key, bool(is_major)


@dataclass
class TempoChange:
    time: float
    bpm: float


@dataclass
class TimeSignature:
    time: float
    numerator: int
    denominator: int


@dataclass
class KeySignature:
    time: float
    key: int
    is_major: bool


@dataclass
class TextEvent:
    time: float
    text: str
    event_type: str


@dataclass
class Marker:
    time: float
    name: str


class MIDIFileEnhancer:
    def __init__(self):
        self.tempo_map: list[TempoChange] = []
        self.time_signatures: list[TimeSignature] = []
        self.key_signatures: list[KeySignature] = []
        self.lyrics: list[TextEvent] = []
        self.markers: list[Marker] = []
        self.text_events: list[TextEvent] = []

    def add_tempo_change(self, time: float, bpm: float):
        time = validate_timeline_time(time)
        if time is None:
            return
        self.tempo_map.append(TempoChange(time=time, bpm=validate_bpm(bpm)))
        self.tempo_map.sort(key=lambda x: x.time)

    def add_time_signature(self, time: float, numerator: int, denominator: int):
        time = validate_timeline_time(time)
        if time is None:
            return
        numerator, denominator = validate_time_signature(
            (numerator, denominator)
        )
        self.time_signatures.append(
            TimeSignature(
                time=time, numerator=numerator, denominator=denominator
            )
        )
        self.time_signatures.sort(key=lambda x: x.time)

    def add_key_signature(self, time: float, key: int, is_major: bool = True):
        time = validate_timeline_time(time)
        if time is None:
            return
        key, is_major = validate_key_signature(key, is_major)
        self.key_signatures.append(
            KeySignature(time=time, key=key, is_major=is_major)
        )
        self.key_signatures.sort(key=lambda x: x.time)

    def add_lyric(self, time: float, text: str):
        time = validate_timeline_time(time)
        if time is None:
            return
        self.lyrics.append(
            TextEvent(
                time=time,
                text=sanitize_lyrics_text(text),
                event_type="lyric",
            )
        )
        self.lyrics.sort(key=lambda x: x.time)

    def add_marker(self, time: float, name: str):
        time = validate_timeline_time(time)
        if time is None:
            return
        self.markers.append(Marker(time=time, name=name))
        self.markers.sort(key=lambda x: x.time)

    def add_text_event(self, time: float, text: str, event_type: str = "text"):
        time = validate_timeline_time(time)
        if time is None:
            return
        self.text_events.append(
            TextEvent(time=time, text=text, event_type=event_type)
        )
        self.text_events.sort(key=lambda x: x.time)

    def get_tempo_at_time(self, time: float) -> float:
        if not self.tempo_map:
            return 120.0

        for i in range(len(self.tempo_map) - 1, -1, -1):
            if self.tempo_map[i].time <= time:
                return self.tempo_map[i].bpm

        return self.tempo_map[0].bpm if self.tempo_map else 120.0

    def get_time_signature_at_time(self, time: float) -> tuple[int, int]:
        if not self.time_signatures:
            return (4, 4)

        for i in range(len(self.time_signatures) - 1, -1, -1):
            if self.time_signatures[i].time <= time:
                return (
                    self.time_signatures[i].numerator,
                    self.time_signatures[i].denominator,
                )

        ts = self.time_signatures[0]
        return (
            (ts.numerator, ts.denominator) if self.time_signatures else (4, 4)
        )

    def get_key_at_time(self, time: float) -> tuple[int, bool]:
        if not self.key_signatures:
            return (0, True)

        for i in range(len(self.key_signatures) - 1, -1, -1):
            if self.key_signatures[i].time <= time:
                return (
                    self.key_signatures[i].key,
                    self.key_signatures[i].is_major,
                )

        ks = self.key_signatures[0]
        return (ks.key, ks.is_major) if self.key_signatures else (0, True)


class MIDIFormatConverter:
    def __init__(self):
        pass

    def convert_type0_to_type1(
        self, tracks: list[list[dict]]
    ) -> list[list[dict]]:
        if len(tracks) != 1:
            return tracks

        single_track = tracks[0]
        new_tracks = []

        channels = set()
        for event in single_track:
            if "channel" in event:
                channels.add(event["channel"])

        for channel in sorted(channels):
            channel_track = [
                e for e in single_track if e.get("channel") == channel
            ]
            if channel_track:
                new_tracks.append(channel_track)

        meta_track = [e for e in single_track if "channel" not in e]
        if meta_track:
            new_tracks.insert(0, meta_track)

        return new_tracks

    def convert_type1_to_type0(
        self, tracks: list[list[dict]]
    ) -> list[list[dict]]:
        merged = []

        for track in tracks:
            merged.extend(track)

        merged.sort(key=lambda x: x.get("time", 0))

        return [merged]

    def merge_channels(
        self, tracks: list[list[dict]], channel_mapping: dict[int, int]
    ) -> list[list[dict]]:
        new_tracks = []

        for track in tracks:
            new_track = []
            for event in track:
                if "channel" in event:
                    old_channel = event["channel"]
                    new_channel = channel_mapping.get(old_channel, old_channel)
                    new_event = event.copy()
                    new_event["channel"] = new_channel
                    new_track.append(new_event)
                else:
                    new_track.append(event)
            new_tracks.append(new_track)

        return new_tracks

    def split_channels(
        self, track: list[dict], channels: list[int]
    ) -> list[list[dict]]:
        channel_tracks = {ch: [] for ch in channels}

        for event in track:
            if "channel" in event:
                ch = event["channel"]
                if ch in channel_tracks:
                    channel_tracks[ch].append(event)

        return [channel_tracks[ch] for ch in channels if channel_tracks[ch]]


class SysExHandler:
    def __init__(self):
        self.sysex_messages: list[tuple[float, bytes]] = []

    def add_sysex(self, time: float, data: bytes):

        from korshmer.midi.message_validator import (
            validate_sysex,
            validate_sysex_time,
        )

        time = validate_sysex_time(time)
        data = validate_sysex(data)
        self.sysex_messages.append((time, data))
        self.sysex_messages.sort(key=lambda x: x[0])

    def get_sysex_at_time(self, time: float) -> list[bytes]:
        return [
            data for t, data in self.sysex_messages if abs(t - time) < 0.001
        ]


_MIDI2_VELOCITY_MAX = 65535
_MIDI2_VELOCITY_SCALE = 512
_MIDI_PITCH_BEND_MIN = -8192
_MIDI_PITCH_BEND_MAX = 16383


def _event_number(event: dict, key: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"MIDI event {event.get('type')!r} needs a numeric {key!r}, "
            f"got {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"MIDI event {event.get('type')!r} has a non-finite {key!r}"
        )
    return int(round(value))


class MIDI2Support:
    def __init__(self):
        self.resolution_multiplier = 16

    def convert_midi1_to_midi2(self, events: list[dict]) -> list[dict]:
        midi2_events = []

        for event in events:
            new_event = event.copy()
            event_type = event.get("type")

            if event_type in ("note_on", "note_off"):
                velocity = min(127, max(0, _event_number(event, "velocity")))
                new_event["velocity"] = velocity * _MIDI2_VELOCITY_SCALE
                if event_type == "note_on":
                    new_event["attribute_type"] = 0
                    new_event["attribute_data"] = 0

            elif event_type == "pitch_bend":
                value = _event_number(event, "value")
                value = min(
                    _MIDI_PITCH_BEND_MAX, max(_MIDI_PITCH_BEND_MIN, value)
                )
                new_event["value"] = value * 65536 // 16384

            elif event_type == "control_change":
                value = min(127, max(0, _event_number(event, "value")))
                new_event["value"] = value * 65536 // 128

            midi2_events.append(new_event)

        return midi2_events

    def convert_midi2_to_midi1(self, events: list[dict]) -> list[dict]:
        midi1_events = []

        for event in events:
            new_event = event.copy()
            event_type = event.get("type")

            if event_type in ("note_on", "note_off"):
                velocity = min(
                    _MIDI2_VELOCITY_MAX,
                    max(0, _event_number(event, "velocity")),
                )
                converted = velocity // _MIDI2_VELOCITY_SCALE
                if event_type == "note_on" and velocity > 0:
                    converted = max(1, converted)
                new_event["velocity"] = min(127, converted)
                new_event.pop("attribute_type", None)
                new_event.pop("attribute_data", None)

            elif event_type == "pitch_bend":
                value = _event_number(event, "value")
                value = min(
                    _MIDI_PITCH_BEND_MAX * 4,
                    max(_MIDI_PITCH_BEND_MIN * 4, value),
                )
                new_event["value"] = value * 16384 // 65536

            elif event_type == "control_change":
                value = min(65535, max(0, _event_number(event, "value")))
                new_event["value"] = value * 128 // 65536

            midi1_events.append(new_event)

        return midi1_events


class MIDIBatchProcessor:
    def __init__(self):
        self.operations: list[tuple[str, dict]] = []

    @staticmethod
    def _resolve_operation(operation: str):
        from korshmer import midi_processors

        if not isinstance(operation, str) or not operation.isidentifier():
            raise ValueError(f"Unknown batch operation: {operation!r}")
        func = getattr(midi_processors, f"process_{operation}", None)
        if not callable(func):
            raise ValueError(f"Unknown batch operation: {operation!r}")
        return func

    def add_operation(self, operation: str, params: dict | None = None):
        params = {} if params is None else dict(params)
        func = self._resolve_operation(operation)
        try:
            inspect.signature(func).bind("", **params)
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for batch operation {operation!r}: {e}"
            ) from e
        self.operations.append((operation, params))

    def process_files(self, file_paths: list[str]) -> list[str]:
        processed_paths = []

        for file_path in file_paths:
            output_path = self._process_single_file(file_path)
            if output_path:
                processed_paths.append(output_path)

        return processed_paths

    def _process_single_file(self, file_path: str) -> str | None:
        current = file_path
        for operation, params in self.operations:
            try:
                current = self._resolve_operation(operation)(current, **params)
            except Exception as e:
                logging.error(
                    f"Batch operation {operation!r} failed for {file_path}: {e}"
                )
                return None
            if not current:
                return None
        return current


class CuePointManager:
    def __init__(self):
        self.cue_points: list[Marker] = []

    def add_cue_point(self, time: float, name: str):
        time = validate_timeline_time(time)
        if time is None:
            return
        self.cue_points.append(Marker(time=time, name=name))
        self.cue_points.sort(key=lambda x: x.time)

    def get_cue_point(self, index: int) -> Marker | None:
        if 0 <= index < len(self.cue_points):
            return self.cue_points[index]
        return None

    def jump_to_cue(self, index: int, engine) -> bool:
        cue = self.get_cue_point(index)
        if cue is None:
            return False
        engine.setPosition(int(round(cue.time * 1000)))
        return True

    def find_nearest_cue(self, time: float) -> Marker | None:
        if not self.cue_points:
            return None

        nearest = min(self.cue_points, key=lambda c: abs(c.time - time))
        return nearest

    def get_section_cues(self, section_names: list[str]) -> list[Marker]:
        return [
            c
            for c in self.cue_points
            if any(name.lower() in c.name.lower() for name in section_names)
        ]
