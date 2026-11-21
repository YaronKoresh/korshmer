import logging
import math
import os
import random

import numpy as np
import pretty_midi

from korshmer.constants import NOTE_NAMES
from korshmer.midi.tempo_map import TempoMap
from korshmer.utils.bend_range import (
    DEFAULT_BEND_RANGE,
    bend_range_of,
    bends_clash,
    clamp_bend,
    clamp_bend_range,
    set_bend_range,
)
from korshmer.utils.instrument_names import sanitize_instrument_name
from korshmer.utils.note_slides import NoteSlide, attach_slides, played_bends
from korshmer.utils.tempo import BPM_DECIMALS, validate_bpm
from korshmer.utils.track_layout import (
    instrument_name,
    is_general_midi_name,
    merge_by_instrument,
)

_logger = logging.getLogger(__name__)

TRACK_PALETTE = [
    "#1A73E8",
    "#EA4335",
    "#1E8E3E",
    "#F29900",
    "#9334E6",
    "#12B5CB",
    "#E52592",
    "#7CB342",
    "#F4511E",
    "#00897B",
    "#C5221F",
    "#5F6368",
]

CLIP_KIND_COLORS = {
    "vocals": "#EA4335",
    "drums": "#F29900",
    "bass": "#1E8E3E",
    "guitar": "#9334E6",
    "piano": "#1A73E8",
    "other": "#5F6368",
    "audio": "#00897B",
}

DEFAULT_TRACK_NAME = "Lead"
MIN_NOTE_DURATION = 0.01
SONG_VERSION = 2
SLIDES_ON_NOTES_SINCE = 2
MAX_UNDO_STEPS = 100
MAX_CLIP_GAIN = 8.0
MIN_CLIP_SECONDS = 0.05
TRIM_TOLERANCE_S = 1e-3
WAVEFORM_POINTS = 512


def _coerce_numeric(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        _logger.warning(
            "Invalid numeric value %r; using default %s", value, default
        )
        return float(default)
    if not math.isfinite(number):
        _logger.warning(
            "Non-finite numeric value %r; using default %s", value, default
        )
        return float(default)
    return number


def clamp_pitch(pitch):
    pitch = _coerce_numeric(pitch, 60)
    return int(max(0, min(127, round(pitch))))


def clamp_velocity(velocity):
    velocity = _coerce_numeric(velocity, 100)
    return int(max(1, min(127, round(velocity))))


def note_name_for_pitch(pitch):
    pitch = clamp_pitch(pitch)
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def format_clock(seconds):
    total = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def clip_name_from_path(path):
    base = os.path.splitext(os.path.basename(str(path)))[0]
    return base or "Clip"


def probe_audio_duration(path):
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 0.0


def _envelope_from_mono(mono, points):
    count = len(mono)
    if count == 0:
        return []
    if count <= points:
        positions = np.linspace(0, count - 1, points)
        env = np.interp(positions, np.arange(count), np.abs(mono))
    else:
        chunk = count // points
        trimmed = mono[: chunk * points].reshape(points, chunk)
        env = np.sqrt(np.mean(np.square(trimmed), axis=1))
    peak = float(np.max(env)) if env.size else 0.0
    if not math.isfinite(peak) or peak <= 1e-9:
        return [0.0] * points
    return [float(v) for v in env / peak]


def _read_mono_blocks(path, points):
    import soundfile as sf

    with sf.SoundFile(str(path)) as handle:
        frames = int(handle.frames)
        rate = int(handle.samplerate)
        if frames <= 0 or rate <= 0:
            return [], 0.0
        duration = frames / float(rate)
        if frames <= points * 4:
            data = handle.read(dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            return _envelope_from_mono(mono, points), duration
        chunk = frames // points
        values = []
        for _ in range(points):
            block = handle.read(chunk, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            mono = np.mean(block, axis=1)
            values.append(float(np.sqrt(np.mean(np.square(mono)))))
        while len(values) < points:
            values.append(0.0)
        env = np.asarray(values, dtype=np.float64)
        peak = float(np.max(env)) if env.size else 0.0
        if not math.isfinite(peak) or peak <= 1e-9:
            return [0.0] * points, duration
        return [float(v) for v in env / peak], duration


def compute_clip_waveform(path, points=WAVEFORM_POINTS):
    points = max(1, int(points))
    try:
        return _read_mono_blocks(path, points)
    except Exception as sf_error:
        _logger.debug(f"soundfile waveform failed for {path}: {sf_error}")
    try:
        import librosa

        samples, rate = librosa.load(str(path), sr=None, mono=True)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0 or not rate:
            return [], 0.0
        duration = samples.size / float(rate)
        return _envelope_from_mono(samples, points), duration
    except Exception as librosa_error:
        _logger.warning(f"Waveform unavailable for {path}: {librosa_error}")
        return [], 0.0


MAX_TRACK_GAIN = 2.0
RHYTHM_BOOST = 1.25
BASS_PROGRAMS = range(32, 40)


def rhythm_boost(program, is_drum):
    return RHYTHM_BOOST if is_drum or int(program) in BASS_PROGRAMS else 1.0


def clamp_track_gain(gain):
    try:
        value = float(gain)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(MAX_TRACK_GAIN, value))


PAN_LEFT = -1.0
PAN_CENTRE = 0.0
PAN_RIGHT = 1.0
PAN_CONTROLLER = 10
PAN_CONTROLLER_CENTRE = 64
PAN_CONTROLLER_RIGHT = 127


def clamp_pan(pan):
    try:
        value = float(pan)
    except (TypeError, ValueError):
        return PAN_CENTRE
    if not math.isfinite(value):
        return PAN_CENTRE
    return max(PAN_LEFT, min(PAN_RIGHT, value))


def _controller_side(offset):
    if offset > 0:
        return PAN_CONTROLLER_RIGHT - PAN_CONTROLLER_CENTRE
    return PAN_CONTROLLER_CENTRE


def pan_to_controller(pan):
    value = clamp_pan(pan)
    return int(round(PAN_CONTROLLER_CENTRE + value * _controller_side(value)))


def controller_to_pan(value):
    offset = int(value) - PAN_CONTROLLER_CENTRE
    return clamp_pan(offset / _controller_side(offset))


def _instrument_pan(instrument):
    moves = [
        change
        for change in getattr(instrument, "control_changes", None) or ()
        if change.number == PAN_CONTROLLER
    ]
    if not moves:
        return PAN_CENTRE
    return controller_to_pan(min(moves, key=lambda move: move.time).value)


def _merged_keeping_sides(instruments):
    by_side = {}
    for instrument in instruments:
        by_side.setdefault(_instrument_pan(instrument), []).append(instrument)
    merged = []
    for same_side in by_side.values():
        merged.extend(merge_by_instrument(same_side, keep_shared_names=True))
    return merged


def _bend_points(bends):
    points = []
    for entry in bends or ():
        try:
            time, value = entry
            points.append([max(0.0, float(time)), clamp_bend(value)])
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda point: point[0])
    return points


class TrackModel:
    __slots__ = (
        "id",
        "name",
        "program",
        "is_drum",
        "color",
        "muted",
        "solo",
        "visible",
        "gain",
        "bends",
        "bend_range",
        "pan",
    )

    def __init__(
        self,
        track_id,
        name,
        program=0,
        is_drum=False,
        color=TRACK_PALETTE[0],
        muted=False,
        solo=False,
        visible=True,
        gain=1.0,
        bends=(),
        bend_range=DEFAULT_BEND_RANGE,
        pan=PAN_CENTRE,
    ):
        self.id = int(track_id)
        self.name = str(name)
        self.program = int(max(0, min(127, program)))
        self.is_drum = bool(is_drum)
        self.color = str(color)
        self.muted = bool(muted)
        self.solo = bool(solo)
        self.visible = bool(visible)
        self.gain = clamp_track_gain(gain)
        self.bends = _bend_points(bends)
        self.bend_range = clamp_bend_range(bend_range)
        self.pan = clamp_pan(pan)

    def instrument_label(self):
        return instrument_name(self.program, self.is_drum)

    def copy(self):
        return TrackModel(
            self.id,
            self.name,
            self.program,
            self.is_drum,
            self.color,
            self.muted,
            self.solo,
            self.visible,
            self.gain,
            self.bends,
            self.bend_range,
            self.pan,
        )

    def to_dict(self):
        return {
            "id": self.id,
            "gain": self.gain,
            "pan": self.pan,
            "bends": [list(point) for point in self.bends],
            "bend_range": self.bend_range,
            "name": self.name,
            "program": self.program,
            "is_drum": self.is_drum,
            "color": self.color,
            "muted": self.muted,
            "solo": self.solo,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get("id", 0),
            data.get("name", DEFAULT_TRACK_NAME),
            data.get("program", 0),
            data.get("is_drum", False),
            data.get("color", TRACK_PALETTE[0]),
            data.get("muted", False),
            data.get("solo", False),
            data.get("visible", True),
            data.get("gain", 1.0),
            data.get("bends") or (),
            data.get("bend_range", DEFAULT_BEND_RANGE),
            data.get("pan", PAN_CENTRE),
        )


class NoteModel:
    __slots__ = (
        "id",
        "track_id",
        "pitch",
        "start",
        "duration",
        "velocity",
        "slide",
    )

    def __init__(
        self, note_id, track_id, pitch, start, duration, velocity, slide=None
    ):
        self.id = int(note_id)
        self.track_id = int(track_id)
        self.pitch = clamp_pitch(pitch)
        self.start = max(0.0, float(start))
        self.duration = max(MIN_NOTE_DURATION, float(duration))
        self.velocity = clamp_velocity(velocity)
        self.slide = slide.copy() if slide else None

    @property
    def end(self):
        return self.start + self.duration

    def copy(self):
        return NoteModel(
            self.id,
            self.track_id,
            self.pitch,
            self.start,
            self.duration,
            self.velocity,
            self.slide,
        )

    def to_dict(self):
        return {
            "id": self.id,
            "track_id": self.track_id,
            "pitch": self.pitch,
            "start": self.start,
            "duration": self.duration,
            "velocity": self.velocity,
            "slide": self.slide.to_dict() if self.slide else None,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get("id", 0),
            data.get("track_id", 0),
            data.get("pitch", 60),
            data.get("start", 0.0),
            data.get("duration", MIN_NOTE_DURATION),
            data.get("velocity", 100),
            NoteSlide.from_dict(data.get("slide")),
        )


class ClipModel:
    __slots__ = (
        "id",
        "path",
        "name",
        "start",
        "duration",
        "gain",
        "muted",
        "solo",
        "kind",
        "color",
        "waveform",
        "source",
        "tempo_base",
        "offset",
        "length",
        "pan",
    )

    def __init__(
        self,
        clip_id,
        path,
        name,
        start=0.0,
        duration=0.0,
        gain=1.0,
        muted=False,
        solo=False,
        kind="audio",
        color=None,
        offset=0.0,
        length=None,
        pan=PAN_CENTRE,
    ):
        self.id = int(clip_id)
        self.path = str(path)
        self.name = str(name)
        self.start = max(0.0, float(start))
        self.duration = max(0.0, float(duration))
        self.gain = max(0.0, min(MAX_CLIP_GAIN, float(gain)))
        self.muted = bool(muted)
        self.solo = bool(solo)
        self.kind = str(kind)
        self.color = color or CLIP_KIND_COLORS.get(
            self.kind, CLIP_KIND_COLORS["audio"]
        )
        self.waveform = []
        self.source = None
        self.tempo_base = None
        self.offset = max(0.0, float(offset))
        whole = self.offset + self.duration
        self.length = max(whole, float(length)) if length else whole
        self.pan = clamp_pan(pan)

    @property
    def end(self):
        return self.start + self.duration

    @property
    def origin(self):
        return self.start - self.offset

    def is_trimmed(self):
        return (
            self.offset > TRIM_TOLERANCE_S
            or self.offset + self.duration < self.length - TRIM_TOLERANCE_S
        )

    def trim(self, start, end):
        earliest = max(0.0, self.origin)
        latest = self.origin + self.length
        start = min(max(float(start), earliest), latest - MIN_CLIP_SECONDS)
        end = max(min(float(end), latest), start + MIN_CLIP_SECONDS)
        self.offset = max(0.0, start - self.origin)
        self.start = start
        self.duration = end - start

    def rescale(self, length):
        ratio = float(length) / self.length if self.length > 0 else 1.0
        self.offset *= ratio
        self.duration *= ratio
        self.length = max(0.0, float(length))

    def copy(self):
        clone = ClipModel(
            self.id,
            self.path,
            self.name,
            self.start,
            self.duration,
            self.gain,
            self.muted,
            self.solo,
            self.kind,
            self.color,
            self.offset,
            self.length,
            self.pan,
        )
        clone.waveform = self.waveform
        clone.source = self.source
        clone.tempo_base = self.tempo_base
        return clone

    def to_dict(self):
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "start": self.start,
            "duration": self.duration,
            "gain": self.gain,
            "pan": self.pan,
            "muted": self.muted,
            "solo": self.solo,
            "kind": self.kind,
            "color": self.color,
            "source": list(self.source) if self.source else None,
            "tempo_base": list(self.tempo_base) if self.tempo_base else None,
            "offset": self.offset,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data):
        clip = cls._from_dict(data)
        clip.source = _clip_source(data.get("source"))
        clip.tempo_base = _tempo_base(data.get("tempo_base"))
        return clip

    @classmethod
    def _from_dict(cls, data):
        return cls(
            data.get("id", 0),
            data.get("path", ""),
            data.get("name") or clip_name_from_path(data.get("path", "")),
            data.get("start", 0.0),
            data.get("duration", 0.0),
            data.get("gain", 1.0),
            data.get("muted", False),
            data.get("solo", False),
            data.get("kind", "audio"),
            data.get("color"),
            _seconds(data.get("offset")),
            _seconds(data.get("length")) or None,
            data.get("pan", PAN_CENTRE),
        )


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(value) and value > 0 else 0.0


def _last_bend_of_each_tick(pm, bends):
    kept = {}
    for time, value in bends:
        kept[pm.time_to_tick(time)] = (time, value)
    return list(kept.values())


def _song_version(data):
    version = data.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, (int, float)):
        return 1
    return version


def _grid_offset_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return float(value)


MAX_CLIP_TEMPO_CHANGE = 0.10


class TempoNormalizeError(ValueError):
    def __init__(self, clips):
        self.clip_names = [name for name, _ in clips]
        worst = max(change for _, change in clips)
        names = ", ".join(f"'{name}'" for name in self.clip_names)
        super().__init__(
            f"Even BPM was not applied: the audio clip {names} would need "
            f"its speed changed by up to {worst:.0%}, more than the "
            f"{MAX_CLIP_TEMPO_CHANGE:.0%} that still sounds right. "
            "Turn off Even BPM to keep the tempo changes of the song, "
            "or remove that audio clip and turn Even BPM on again."
        )


def _stretch_notes(notes, factor):
    for note in notes:
        note.start *= factor
        note.duration = max(MIN_NOTE_DURATION, note.duration * factor)
        if note.slide:
            note.slide = note.slide.scaled(factor)


def _stretch_bends(tracks, factor):
    for track in tracks:
        for point in track.bends:
            point[0] *= factor


def _move_clips(clips, factor):
    for clip in clips:
        clip.start *= factor


def _stretched_normalized(normalized, factor):
    if normalized is None:
        return None
    return {
        "map": normalized["map"].stretched(factor),
        "bpm": normalized["bpm"] / factor,
        "tempo": normalized["tempo"] / factor,
    }


def _stretched_state(state, factor):
    tracks, notes, clips, tempo_map, normalized = state
    _stretch_notes(notes, factor)
    _stretch_bends(tracks.values(), factor)
    _move_clips(clips, factor)
    return (
        tracks,
        notes,
        clips,
        tempo_map.stretched(factor) if tempo_map is not None else None,
        _stretched_normalized(normalized, factor),
    )


def _tempo_base(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    path, bpm = value
    if not isinstance(path, str) or not path:
        return None
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)):
        return None
    if not math.isfinite(bpm) or bpm <= 0:
        return None
    return (path, float(bpm))


def _clip_source(value):
    try:
        path, start, duration = value
        return str(path), float(start), float(duration)
    except (TypeError, ValueError):
        return None


def _normalized_from_dict(data):
    try:
        tempo_map = TempoMap.from_pairs(data["map"])
        return {
            "map": tempo_map,
            "bpm": float(data["bpm"]),
            "tempo": float(data["tempo"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _changing_tempo_map(pm):
    try:
        found = TempoMap.from_pretty_midi(pm)
    except (AttributeError, TypeError, ValueError):
        return None
    return None if found.is_uniform else found


def _tempo_map_from_pairs(pairs):
    try:
        found = TempoMap.from_pairs(pairs or [])
    except (TypeError, ValueError):
        return None
    return None if found.is_uniform else found


class EditorModel:
    def __init__(self, tempo=120.0):
        self.tracks = {}
        self.notes = []
        self.clips = []
        self.tempo = validate_bpm(tempo)
        self.grid_offset = 0.0
        self.tempo_map = None
        self.normalized = None
        self._next_track_id = 1
        self._next_note_id = 1
        self._next_clip_id = 1
        self._undo_stack = []
        self._redo_stack = []
        self._history_changes = 0
        self.max_undo_steps = MAX_UNDO_STEPS

    def clear(self):
        self.tracks.clear()
        self.notes.clear()
        self.clips.clear()
        self.tempo_map = None
        self.normalized = None

    def _allocate_track_id(self):
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _allocate_note_id(self):
        note_id = self._next_note_id
        self._next_note_id += 1
        return note_id

    def _allocate_clip_id(self):
        clip_id = self._next_clip_id
        self._next_clip_id += 1
        return clip_id

    def next_palette_color(self, current=None):
        if current in TRACK_PALETTE:
            index = (TRACK_PALETTE.index(current) + 1) % len(TRACK_PALETTE)
            return TRACK_PALETTE[index]
        used = {track.color for track in self.tracks.values()}
        for color in TRACK_PALETTE:
            if color not in used:
                return color
        return TRACK_PALETTE[len(self.tracks) % len(TRACK_PALETTE)]

    def add_track(self, name, program=0, is_drum=False, color=None):
        track = TrackModel(
            self._allocate_track_id(),
            name or DEFAULT_TRACK_NAME,
            program,
            is_drum,
            color or self.next_palette_color(),
        )
        self.tracks[track.id] = track
        return track

    def remove_track(self, track_id):
        if track_id not in self.tracks:
            return
        del self.tracks[track_id]
        self.notes = [n for n in self.notes if n.track_id != track_id]

    def rename_track(self, track_id, name):
        track = self.tracks.get(track_id)
        if track is not None and str(name).strip():
            track.name = str(name).strip()

    def set_track_program(self, track_id, program):
        track = self.tracks.get(track_id)
        if track is not None:
            track.program = int(max(0, min(127, program)))

    def set_track_drum(self, track_id, is_drum):
        track = self.tracks.get(track_id)
        if track is not None:
            track.is_drum = bool(is_drum)

    def set_track_mute(self, track_id, muted):
        track = self.tracks.get(track_id)
        if track is not None:
            track.muted = bool(muted)

    def set_track_solo(self, track_id, solo):
        track = self.tracks.get(track_id)
        if track is not None:
            track.solo = bool(solo)

    def set_track_visible(self, track_id, visible):
        track = self.tracks.get(track_id)
        if track is not None:
            track.visible = bool(visible)

    def set_track_color(self, track_id, color):
        track = self.tracks.get(track_id)
        if track is not None and color:
            track.color = str(color)

    def first_track_id(self):
        for track_id in self.tracks:
            return track_id
        return None

    def ensure_default_track(self):
        first = self.first_track_id()
        if first is not None:
            return self.tracks[first]
        return self.add_track(DEFAULT_TRACK_NAME)

    def resolve_track_id(self, track_id):
        if track_id in self.tracks:
            return track_id
        return self.ensure_default_track().id

    def add_note(
        self, track_id, pitch, start, duration, velocity=100, slide=None
    ):
        note = NoteModel(
            self._allocate_note_id(),
            self.resolve_track_id(track_id),
            pitch,
            start,
            duration,
            velocity,
            slide,
        )
        self.notes.append(note)
        return note

    def remove_notes(self, note_ids):
        doomed = set(note_ids)
        if not doomed:
            return
        self.notes = [n for n in self.notes if n.id not in doomed]

    def remove_slides(self, note_ids):
        slid = [n for n in self._selected_notes(note_ids) if n.slide]
        for note in slid:
            note.slide = None
        return len(slid)

    def notes_by_id(self):
        return {note.id: note for note in self.notes}

    def notes_for_track(self, track_id):
        return [n for n in self.notes if n.track_id == track_id]

    def note_count(self):
        return len(self.notes)

    def _selected_notes(self, note_ids):
        wanted = set(note_ids)
        return [n for n in self.notes if n.id in wanted]

    def transpose_notes(self, note_ids, semitones):
        for note in self._selected_notes(note_ids):
            note.pitch = clamp_pitch(note.pitch + int(semitones))

    def shift_notes(self, note_ids, seconds):
        for note in self._selected_notes(note_ids):
            note.start = max(0.0, note.start + float(seconds))

    def _moved_by_beats(self, note, beats):
        tempo_map = self.tempo_map
        start = tempo_map.seconds_at(tempo_map.beats_at(note.start) + beats)
        end = tempo_map.seconds_at(tempo_map.beats_at(note.end) + beats)
        start = max(0.0, start)
        duration = max(MIN_NOTE_DURATION, end - start)
        slide = note.slide and note.slide.remapped(
            note.start,
            note.end,
            start,
            start + duration,
            lambda t: tempo_map.seconds_at(tempo_map.beats_at(t) + beats),
        )
        return start, duration, slide

    def shift_notes_by_beats(self, note_ids, beats):
        if self.tempo_map is None:
            self.shift_notes(note_ids, float(beats) * 60.0 / self.tempo)
            return
        for note in self._selected_notes(note_ids):
            note.start, note.duration, note.slide = self._moved_by_beats(
                note, float(beats)
            )

    def beats_length(self, start, beats):
        if self.tempo_map is None:
            return float(beats) * 60.0 / self.tempo
        tempo_map = self.tempo_map
        end = tempo_map.seconds_at(tempo_map.beats_at(start) + float(beats))
        return end - float(start)

    def quantize_notes(
        self, note_ids, grid_seconds, strength=1.0, grid_offset=None
    ):
        grid = float(grid_seconds)
        if grid <= 0:
            return
        amount = max(0.0, min(1.0, float(strength)))
        origin = self.grid_offset if grid_offset is None else float(grid_offset)
        tempo_map = self.tempo_map
        if tempo_map is not None:
            grid = grid * self.tempo / 60.0
            origin = tempo_map.beats_at(origin)
        for note in self._selected_notes(note_ids):
            start = note.start
            if tempo_map is not None:
                start = tempo_map.beats_at(start)
            target = round((start - origin) / grid) * grid + origin
            if tempo_map is not None:
                target = tempo_map.seconds_at(target)
            note.start = max(0.0, note.start + (target - note.start) * amount)

    def humanize_notes(self, note_ids, timing_sec, velocity_amt, seed=None):
        rng = random.Random(seed)
        timing = abs(float(timing_sec))
        vel_amt = abs(int(velocity_amt))
        for note in self._selected_notes(note_ids):
            if timing > 0:
                note.start = max(0.0, note.start + rng.uniform(-timing, timing))
            if vel_amt > 0:
                note.velocity = clamp_velocity(
                    note.velocity + rng.randint(-vel_amt, vel_amt)
                )

    def set_notes_velocity(self, note_ids, velocity):
        clamped = clamp_velocity(velocity)
        for note in self._selected_notes(note_ids):
            note.velocity = clamped

    def adjust_notes_velocity(self, note_ids, delta):
        for note in self._selected_notes(note_ids):
            note.velocity = clamp_velocity(note.velocity + int(delta))

    def scale_note_times(self, ratio):
        factor = float(ratio)
        if factor <= 0 or abs(factor - 1.0) < 1e-9:
            return
        if self.tempo_map is not None:
            self.tempo_map = self.tempo_map.stretched(factor)
        self.normalized = _stretched_normalized(self.normalized, factor)
        _stretch_notes(self.notes, factor)
        _stretch_bends(self.tracks.values(), factor)

    def follow_tempo(self, ratio):
        factor = float(ratio)
        if factor <= 0 or abs(factor - 1.0) < 1e-9:
            return
        self.scale_note_times(factor)
        _move_clips(self.clips, factor)
        self._undo_stack = [
            _stretched_state(state, factor) for state in self._undo_stack
        ]
        self._redo_stack = [
            _stretched_state(state, factor) for state in self._redo_stack
        ]

    def copy_notes(self, note_ids):
        selected = self._selected_notes(note_ids)
        if not selected:
            return []
        origin = min(n.start for n in selected)
        return [
            {
                "track_id": n.track_id,
                "pitch": n.pitch,
                "offset": n.start - origin,
                "duration": n.duration,
                "velocity": n.velocity,
                "slide": n.slide.to_dict() if n.slide else None,
            }
            for n in sorted(selected, key=lambda n: (n.start, n.pitch))
        ]

    def paste_notes(self, clipboard, at_time, track_id=None):
        created = []
        base = max(0.0, float(at_time))
        for entry in clipboard or []:
            target = entry.get("track_id")
            if target not in self.tracks:
                target = track_id
            created.append(
                self.add_note(
                    self.resolve_track_id(target),
                    entry.get("pitch", 60),
                    base + float(entry.get("offset", 0.0)),
                    entry.get("duration", MIN_NOTE_DURATION),
                    entry.get("velocity", 100),
                    NoteSlide.from_dict(entry.get("slide")),
                )
            )
        return created

    def duplicate_notes(self, note_ids, beat_seconds=None):
        selected = self._selected_notes(note_ids)
        if not selected:
            return []
        span_start = min(n.start for n in selected)
        span_end = max(n.end for n in selected)
        if self.tempo_map is not None and beat_seconds and beat_seconds > 0:
            beats = self.tempo_map.beats_at(span_end) - self.tempo_map.beats_at(
                span_start
            )
            beats = max(1, math.ceil(beats - 1e-9))
            created = []
            for n in selected:
                start, duration, slide = self._moved_by_beats(n, beats)
                created.append(
                    self.add_note(
                        n.track_id, n.pitch, start, duration, n.velocity, slide
                    )
                )
            return created
        offset = span_end - span_start
        if beat_seconds and beat_seconds > 0:
            offset = math.ceil(offset / beat_seconds - 1e-9) * beat_seconds
        offset = max(offset, MIN_NOTE_DURATION)
        return [
            self.add_note(
                n.track_id,
                n.pitch,
                n.start + offset,
                n.duration,
                n.velocity,
                n.slide,
            )
            for n in selected
        ]

    def add_clip(
        self,
        path,
        start=0.0,
        name=None,
        kind="audio",
        gain=1.0,
        duration=None,
        pan=PAN_CENTRE,
    ):
        if duration is None:
            duration = probe_audio_duration(path)
        clip = ClipModel(
            self._allocate_clip_id(),
            path,
            name or clip_name_from_path(path),
            start,
            duration,
            gain,
            kind=kind,
            pan=pan,
        )
        self.clips.append(clip)
        return clip

    def clip_by_id(self, clip_id):
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None

    def move_clip(self, clip_id, start):
        clip = self.clip_by_id(clip_id)
        if clip is not None:
            clip.start = max(0.0, float(start))

    def trim_clip(self, clip_id, start, end):
        clip = self.clip_by_id(clip_id)
        if clip is None:
            return False
        before = (clip.start, clip.duration, clip.offset)
        clip.trim(start, end)
        return (clip.start, clip.duration, clip.offset) != before

    def remove_clip(self, clip_id):
        self.clips = [c for c in self.clips if c.id != clip_id]

    def set_track_gain(self, track_id, gain):
        track = self.tracks.get(track_id)
        if track is not None:
            track.gain = clamp_track_gain(gain)

    def set_track_pan(self, track_id, pan):
        track = self.tracks.get(track_id)
        if track is not None:
            track.pan = clamp_pan(pan)

    def set_clip_pan(self, clip_id, pan):
        clip = self.clip_by_id(clip_id)
        if clip is not None:
            clip.pan = clamp_pan(pan)

    def set_clip_mute(self, clip_id, muted):
        clip = self.clip_by_id(clip_id)
        if clip is not None:
            clip.muted = bool(muted)

    def set_clip_solo(self, clip_id, solo):
        clip = self.clip_by_id(clip_id)
        if clip is not None:
            clip.solo = bool(solo)

    def set_clip_gain(self, clip_id, gain):
        clip = self.clip_by_id(clip_id)
        if clip is not None:
            clip.gain = max(0.0, min(MAX_CLIP_GAIN, float(gain)))

    def set_clip_waveform(self, clip_id, waveform, duration=None):
        clip = self.clip_by_id(clip_id)
        if clip is None:
            return
        clip.waveform = list(waveform or [])
        if duration is not None and duration > 0:
            whole = float(duration)
            if clip.is_trimmed():
                clip.length = whole
                clip.trim(clip.start, clip.end)
            else:
                clip.duration = clip.length = whole

    def _any_solo(self):
        return any(t.solo for t in self.tracks.values()) or any(
            c.solo for c in self.clips
        )

    def active_tracks(self):
        any_solo = self._any_solo()
        return [
            t
            for t in self.tracks.values()
            if not t.muted and (t.solo or not any_solo)
        ]

    def active_clips(self):
        any_solo = self._any_solo()
        return [
            c.to_dict()
            for c in self.clips
            if not c.muted and (c.solo or not any_solo) and c.path
        ]

    def track_for_instrument(self, program, is_drum, joining=None):
        for track in self.tracks.values():
            if (
                track.visible
                and not track.muted
                and track.is_drum == bool(is_drum)
                and (is_drum or track.program == int(program))
                and (joining is None or self._bends_allow(track, *joining))
            ):
                return track
        return None

    def track_bends(self, track_id):
        track = self.tracks.get(track_id)
        if track is None:
            return [], DEFAULT_BEND_RANGE
        return played_bends(
            self.notes_for_track(track_id), track.bends, track.bend_range
        )

    def _bends_allow(self, track, spans, bends, bend_range):
        played, _wide = self.track_bends(track.id)
        if not bends and not played:
            return True
        if bends and played and bend_range != track.bend_range:
            return False
        return not bends_clash(
            [(n.start, n.end) for n in self.notes_for_track(track.id)],
            played,
            spans,
            bends,
        )

    def load_midi(
        self, pm, replace=True, name_prefix=None, merge_instruments=False
    ):
        if replace:
            self.tracks.clear()
            self.notes.clear()
            self.tempo_map = None
            self.normalized = None
        convert = self._adopt_tempo_changes(pm)
        created = []
        instruments = list(getattr(pm, "instruments", None) or [])
        for inst in instruments:
            spans = [(convert(n.start), convert(n.end)) for n in inst.notes]
            bends = [
                [max(0.0, convert(b.time)), clamp_bend(b.pitch)]
                for b in getattr(inst, "pitch_bends", None) or ()
            ]
            bend_range = bend_range_of(inst) if bends else DEFAULT_BEND_RANGE
            track = (
                self.track_for_instrument(
                    inst.program, inst.is_drum, (spans, bends, bend_range)
                )
                if merge_instruments
                else None
            )
            if track is None:
                label = sanitize_instrument_name(getattr(inst, "name", ""))
                if not label or is_general_midi_name(label):
                    label = instrument_name(inst.program, inst.is_drum)
                if name_prefix:
                    label = f"{name_prefix} {label}"
                track = self.add_track(
                    label,
                    program=int(inst.program),
                    is_drum=bool(inst.is_drum),
                )
                track.pan = _instrument_pan(inst)
            bends.sort(key=lambda point: point[0])
            slides, free = attach_slides(spans, bends, bend_range)
            for index, (note, (start, end)) in enumerate(
                zip(inst.notes, spans, strict=True)
            ):
                self.add_note(
                    track.id,
                    note.pitch,
                    start,
                    end - start,
                    note.velocity,
                    slides.get(index),
                )
            if bends:
                track.bends = sorted(
                    track.bends + free, key=lambda point: point[0]
                )
                track.bend_range = bend_range
            if track not in created:
                created.append(track)
        return created

    def _adopt_tempo_changes(self, pm):
        if _changing_tempo_map(pm) is None:
            return float
        state = self.normalized
        if state is not None:
            return lambda t: float(state["map"].warp(t, state["bpm"]))
        if self.tempo_map is None:
            self.tempo_map = _changing_tempo_map(pm)
        return float

    def to_pretty_midi(self, tempo=None, respect_mute_solo=True):
        bpm = validate_bpm(tempo, default=self.tempo) if tempo else self.tempo
        if self.tempo_map is not None and abs(bpm - self.tempo) < 1e-6:
            pm = self.tempo_map.new_pretty_midi()
        else:
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        tracks = (
            self.active_tracks()
            if respect_mute_solo
            else list(self.tracks.values())
        )
        grouped = {}
        for note in self.notes:
            grouped.setdefault(note.track_id, []).append(note)
        for track in tracks:
            notes = grouped.get(track.id)
            if not notes:
                continue
            inst = pretty_midi.Instrument(
                program=track.program, is_drum=track.is_drum, name=track.name
            )
            level = track.gain * rhythm_boost(track.program, track.is_drum)
            for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
                end = max(note.start + MIN_NOTE_DURATION, note.end)
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=clamp_velocity(
                            max(1, round(note.velocity * level))
                        ),
                        pitch=note.pitch,
                        start=note.start,
                        end=end,
                    )
                )
            bends, bend_range = played_bends(
                notes, track.bends, track.bend_range
            )
            if bends:
                inst.pitch_bends = [
                    pretty_midi.PitchBend(value, time)
                    for time, value in _last_bend_of_each_tick(pm, bends)
                ]
                if bend_range != DEFAULT_BEND_RANGE:
                    set_bend_range(inst, bend_range)
            if track.pan != PAN_CENTRE:
                inst.control_changes.append(
                    pretty_midi.ControlChange(
                        PAN_CONTROLLER, pan_to_controller(track.pan), 0.0
                    )
                )
            pm.instruments.append(inst)
        pm.instruments = _merged_keeping_sides(pm.instruments)
        return pm

    def _retime(self, convert):
        for note in self.notes:
            start = max(0.0, float(convert(note.start)))
            duration = max(MIN_NOTE_DURATION, float(convert(note.end)) - start)
            if note.slide:
                note.slide = note.slide.remapped(
                    note.start, note.end, start, start + duration, convert
                )
            note.start = start
            note.duration = duration
        for track in self.tracks.values():
            for point in track.bends:
                point[0] = max(0.0, float(convert(point[0])))
        self.grid_offset = max(0.0, float(convert(self.grid_offset)))
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._history_changes += 1

    def normalize_tempo(self, align_clip):
        tempo_map = self.tempo_map
        if tempo_map is None:
            return False
        end = max(self.end_time(), tempo_map.starts[-1])
        bpm = round(tempo_map.average_bpm(end), BPM_DECIMALS)
        changes = [
            (clip, tempo_map.largest_change(clip.start, clip.end, bpm))
            for clip in self.clips
        ]
        unfit = [
            (clip.name, change)
            for clip, change in changes
            if change > MAX_CLIP_TEMPO_CHANGE
        ]
        if unfit:
            raise TempoNormalizeError(unfit)
        aligned = [
            (clip, *align_clip(clip, tempo_map, bpm)) for clip in self.clips
        ]
        for clip, path, duration in aligned:
            self._adopt_aligned(clip, path, duration, tempo_map, bpm)
        self._retime(lambda t: tempo_map.warp(t, bpm))
        self.normalized = {"map": tempo_map, "bpm": bpm, "tempo": self.tempo}
        self.tempo_map = None
        self.tempo = validate_bpm(bpm)
        return True

    def normalize_clip(self, clip, align_clip):
        state = self.normalized
        if state is None or clip.source is not None:
            return False
        tempo_map, bpm = state["map"], state["bpm"]
        change = tempo_map.largest_change(clip.start, clip.end, bpm)
        if change > MAX_CLIP_TEMPO_CHANGE:
            raise TempoNormalizeError([(clip.name, change)])
        path, duration = align_clip(clip, tempo_map, bpm)
        self._adopt_aligned(clip, path, duration, tempo_map, bpm)
        return True

    @staticmethod
    def _adopt_aligned(clip, path, length, tempo_map, bpm):
        clip.source = (clip.path, clip.start, clip.length)
        clip.path = str(path)
        clip.start = max(0.0, float(tempo_map.warp(clip.start, bpm)))
        clip.rescale(length)
        clip.waveform = []

    def restore_tempo(self):
        state = self.normalized
        if state is None:
            return False
        tempo_map, bpm = state["map"], state["bpm"]
        for clip in self.clips:
            clip.start = max(0.0, float(tempo_map.unwarp(clip.start, bpm)))
            if clip.source is not None:
                clip.path, _, length = clip.source
                clip.rescale(length)
                clip.source = None
                clip.waveform = []
        self._retime(lambda t: tempo_map.unwarp(t, bpm))
        self.tempo_map = tempo_map
        self.tempo = validate_bpm(state["tempo"])
        self.normalized = None
        return True

    def to_editor_time(self, seconds):
        if self.normalized is None:
            return float(seconds)
        return float(
            self.normalized["map"].warp(seconds, self.normalized["bpm"])
        )

    def end_time(self):
        latest = 0.0
        for note in self.notes:
            if note.end > latest:
                latest = note.end
        for clip in self.clips:
            if clip.end > latest:
                latest = clip.end
        return latest

    def silent_notes(self, range_for):
        counts = {}
        for note in self.notes:
            track = self.tracks.get(note.track_id)
            if track is None:
                continue
            low, high = range_for(track.program, track.is_drum)
            if not low <= note.pitch <= high:
                counts[track.id] = counts.get(track.id, 0) + 1
        return counts

    def summary_text(self):
        return (
            f"{len(self.tracks)} tracks · {self.note_count()} notes · "
            f"{len(self.clips)} clips · end {format_clock(self.end_time())}"
        )

    def _capture(self):
        return (
            {tid: t.copy() for tid, t in self.tracks.items()},
            [n.copy() for n in self.notes],
            [c.copy() for c in self.clips],
            self.tempo_map,
            self.normalized,
        )

    def _restore(self, state):
        tracks, notes, clips, self.tempo_map, self.normalized = state
        self.tracks = {tid: t.copy() for tid, t in tracks.items()}
        self.notes = [n.copy() for n in notes]
        self.clips = [c.copy() for c in clips]

    def clip_files_in_use(self):
        used = set()
        history = [state[2] for state in self._undo_stack + self._redo_stack]
        for clips in [self.clips, *history]:
            for clip in clips:
                used.add(clip.path)
                if clip.source is not None:
                    used.add(clip.source[0])
                if clip.tempo_base is not None:
                    used.add(clip.tempo_base[0])
        return used

    def clips_to_follow_tempo(self, earlier_bpm):
        if self.tempo_map is not None or self.normalized is not None:
            return []
        waiting = []
        for clip in self.clips:
            if clip.source is not None:
                continue
            if clip.tempo_base is None:
                recorded = (clip.path, float(earlier_bpm))
                for earlier in self._earlier_copies(clip.id):
                    if earlier.path == clip.path and earlier.tempo_base is None:
                        earlier.tempo_base = recorded
                clip.tempo_base = recorded
            waiting.append(clip)
        return waiting

    def _earlier_copies(self, clip_id):
        return [
            clip
            for state in self._undo_stack + self._redo_stack
            for clip in state[2]
            if clip.id == clip_id
        ]

    def retune_clip(self, clip_id, path, length):
        current = self.clip_by_id(clip_id)
        if current is None or current.tempo_base is None:
            return False
        for clip in [current, *self._earlier_copies(clip_id)]:
            if clip.tempo_base == current.tempo_base:
                clip.path = str(path)
                clip.rescale(length)
                clip.waveform = []
        return True

    def snapshot(self):
        self._undo_stack.append(self._capture())
        overflow = len(self._undo_stack) - self.max_undo_steps
        if overflow > 0:
            del self._undo_stack[:overflow]
        self._redo_stack.clear()
        self._history_changes += 1

    def history_mark(self):
        return self._history_changes

    def discard_last_snapshot(self):
        if self._undo_stack:
            self._undo_stack.pop()
            self._history_changes += 1

    def can_undo(self):
        return bool(self._undo_stack)

    def can_redo(self):
        return bool(self._redo_stack)

    def undo(self):
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture())
        self._restore(self._undo_stack.pop())
        self._history_changes += 1
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture())
        self._restore(self._redo_stack.pop())
        self._history_changes += 1
        return True

    def to_dict(self):
        return {
            "version": SONG_VERSION,
            "tempo": self.tempo,
            "grid_offset": self.grid_offset,
            "tempo_map": self.tempo_map.as_pairs() if self.tempo_map else None,
            "normalized": {
                "map": self.normalized["map"].as_pairs(),
                "bpm": self.normalized["bpm"],
                "tempo": self.normalized["tempo"],
            }
            if self.normalized
            else None,
            "tracks": [t.to_dict() for t in self.tracks.values()],
            "notes": [n.to_dict() for n in self.notes],
            "clips": [c.to_dict() for c in self.clips],
        }

    def load_state(self, data):
        data = data or {}
        self.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._history_changes += 1
        self.tempo = validate_bpm(data.get("tempo", self.tempo))
        self.grid_offset = _grid_offset_value(data.get("grid_offset", 0.0))
        self.tempo_map = _tempo_map_from_pairs(data.get("tempo_map"))
        self.normalized = _normalized_from_dict(data.get("normalized"))
        for entry in data.get("tracks", []) or []:
            track = TrackModel.from_dict(entry)
            self.tracks[track.id] = track
        for entry in data.get("notes", []) or []:
            note = NoteModel.from_dict(entry)
            if note.track_id in self.tracks:
                self.notes.append(note)
        if _song_version(data) < SLIDES_ON_NOTES_SINCE:
            self._attach_track_bends()
        for entry in data.get("clips", []) or []:
            if entry.get("path"):
                self.clips.append(ClipModel.from_dict(entry))
        self._next_track_id = max(self.tracks, default=0) + 1
        self._next_note_id = max((n.id for n in self.notes), default=0) + 1
        self._next_clip_id = max((c.id for c in self.clips), default=0) + 1
        return self

    def _attach_track_bends(self):
        for track in self.tracks.values():
            if not track.bends:
                continue
            notes = [n for n in self.notes_for_track(track.id) if not n.slide]
            slides, track.bends = attach_slides(
                [(n.start, n.end) for n in notes], track.bends, track.bend_range
            )
            for index, slide in slides.items():
                notes[index].slide = slide

    @classmethod
    def from_dict(cls, data):
        return cls().load_state(data)
