import logging
import math
import os
import random

import numpy as np
import pretty_midi

from korshmer.constants import NOTE_NAMES
from korshmer.utils.instrument_names import sanitize_instrument_name
from korshmer.utils.tempo import validate_bpm
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
MAX_UNDO_STEPS = 100
MAX_CLIP_GAIN = 4.0
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
    ):
        self.id = int(track_id)
        self.name = str(name)
        self.program = int(max(0, min(127, program)))
        self.is_drum = bool(is_drum)
        self.color = str(color)
        self.muted = bool(muted)
        self.solo = bool(solo)
        self.visible = bool(visible)

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
        )

    def to_dict(self):
        return {
            "id": self.id,
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
        )


class NoteModel:
    __slots__ = ("id", "track_id", "pitch", "start", "duration", "velocity")

    def __init__(self, note_id, track_id, pitch, start, duration, velocity):
        self.id = int(note_id)
        self.track_id = int(track_id)
        self.pitch = clamp_pitch(pitch)
        self.start = max(0.0, float(start))
        self.duration = max(MIN_NOTE_DURATION, float(duration))
        self.velocity = clamp_velocity(velocity)

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
        )

    def to_dict(self):
        return {
            "id": self.id,
            "track_id": self.track_id,
            "pitch": self.pitch,
            "start": self.start,
            "duration": self.duration,
            "velocity": self.velocity,
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

    @property
    def end(self):
        return self.start + self.duration

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
        )
        clone.waveform = self.waveform
        return clone

    def to_dict(self):
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "start": self.start,
            "duration": self.duration,
            "gain": self.gain,
            "muted": self.muted,
            "solo": self.solo,
            "kind": self.kind,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data):
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
        )


def _grid_offset_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return float(value)


class EditorModel:
    def __init__(self, tempo=120.0):
        self.tracks = {}
        self.notes = []
        self.clips = []
        self.tempo = validate_bpm(tempo)
        self.grid_offset = 0.0
        self._next_track_id = 1
        self._next_note_id = 1
        self._next_clip_id = 1
        self._undo_stack = []
        self._redo_stack = []
        self.max_undo_steps = MAX_UNDO_STEPS

    def clear(self):
        self.tracks.clear()
        self.notes.clear()
        self.clips.clear()

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

    def add_note(self, track_id, pitch, start, duration, velocity=100):
        note = NoteModel(
            self._allocate_note_id(),
            self.resolve_track_id(track_id),
            pitch,
            start,
            duration,
            velocity,
        )
        self.notes.append(note)
        return note

    def remove_notes(self, note_ids):
        doomed = set(note_ids)
        if not doomed:
            return
        self.notes = [n for n in self.notes if n.id not in doomed]

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

    def quantize_notes(
        self, note_ids, grid_seconds, strength=1.0, grid_offset=None
    ):
        grid = float(grid_seconds)
        if grid <= 0:
            return
        amount = max(0.0, min(1.0, float(strength)))
        origin = self.grid_offset if grid_offset is None else float(grid_offset)
        for note in self._selected_notes(note_ids):
            target = round((note.start - origin) / grid) * grid + origin
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
        for note in self.notes:
            note.start *= factor
            note.duration = max(MIN_NOTE_DURATION, note.duration * factor)

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
                )
            )
        return created

    def duplicate_notes(self, note_ids, beat_seconds=None):
        selected = self._selected_notes(note_ids)
        if not selected:
            return []
        span_start = min(n.start for n in selected)
        span_end = max(n.end for n in selected)
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

    def remove_clip(self, clip_id):
        self.clips = [c for c in self.clips if c.id != clip_id]

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
            clip.duration = float(duration)

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

    def track_for_instrument(self, program, is_drum):
        for track in self.tracks.values():
            if (
                track.visible
                and not track.muted
                and track.is_drum == bool(is_drum)
                and (is_drum or track.program == int(program))
            ):
                return track
        return None

    def load_midi(
        self, pm, replace=True, name_prefix=None, merge_instruments=False
    ):
        if replace:
            self.tracks.clear()
            self.notes.clear()
        created = []
        instruments = list(getattr(pm, "instruments", None) or [])
        for inst in instruments:
            track = (
                self.track_for_instrument(inst.program, inst.is_drum)
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
            for note in inst.notes:
                self.add_note(
                    track.id,
                    note.pitch,
                    note.start,
                    note.end - note.start,
                    note.velocity,
                )
            if track not in created:
                created.append(track)
        return created

    def to_pretty_midi(self, tempo=None, respect_mute_solo=True):
        bpm = validate_bpm(tempo, default=self.tempo) if tempo else self.tempo
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
            for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
                end = max(note.start + MIN_NOTE_DURATION, note.end)
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=end,
                    )
                )
            pm.instruments.append(inst)
        pm.instruments = merge_by_instrument(
            pm.instruments, keep_shared_names=True
        )
        return pm

    def end_time(self):
        latest = 0.0
        for note in self.notes:
            if note.end > latest:
                latest = note.end
        for clip in self.clips:
            if clip.end > latest:
                latest = clip.end
        return latest

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
        )

    def _restore(self, state):
        tracks, notes, clips = state
        self.tracks = {tid: t.copy() for tid, t in tracks.items()}
        self.notes = [n.copy() for n in notes]
        self.clips = [c.copy() for c in clips]

    def snapshot(self):
        self._undo_stack.append(self._capture())
        overflow = len(self._undo_stack) - self.max_undo_steps
        if overflow > 0:
            del self._undo_stack[:overflow]
        self._redo_stack.clear()

    def discard_last_snapshot(self):
        if self._undo_stack:
            self._undo_stack.pop()

    def can_undo(self):
        return bool(self._undo_stack)

    def can_redo(self):
        return bool(self._redo_stack)

    def undo(self):
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture())
        self._restore(self._undo_stack.pop())
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture())
        self._restore(self._redo_stack.pop())
        return True

    def to_dict(self):
        return {
            "version": 1,
            "tempo": self.tempo,
            "grid_offset": self.grid_offset,
            "tracks": [t.to_dict() for t in self.tracks.values()],
            "notes": [n.to_dict() for n in self.notes],
            "clips": [c.to_dict() for c in self.clips],
        }

    def load_state(self, data):
        data = data or {}
        self.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.tempo = validate_bpm(data.get("tempo", self.tempo))
        self.grid_offset = _grid_offset_value(data.get("grid_offset", 0.0))
        for entry in data.get("tracks", []) or []:
            track = TrackModel.from_dict(entry)
            self.tracks[track.id] = track
        for entry in data.get("notes", []) or []:
            note = NoteModel.from_dict(entry)
            if note.track_id in self.tracks:
                self.notes.append(note)
        for entry in data.get("clips", []) or []:
            if entry.get("path"):
                self.clips.append(ClipModel.from_dict(entry))
        self._next_track_id = max(self.tracks, default=0) + 1
        self._next_note_id = max((n.id for n in self.notes), default=0) + 1
        self._next_clip_id = max((c.id for c in self.clips), default=0) + 1
        return self

    @classmethod
    def from_dict(cls, data):
        return cls().load_state(data)
