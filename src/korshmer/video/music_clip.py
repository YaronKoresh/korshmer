import bisect
import colorsys
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pretty_midi
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from korshmer.midi.bar_lines import bar_lines, beat_lines
from korshmer.midi_processing.chord_detection import ChordDetector
from korshmer.utils.tempo import midi_file_tempo
from korshmer.utils.track_layout import instrument_name

BASE_WIDTH = 1280
BASE_HEIGHT = 720
RESOLUTIONS = {"4K": 3.0, "1080p": 1.5, "720p": 1.0}
DEFAULT_RESOLUTION = "4K"
DEFAULT_FPS = 24
MIN_FPS = 5
MAX_FPS = 60

LOOKAHEAD_PIXELS_PER_SECOND = 100.0
DEFAULT_TEMPO = 120.0

BACKGROUND = (15, 12, 41)
PANEL = (26, 22, 62)
TEXT = (230, 240, 255)
MUTED = (140, 150, 190)
ACCENT = (0, 242, 255)
WARNING = (255, 190, 60)
WAVE_DIM = (50, 80, 130)
WAVE_BRIGHT = (0, 242, 255)
GRID = (44, 40, 96)
BAR_GRID = (80, 76, 150)
WHITE_KEY = (232, 236, 245)
BLACK_KEY = (30, 28, 52)
DRUM_COLOR = (255, 255, 255)
BLACK_PITCH_CLASSES = (1, 3, 6, 8, 10)

TITLE_Y = 14
ROLL_BOX = (20, 76, 900, 496)
KEYS_BOX = (20, 496, 900, 556)
WAVE_BOX = (20, 590, 900, 646)
NOW_BOX = (20, 654, 900, 710)
INFO_BOX = (920, 76, 1260, 710)
DRUM_LANE_WIDTH = 44
INSTRUMENT_ROW = 24
INSTRUMENT_LIST_BOTTOM = 392
MAX_NOTE_NAMES = 7

_chord_detector = ChordDetector()


class ClipExportError(Exception):
    pass


class ClipExportCancelled(ClipExportError):
    pass


def track_color(index, total):
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.55, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


@dataclass
class TrackInfo:
    name: str
    instrument: str
    is_drum: bool
    note_count: int
    lowest: str
    highest: str
    color: tuple


@dataclass
class TechSpecs:
    title: str
    duration: float
    tempo: float
    tempo_label: str
    time_signature: str
    key: str
    track_count: int
    listed_tracks: int
    note_count: int
    pitch_range: str
    max_polyphony: int
    sample_rate: int
    bit_depth: int
    loudness_lufs: float | None
    true_peak_dbtp: float | None
    raw_loudness_lufs: float | None = None
    mastered: bool = True
    tracks: list = field(default_factory=list)

    def lines(self):
        minutes, seconds = divmod(int(round(self.duration)), 60)
        tracks = str(self.track_count)
        if self.listed_tracks < self.track_count:
            tracks += f" ({self.listed_tracks} listed)"
        lines = [
            ("Duration", f"{minutes}:{seconds:02d}"),
            ("Tempo", self.tempo_label),
            ("Time signature", self.time_signature),
            ("Key", self.key),
            ("Tracks", tracks),
            ("Notes", str(self.note_count)),
            ("Pitch range", self.pitch_range),
            ("Max polyphony", str(self.max_polyphony)),
            ("Audio", f"{self.sample_rate} Hz / {self.bit_depth} bit"),
        ]
        if self.mastered and self.raw_loudness_lufs is not None:
            lines.append(("LUFS before", _lufs_text(self.raw_loudness_lufs)))
        lines.append(
            (
                "LUFS mastered" if self.mastered else "LUFS",
                _lufs_text(self.loudness_lufs),
            )
        )
        peak = (
            "n/a"
            if self.true_peak_dbtp is None
            else f"{self.true_peak_dbtp:.1f} dBTP"
        )
        lines.append(("True peak", peak))
        return lines


def _lufs_text(value):
    return "n/a" if value is None else f"{value:.1f} LUFS"


def _pitch_name(pitch):
    return pretty_midi.note_number_to_name(int(pitch))


def _max_polyphony(notes):
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _midi_key(midi_path):
    try:
        from korshmer.analysis import detect_midi_key

        root, mode = detect_midi_key(midi_path)
        return f"{root} {mode}"
    except Exception:
        logging.debug("Key detection failed for the clip", exc_info=True)
        return "Unknown"


def tempo_label(pm, midi_path=None):
    times, tempos = pm.get_tempo_changes()
    rounded = [round(float(t), 1) for t in tempos]
    if not rounded:
        return DEFAULT_TEMPO, f"{DEFAULT_TEMPO:g} BPM (default)"
    declared = midi_file_tempo(midi_path) if midi_path else rounded[0]
    first = rounded[0]
    if declared is None:
        return first, f"{first:g} BPM (default)"
    if len(set(rounded)) == 1:
        return first, f"{first:g} BPM"
    return first, (
        f"{min(rounded):g}-{max(rounded):g} BPM ({len(rounded) - 1} changes)"
    )


def time_signature_label(pm):
    changes = pm.time_signature_changes
    if not changes:
        return "4/4"
    seen = []
    for change in changes:
        text = f"{change.numerator}/{change.denominator}"
        if text not in seen:
            seen.append(text)
    if len(seen) == 1:
        return seen[0]
    return f"{seen[0]} (+{len(seen) - 1} more)"


def measure_loudness(y, sr):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.size == 0 or not np.any(y):
        return None, None
    lufs = None
    try:
        import pyloudnorm

        value = pyloudnorm.Meter(int(sr)).integrated_loudness(y)
        if math.isfinite(value):
            lufs = float(value)
    except ValueError:
        logging.debug("Audio too short for LUFS", exc_info=True)
    from scipy.signal import resample_poly

    peak = float(np.max(np.abs(resample_poly(y, 4, 1, axis=0))))
    true_peak = 20 * math.log10(peak) if peak > 0 else None
    return lufs, true_peak


def collect_tech_specs(
    pm, midi_path, audio_path, title=None, raw_audio_path=None, mastered=True
):
    y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    info = sf.info(audio_path)
    bit_depth = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}.get(
        info.subtype, 16
    )
    lufs, true_peak = measure_loudness(y, sr)
    raw_lufs = None
    if raw_audio_path:
        raw, raw_sr = sf.read(raw_audio_path, dtype="float32", always_2d=True)
        raw_lufs, _ = measure_loudness(raw, raw_sr)

    sources = [inst for inst in pm.instruments if inst.notes]
    pitched = [n for inst in sources if not inst.is_drum for n in inst.notes]
    tracks = []
    for index, inst in enumerate(sources):
        pitches = [n.pitch for n in inst.notes]
        tracks.append(
            TrackInfo(
                name=inst.name or f"Track {index + 1}",
                instrument=instrument_name(inst.program, inst.is_drum),
                is_drum=inst.is_drum,
                note_count=len(inst.notes),
                lowest=_pitch_name(min(pitches)),
                highest=_pitch_name(max(pitches)),
                color=DRUM_COLOR
                if inst.is_drum
                else track_color(index, len(sources)),
            )
        )

    tempo, tempo_text = tempo_label(pm, midi_path)
    if pitched:
        low = min(n.pitch for n in pitched)
        high = max(n.pitch for n in pitched)
        pitch_range = f"{_pitch_name(low)} - {_pitch_name(high)}"
    else:
        pitch_range = "Drums only"

    return TechSpecs(
        title=title or os.path.splitext(os.path.basename(midi_path))[0],
        duration=len(y) / float(sr),
        tempo=tempo,
        tempo_label=tempo_text,
        time_signature=time_signature_label(pm),
        key=_midi_key(midi_path) if pitched else "None",
        track_count=len(tracks),
        listed_tracks=min(len(tracks), listed_track_capacity()),
        note_count=sum(t.note_count for t in tracks),
        pitch_range=pitch_range,
        max_polyphony=_max_polyphony(pitched),
        sample_rate=int(sr),
        bit_depth=bit_depth,
        loudness_lufs=lufs,
        true_peak_dbtp=true_peak,
        raw_loudness_lufs=raw_lufs,
        mastered=mastered,
        tracks=tracks,
    )


def listed_track_capacity():
    first_row = INFO_BOX[1] + 42
    return max((INSTRUMENT_LIST_BOTTOM - first_row) // INSTRUMENT_ROW, 1)


class ClipScene:
    def __init__(self, pm, specs, audio_path, scale=1.0):
        y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        self.k = float(scale)
        self.width = int(round(BASE_WIDTH * self.k))
        self.height = int(round(BASE_HEIGHT * self.k))
        self.duration = max(len(y) / float(sr), 0.1)
        self.specs = specs
        self.f_title = self._font(30)
        self.f_head = self._font(20)
        self.f_text = self._font(17)
        self.f_big = self._font(34)
        self.f_small = self._font(11)

        self._collect_notes(pm, specs)
        self._roll = self._box(ROLL_BOX)
        self._keys = self._box(KEYS_BOX)
        self._pitch_x0 = self._roll[0] + self._s(DRUM_LANE_WIDTH + 6)
        self._pitch_w = self._roll[2] - self._pitch_x0
        self._cols = self._high - self._low + 1
        self._col_w = self._pitch_w / self._cols
        self._pps = LOOKAHEAD_PIXELS_PER_SECOND * self.k
        self._chords = self._build_chords(pm)
        self._chord_times = [t for t, _ in self._chords]
        beats = beat_lines(pm)
        downbeats = bar_lines(pm)
        self._beats = beats[beats <= self.duration + 5]
        self._downbeats = downbeats[downbeats <= self.duration + 5]
        self._wave_dim, self._wave_bright = self._build_waveform(y)
        self._background = self._build_background()

    def _s(self, value):
        return int(round(value * self.k))

    def _font(self, size):
        return ImageFont.load_default(size=self._s(size))

    def _box(self, box):
        return tuple(self._s(v) for v in box)

    def _collect_notes(self, pm, specs):
        starts, ends, pitches, drums, tracks = [], [], [], [], []
        sources = [inst for inst in pm.instruments if inst.notes]
        self._track_notes = []
        for index, inst in enumerate(sources):
            self._track_notes.append(
                (
                    np.array([n.start for n in inst.notes]),
                    np.array([n.end for n in inst.notes]),
                )
            )
            for n in inst.notes:
                starts.append(n.start)
                ends.append(n.end)
                pitches.append(n.pitch)
                drums.append(inst.is_drum)
                tracks.append(index)
        order = np.argsort(starts, kind="stable") if starts else []
        self._starts = np.array(starts, dtype=np.float64)[order]
        self._ends = np.array(ends, dtype=np.float64)[order]
        self._pitches = np.array(pitches, dtype=np.int32)[order]
        self._drum = np.array(drums, dtype=bool)[order]
        self._track_of = np.array(tracks, dtype=np.int32)[order]
        self._colors = [specs.tracks[i].color for i in self._track_of]
        pitched = self._pitches[~self._drum] if len(self._pitches) else []
        if len(pitched):
            self._low = max(int(np.min(pitched)) - 2, 0)
            self._high = min(int(np.max(pitched)) + 2, 127)
        else:
            self._low, self._high = 48, 72

    def _build_chords(self, pm):
        pitched = [
            (n.start, n.end, n.pitch)
            for inst in pm.instruments
            if not inst.is_drum
            for n in inst.notes
        ]
        times = sorted({round(s, 3) for s, _, _ in pitched})
        chords = []
        last = None
        for t in times:
            sounding = [p for s, e, p in pitched if s <= t + 1e-6 < e]
            chord = _chord_detector.detect_chord_from_notes(sounding)
            name = chord.get_name() if chord else None
            if name != last:
                chords.append((t, name))
                last = name
        return chords

    def chord_at(self, t):
        i = bisect.bisect_right(self._chord_times, t) - 1
        return self._chords[i][1] if i >= 0 else None

    def _build_waveform(self, y):
        x0, y0, x1, y1 = self._box(WAVE_BOX)
        width, height = x1 - x0, y1 - y0
        mono = np.max(np.abs(y), axis=1) if y.size else np.zeros(1)
        edges = np.linspace(0, len(mono), width + 1).astype(int)
        peaks = np.zeros(width, dtype=np.float32)
        for i in range(width):
            segment = mono[edges[i] : max(edges[i + 1], edges[i] + 1)]
            peaks[i] = segment.max() if segment.size else 0.0
        top = peaks.max()
        if top > 0:
            peaks = peaks / top
        half = height // 2
        heights = np.maximum((peaks * (half - 2)).astype(int), 1)
        dim = np.zeros((height, width, 3), dtype=np.uint8)
        dim[:] = PANEL
        bright = dim.copy()
        for x in range(width):
            dim[half - heights[x] : half + heights[x], x, :] = WAVE_DIM
            bright[half - heights[x] : half + heights[x], x, :] = WAVE_BRIGHT
        return dim, bright

    def _key_rect(self, pitch):
        x0 = self._pitch_x0 + (pitch - self._low) * self._col_w
        return x0, x0 + self._col_w

    def _build_background(self):
        img = Image.new("RGB", (self.width, self.height), BACKGROUND)
        d = ImageDraw.Draw(img)
        s = self._s
        d.text(
            (s(20), s(TITLE_Y)), self.specs.title, font=self.f_title, fill=TEXT
        )
        d.text(
            (self.width - s(20), s(22)),
            "KORSHMER",
            font=self.f_head,
            fill=ACCENT,
            anchor="ra",
        )
        roll = self._roll
        d.rectangle(roll, fill=PANEL)
        for pitch in range(self._low, self._high + 1):
            if pitch % 12 in BLACK_PITCH_CLASSES:
                continue
            x0, x1 = self._key_rect(pitch)
            d.rectangle((x0, roll[1], x1, self._keys[1]), fill=(30, 26, 70))
        for box in (WAVE_BOX, NOW_BOX, INFO_BOX):
            d.rectangle(self._box(box), fill=PANEL)
        d.text(
            (s(WAVE_BOX[0]), s(WAVE_BOX[1] - 22)),
            "WAVEFORM (mastered)" if self.specs.mastered else "WAVEFORM",
            font=self.f_text,
            fill=MUTED,
        )
        d.text(
            (s(ROLL_BOX[0]), s(ROLL_BOX[1] - 22)),
            "NOTES",
            font=self.f_text,
            fill=MUTED,
        )
        self._draw_keyboard_base(d)
        self._draw_info(d)
        return img

    def _draw_keyboard_base(self, d):
        s = self._s
        top, bottom = self._keys[1], self._keys[3]
        d.rectangle((self._roll[0], top, self._roll[2], bottom), fill=BLACK_KEY)
        for pitch in range(self._low, self._high + 1):
            x0, x1 = self._key_rect(pitch)
            if pitch % 12 in BLACK_PITCH_CLASSES:
                continue
            d.rectangle((x0, top, x1 - 1, bottom), fill=WHITE_KEY)
        for pitch in range(self._low, self._high + 1):
            x0, x1 = self._key_rect(pitch)
            if pitch % 12 in BLACK_PITCH_CLASSES:
                d.rectangle(
                    (x0, top, x1 - 1, top + (bottom - top) * 0.62),
                    fill=BLACK_KEY,
                )
            elif pitch % 12 == 0 and self._col_w >= s(3):
                d.text(
                    ((x0 + x1) / 2, bottom - s(2)),
                    _pitch_name(pitch),
                    font=self.f_small,
                    fill=(90, 90, 130),
                    anchor="mb",
                )
        drum = (
            self._roll[0],
            top,
            self._roll[0] + s(DRUM_LANE_WIDTH),
            bottom,
        )
        d.rectangle(drum, fill=(60, 56, 100))
        d.text(
            ((drum[0] + drum[2]) / 2, (drum[1] + drum[3]) / 2),
            "DRUMS",
            font=self.f_small,
            fill=TEXT,
            anchor="mm",
        )

    def _draw_info(self, d):
        s = self._s
        x = s(INFO_BOX[0] + 14)
        y = s(INFO_BOX[1] + 10)
        total = len(self.specs.tracks)
        d.text(
            (x, y),
            f"INSTRUMENTS ({total})",
            font=self.f_head,
            fill=ACCENT,
        )
        self._list_y0 = s(INFO_BOX[1] + 42)
        self._visible_tracks = self.specs.listed_tracks
        self._listed = sorted(
            range(total),
            key=lambda i: self.specs.tracks[i].note_count,
            reverse=True,
        )[: self._visible_tracks]
        hidden = total - self._visible_tracks
        if hidden > 0:
            hidden_notes = sum(
                self.specs.tracks[i].note_count
                for i in range(total)
                if i not in self._listed
            )
            ry = self._list_y0 + self._visible_tracks * s(INSTRUMENT_ROW)
            d.text(
                (x, ry),
                f"+{hidden} more tracks ({hidden_notes} notes)",
                font=self.f_text,
                fill=WARNING,
            )
            d.text(
                (x, ry + s(20)),
                "not listed here, still shown in the notes",
                font=self.f_small,
                fill=WARNING,
            )
        spec_y = s(INSTRUMENT_LIST_BOTTOM + 52)
        d.text((x, spec_y), "TECHNICAL SPECS", font=self.f_head, fill=ACCENT)
        for k, (label, value) in enumerate(self.specs.lines()):
            ry = spec_y + s(28) + k * s(19)
            d.text((x, ry), label, font=self.f_text, fill=MUTED)
            d.text(
                (s(INFO_BOX[2] - 14), ry),
                value,
                font=self.f_text,
                fill=TEXT,
                anchor="ra",
            )

    def _track_active(self, k, t):
        starts, ends = self._track_notes[k]
        return bool(np.any((starts <= t) & (ends > t)))

    def frame(self, t):
        t = min(max(t, 0.0), self.duration)
        img = self._background.copy()
        d = ImageDraw.Draw(img)
        s = self._s
        roll_top, key_top = self._roll[1], self._keys[1]
        window = (key_top - roll_top) / self._pps

        for times, color, width in (
            (self._beats, GRID, 1),
            (self._downbeats, BAR_GRID, 2),
        ):
            visible = times[(times >= t) & (times <= t + window)]
            for beat in visible:
                y = key_top - (beat - t) * self._pps
                d.line(
                    (self._roll[0], y, self._roll[2], y),
                    fill=color,
                    width=max(1, s(width)),
                )

        mask = (self._starts < t + window) & (self._ends > t)
        active_color = {}
        names = set()
        drum_hit = False
        gap = max(1, s(1))
        for i in np.nonzero(mask)[0]:
            top = max(key_top - (self._ends[i] - t) * self._pps, roll_top)
            bottom = min(key_top - (self._starts[i] - t) * self._pps, key_top)
            if bottom <= top:
                continue
            on = self._starts[i] <= t
            if self._drum[i]:
                x0 = self._roll[0]
                x1 = x0 + s(DRUM_LANE_WIDTH)
                drum_hit = drum_hit or on
            else:
                kx0, kx1 = self._key_rect(int(self._pitches[i]))
                x0, x1 = kx0 + gap, kx1 - gap
                if on:
                    active_color[int(self._pitches[i])] = self._colors[i]
                    names.add(int(self._pitches[i]))
            color = self._colors[i]
            if on:
                d.rectangle(
                    (x0, top, x1, bottom), fill=color, outline=(255, 255, 255)
                )
            else:
                d.rectangle(
                    (x0, top, x1, bottom),
                    fill=tuple(int(c * 0.7) for c in color),
                )

        key_bottom = self._keys[3]
        for pitch, color in active_color.items():
            kx0, kx1 = self._key_rect(pitch)
            black = pitch % 12 in BLACK_PITCH_CLASSES
            bottom = key_top + (key_bottom - key_top) * (0.62 if black else 1)
            d.rectangle((kx0, key_top, kx1 - 1, bottom), fill=color)
        if drum_hit:
            d.rectangle(
                (
                    self._roll[0],
                    key_top,
                    self._roll[0] + s(DRUM_LANE_WIDTH),
                    key_bottom,
                ),
                fill=ACCENT,
            )
        d.line(
            (self._roll[0], key_top, self._roll[2], key_top),
            fill=ACCENT,
            width=max(2, s(2)),
        )

        self._draw_wave(img, d, t)
        self._draw_now(d, t, names)
        self._draw_instruments(d, t)
        return img

    def _draw_wave(self, img, d, t):
        x0, y0, x1, y1 = self._box(WAVE_BOX)
        width = x1 - x0
        progress = int(width * t / self.duration)
        wave = self._wave_dim.copy()
        wave[:, :progress, :] = self._wave_bright[:, :progress, :]
        img.paste(Image.fromarray(wave), (x0, y0))
        d.line(
            (x0 + progress, y0, x0 + progress, y1),
            fill=(255, 255, 255),
            width=max(2, self._s(2)),
        )

    def _draw_now(self, d, t, names):
        s = self._s
        box = NOW_BOX
        clock = (
            f"{int(t) // 60}:{int(t) % 60:02d} / "
            f"{int(self.duration) // 60}:{int(self.duration) % 60:02d}"
        )
        d.text(
            (s(box[2] - 14), s(box[1] + 8)),
            clock,
            font=self.f_head,
            fill=MUTED,
            anchor="ra",
        )
        shown = sorted(names)[:MAX_NOTE_NAMES]
        note_text = "  ".join(_pitch_name(p) for p in shown) or "-"
        if len(names) > len(shown):
            note_text += f"  +{len(names) - len(shown)}"
        d.text(
            (s(box[0] + 14), s(box[1] + 6)),
            "NOTES NOW",
            font=self.f_text,
            fill=MUTED,
        )
        d.text(
            (s(box[0] + 14), s(box[1] + 28)),
            note_text,
            font=self.f_head,
            fill=TEXT,
        )
        d.text(
            (s(box[0] + 520), s(box[1] + 4)),
            "CHORD",
            font=self.f_text,
            fill=MUTED,
        )
        d.text(
            (s(box[0] + 520), s(box[1] + 20)),
            self.chord_at(t) or "-",
            font=self.f_big,
            fill=ACCENT,
        )

    def _draw_instruments(self, d, t):
        s = self._s
        x = s(INFO_BOX[0] + 14)
        for row, k in enumerate(self._listed):
            info = self.specs.tracks[k]
            ry = self._list_y0 + row * s(INSTRUMENT_ROW)
            on = self._track_active(k, t)
            d.ellipse(
                (x, ry + s(5), x + s(14), ry + s(19)),
                fill=info.color if on else (60, 60, 90),
            )
            d.text(
                (x + s(24), ry),
                info.instrument[:24],
                font=self.f_text,
                fill=TEXT if on else MUTED,
            )
            d.text(
                (s(INFO_BOX[2] - 14), ry),
                "drums" if info.is_drum else f"{info.lowest}-{info.highest}",
                font=self.f_text,
                fill=MUTED,
                anchor="ra",
            )


def _ffmpeg_path():
    path = shutil.which("ffmpeg")
    if not path:
        raise ClipExportError(
            "ffmpeg was not found. Install ffmpeg to export a music clip."
        )
    return path


def _check_cancel(stop_callback):
    if stop_callback and stop_callback():
        raise ClipExportCancelled("Clip export was cancelled.")


def _report(progress_callback, percent, text):
    if progress_callback:
        progress_callback(int(percent), text)


def encode_clip(
    scene, audio_path, save_path, fps, stop_callback, progress_callback
):
    ffmpeg = _ffmpeg_path()
    frame_count = max(int(round(scene.duration * fps)), 1)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{scene.width}x{scene.height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        "-movflags",
        "+faststart",
        save_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for n in range(frame_count):
            _check_cancel(stop_callback)
            try:
                proc.stdin.write(scene.frame(n / fps).tobytes())
            except (BrokenPipeError, OSError):
                break
            if n % fps == 0:
                _report(
                    progress_callback,
                    30 + 68 * n / frame_count,
                    "Drawing frames",
                )
        try:
            proc.stdin.close()
        except OSError:
            pass
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if code != 0:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise ClipExportError(f"ffmpeg failed (exit {code}): {err[-300:]}")


def export_music_clip(
    midi_path,
    save_path,
    fluid_dir,
    sf3_path,
    bpm=DEFAULT_TEMPO,
    fps=DEFAULT_FPS,
    master=True,
    title=None,
    stop_callback=None,
    progress_callback=None,
    render_func=None,
    resolution=DEFAULT_RESOLUTION,
    clips=None,
):
    if not fps or not MIN_FPS <= int(fps) <= MAX_FPS:
        raise ClipExportError(f"fps must be {MIN_FPS}-{MAX_FPS}")
    fps = int(fps)
    if resolution not in RESOLUTIONS:
        raise ClipExportError(
            f"resolution must be one of {', '.join(RESOLUTIONS)}"
        )
    _ffmpeg_path()
    if render_func is None:
        from korshmer.audio_processors import render_midi_to_wav

        render_func = render_midi_to_wav

    base = os.path.splitext(save_path)[0]
    wav_path = f"{base}.wav"
    pm = pretty_midi.PrettyMIDI(midi_path)

    _report(progress_callback, 2, "Rendering MIDI to WAV")
    with tempfile.TemporaryDirectory() as work:
        raw_path = os.path.join(work, "render.wav")
        has_notes = any(inst.notes for inst in pm.instruments)
        rendered = None
        if has_notes or not any(not c.get("muted") for c in clips or []):
            rendered = render_func(
                midi_path,
                raw_path,
                fluid_dir,
                sf3_path,
                bpm=bpm,
                stop_callback=stop_callback,
            )
        _check_cancel(stop_callback)
        if rendered and not os.path.exists(rendered):
            rendered = None
        active_clips = [c for c in (clips or []) if not c.get("muted")]
        if active_clips:
            _report(progress_callback, 8, "Mixing audio clips")
            from korshmer.audio_processing.stem_session import (
                mix_render_with_clips,
            )

            rendered = mix_render_with_clips(
                rendered, active_clips, os.path.join(work, "mix.wav")
            )
            _check_cancel(stop_callback)
        if not rendered or not os.path.exists(rendered):
            raise ClipExportError(
                "The MIDI could not be rendered (check the SoundFont)."
            )

        if master:
            _report(progress_callback, 12, "Smart Mastering")
            from korshmer.audio_processing.save_audio import save_audio
            from korshmer.audio_processing.smart_mastering import (
                SmartMastering,
                SmartMasteringConfig,
            )

            y, sr = sf.read(rendered, dtype="float32", always_2d=True)
            mastered, out_sr = SmartMastering(SmartMasteringConfig()).process(
                y.T, sr
            )
            _check_cancel(stop_callback)
            save_audio(
                path=wav_path,
                y=np.asarray(mastered).T,
                sr=out_sr,
                file_format="WAV",
                bits=32,
            )
        else:
            shutil.copyfile(rendered, wav_path)

        _report(progress_callback, 24, "Analyzing the song")
        specs = collect_tech_specs(
            pm,
            midi_path,
            wav_path,
            title=title,
            raw_audio_path=rendered if master else None,
            mastered=master,
        )
    scene = ClipScene(pm, specs, wav_path, scale=RESOLUTIONS[resolution])
    _check_cancel(stop_callback)
    encode_clip(
        scene, wav_path, save_path, fps, stop_callback, progress_callback
    )
    _report(progress_callback, 100, "Clip ready")
    return save_path
