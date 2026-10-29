import math
import os
from functools import lru_cache

import librosa
import numpy as np
import pretty_midi

from korshmer.audio_processing import analyse_tempo, estimate_meter
from korshmer.constants import NOTE_NAMES, SCALES
from korshmer.utils import run_async, wait_all

_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


DETECTABLE_SCALES = (
    "Major",
    "Minor",
    "Dorian",
    "Phrygian",
    "Lydian",
    "Mixolydian",
    "Harmonic Minor",
    "Pentatonic Major",
    "Pentatonic Minor",
    "Blues",
    "Arabian (Maqam Hijaz)",
    "Gypsy Minor",
)
_OUTSIDE_WEIGHT = 2.3
_INSIDE_WEIGHT = 3.6
_THIRD_WEIGHT = 4.2
_FIFTH_WEIGHT = 5.0
_TONIC_WEIGHT = 6.35
_PROFILE_BIAS = 0.04
_AUDIO_PROFILE_BIAS = 0.05


def _scale_profile(name):
    if name == "Major":
        return _MAJOR_PROFILE
    if name == "Minor":
        return _MINOR_PROFILE
    intervals = SCALES[name]
    profile = np.full(12, _OUTSIDE_WEIGHT)
    for interval in intervals:
        profile[interval] = _INSIDE_WEIGHT
    thirds = [i for i in intervals if i in (3, 4)]
    if thirds:
        profile[thirds[0]] = _THIRD_WEIGHT
    if 7 in intervals:
        profile[7] = _FIFTH_WEIGHT
    profile[0] = _TONIC_WEIGHT
    return profile


def _build_scale_profiles():
    profiles = []
    for name in DETECTABLE_SCALES:
        base = _scale_profile(name)
        for root in range(12):
            vector = np.roll(base, root)
            vector = vector - vector.mean()
            profiles.append((root, name, vector, np.linalg.norm(vector)))
    return profiles


_SCALE_PROFILES = _build_scale_profiles()


def rank_scales(chroma_vals, limit=3, bias=_PROFILE_BIAS):
    cv = np.asarray(chroma_vals, dtype=float)
    cv = cv - np.mean(cv)
    cv_norm = float(np.linalg.norm(cv))
    if cv_norm == 0:
        return [(0, "Major", 0.0)][:limit]
    scored = []
    for root, name, vector, norm in _SCALE_PROFILES:
        score = float(vector @ cv) / (norm * cv_norm + 1e-12)
        if name in ("Major", "Minor"):
            score += bias
        scored.append((score, root, name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(root, name, score) for score, root, name in scored[:limit]]


def _validate_chord_interval(interval):
    if isinstance(interval, bool) or not isinstance(
        interval, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Chord interval must be a positive number, got {interval!r}"
        )
    value = float(interval)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Chord interval must be a finite number above 0, got {interval!r}"
        )
    return value


def _validate_chord_time(value, name):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Chord {name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"Chord {name} must be a finite number of 0 or more, got {value!r}"
        )
    return value


def _chord_windows(start, origin, interval, total_seconds):
    edges = []
    if origin > start + 1e-6:
        edges.append((start, min(origin, total_seconds)))
        first = origin
    else:
        first = start
    count = int(math.ceil((total_seconds - first) / interval - 1e-9))
    for k in range(max(count, 0)):
        begin = first + k * interval
        edges.append((begin, min(begin + interval, total_seconds)))
    return edges


def analyze_chords_full(audio_path, interval=1.0, origin=0.0, start=0.0):
    interval = _validate_chord_interval(interval)
    origin = _validate_chord_time(origin, "origin")
    start = _validate_chord_time(start, "start")
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = np.transpose(chroma)
    frames_per_sec = sr / hop_length
    chord_templates = {
        "Maj": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
        "Min": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
        "Dim": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
        "Aug": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    }

    template_matrix = np.zeros((48, 12))
    template_labels = []
    for root_idx in range(12):
        for ci, (chord_type, template) in enumerate(chord_templates.items()):
            row = root_idx * 4 + ci
            template_matrix[row] = np.roll(template, root_idx)
            template_labels.append(f"{NOTE_NAMES[root_idx]}{chord_type}")

    def analyze_chunk(begin, end):
        first_frame = int(begin * frames_per_sec + 1e-9)
        end_frame = min(
            max(int(end * frames_per_sec + 1e-9), first_frame + 1),
            chroma.shape[0],
        )
        segment = chroma[first_frame:end_frame]
        if segment.shape[0] == 0:
            return None
        avg_chroma = np.mean(segment, axis=0)
        avg_chroma /= np.max(avg_chroma) + 1e-6
        scores = template_matrix @ avg_chroma
        best_idx = np.argmax(scores)
        return (begin, template_labels[best_idx])

    total_seconds = chroma.shape[0] / frames_per_sec
    futures = [
        run_async(analyze_chunk, begin, end)
        for begin, end in _chord_windows(start, origin, interval, total_seconds)
    ]
    results = [r for r in wait_all(*futures) if r is not None]
    return sorted(results, key=lambda x: x[0])


@lru_cache(maxsize=32)
def detect_key_and_mode(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=60)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.sum(chroma, axis=1)
        root_idx, mode, _score = rank_scales(
            chroma_vals, 1, _AUDIO_PROFILE_BIAS
        )[0]
        return root_idx, mode
    except Exception:
        return 0, "Major"


def detect_midi_key(midi_path):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        chroma = np.zeros(12)
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                chroma[note.pitch % 12] += note.end - note.start

        if np.max(chroma) > 0:
            chroma /= np.max(chroma)

        root_idx, mode, _score = rank_scales(chroma, 1)[0]
        return NOTE_NAMES[root_idx], mode
    except Exception:
        return "C", "Major"


def suggest_next_chord(midi_data):
    max_time = 0
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.end > max_time:
                max_time = note.end
    window_start = max(0, max_time - 2.0)
    pitches = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.start >= window_start:
                pitches.append(note.pitch % 12)
    if not pitches:
        return "C Major"
    root_counts = np.bincount(pitches)
    likely_root = np.argmax(root_counts)
    transitions = {
        0: "F Major or G Major",
        2: "G Major or A Minor",
        4: "A Minor or F Major",
        5: "G Major or C Major",
        7: "C Major or A Minor",
        9: "D Minor or F Major",
        11: "C Major",
    }
    root_name = NOTE_NAMES[likely_root]
    suggestion = transitions.get(
        likely_root, "V (Dominant) or IV (Subdominant)"
    )
    return f"Based on {root_name}: Try {suggestion}"


def export_scale_stats(midi_data, output_path):
    all_notes = []
    for inst in midi_data.instruments:
        if not inst.is_drum:
            all_notes.extend([n.pitch % 12 for n in inst.notes])
    if not all_notes:
        return None
    counts = np.bincount(all_notes, minlength=12)
    total = np.sum(counts)
    lines = ["Scale Analysis Report", "=====================", ""]
    sorted_indices = np.argsort(counts)[::-1]
    for idx in sorted_indices:
        pct = (counts[idx] / total) * 100
        lines.append(f"{NOTE_NAMES[idx]}: {counts[idx]} notes ({pct:.1f}%)")
    lines.append("")
    lines.append("Probable Key Candidates:")
    ranked = rank_scales(counts.astype(float), 3)
    root_idx, name, _score = ranked[0]
    lines.append(f"Top Estimation: {NOTE_NAMES[root_idx]} {name}")
    for root_idx, name, _score in ranked[1:]:
        lines.append(f"Also possible: {NOTE_NAMES[root_idx]} {name}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def _sheet_bar_grid(audio_path, tempo, beats_per_bar):
    try:
        from korshmer.audio_processing.bar_grid import find_bar_grid

        y, sr = librosa.load(audio_path, sr=None)
        return find_bar_grid(y, sr, tempo, beats_per_bar)
    except Exception as e:
        print(f"Bar grid search failed, bars start at 0: {e}")
        return None


def _format_clock(seconds):
    whole = int(round(seconds, 2))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def create_chord_sheet(audio_path, output_dir, precision_mode="1 Bar"):
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    tempo = analyse_tempo(audio_path)
    if not tempo:
        tempo = 120

    beats_per_bar = estimate_meter(audio_path)
    if not beats_per_bar:
        beats_per_bar = 4
    beat_duration = 60.0 / tempo

    multiplier = beats_per_bar
    if precision_mode == "Every Beat":
        multiplier = 1
    elif precision_mode == "Half Bar":
        multiplier = beats_per_bar / 2
    elif precision_mode == "1 Bar":
        multiplier = beats_per_bar
    elif precision_mode == "2 Bars":
        multiplier = beats_per_bar * 2

    grid = _sheet_bar_grid(audio_path, tempo, beats_per_bar)
    origin = grid.first_downbeat if grid and grid.pickup_beats else 0.0
    start = grid.content_start if origin else 0.0
    chords = analyze_chords_full(
        audio_path,
        interval=beat_duration * multiplier,
        origin=origin,
        start=start,
    )
    if not chords:
        return None
    pickup = [c for c in chords if origin and c[0] < origin - 1e-6]
    chords = chords[len(pickup) :]
    output_file = os.path.join(output_dir, f"{file_name}_Chords.txt")
    lines = []
    lines.append(f"Chord Sheet: {file_name}")
    lines.append(f"Tempo: {tempo} BPM | Meter: {beats_per_bar}/4")
    if pickup:
        unit = "beat" if grid.pickup_beats == 1 else "beats"
        lines.append(
            f"Pickup: {grid.pickup_beats} {unit} before the first bar line"
        )
    lines.append("=" * 40)
    lines.append("")
    if pickup:
        lines.append("Pickup")
        lines.append("-" * 10)
        for t, chord in pickup:
            lines.append(f"[{_format_clock(t)}] {chord}")
        lines.append("")
    last_bar = 0
    total_bars = (
        int((len(chords) - 1) * multiplier / beats_per_bar + 1e-9) + 1
        if chords
        else 0
    )
    for index, (t, chord) in enumerate(chords):
        bar = int(index * multiplier / beats_per_bar + 1e-9) + 1
        if bar != last_bar:
            if last_bar:
                lines.append("")
            title = f"Bar {bar}"
            if grid and grid.ends_mid_bar and bar == total_bars:
                title += f" ({grid.last_bar_beats} of {beats_per_bar} beats)"
            lines.append(title)
            lines.append("-" * 10)
            last_bar = bar
        lines.append(f"[{_format_clock(t)}] {chord}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_file
