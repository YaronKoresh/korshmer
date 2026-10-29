import copy
import logging
import math
import numbers
import os
import random

import librosa
import numpy as np
import pretty_midi

from korshmer.constants import (
    DATA_DIR,
    DRUM_MAPPINGS,
    GM_DRUM_MAP,
    INSTRUMENT_DB,
    MIN_NOTES_FOR_SPLIT,
    NOTE_NAMES,
    SCALES,
    SPLIT_CLUSTERS,
)
from korshmer.lazy_imports import _get_sklearn_kmeans
from korshmer.midi_processing.quantization import (
    validate_grid_division,
    validate_quantize_strength,
)
from korshmer.utils.tempo import validate_bpm
from korshmer.utils.track_layout import merge_by_instrument

logger = logging.getLogger(__name__)

MAX_DELAY_SECONDS = 60.0
MAX_DELAY_REPEATS = 64
LEGATO_MODES = ("Legato", "Staccato")
VELOCITY_MODES = ("Original", "Fixed", "Compressed")


def _finite_number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(
            f"{name} must be between {low} and {high}, got {value!r}"
        )
    return float(value)


def _save_modified_midi(pm, original_path, suffix, one_track_each=True):
    dir_name = os.path.join(DATA_DIR, "Processed_MIDI")
    os.makedirs(dir_name, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    new_name = f"{base_name}_{suffix}.mid"
    out_path = os.path.join(dir_name, new_name)
    if one_track_each:
        pm.instruments = merge_by_instrument(
            pm.instruments, keep_shared_names=True
        )
    pm.write(out_path)
    return out_path


def apply_humanize(midi_data, timing_amount=0.05, velocity_amount=10):
    timing_amount = _finite_number(timing_amount, "Timing amount", 0)
    velocity_amount = _finite_number(velocity_amount, "Velocity amount", 0)
    for inst in midi_data.instruments:
        for note in inst.notes:
            start_offset = (np.random.random() - 0.5) * timing_amount
            note.start = max(0, note.start + start_offset)
            note.end = max(note.start + 0.05, note.end + start_offset)
            vel_offset = int((np.random.random() - 0.5) * velocity_amount * 2)
            note.velocity = max(1, min(127, note.velocity + vel_offset))
        inst.notes.sort(key=lambda x: x.start)
    return midi_data


def apply_scale_snap(midi_data, root_idx, scale_name):
    if scale_name not in SCALES:
        return midi_data
    intervals = SCALES[scale_name]
    allowed_pitches = {(root_idx + i) % 12 for i in intervals}
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if (note.pitch % 12) not in allowed_pitches:
                original_pitch = note.pitch
                nearest = min(
                    allowed_pitches,
                    key=lambda x: min(
                        abs(x - (original_pitch % 12)),
                        12 - abs(x - (original_pitch % 12)),
                    ),
                )
                diff = nearest - (original_pitch % 12)
                if abs(diff) > 6:
                    diff = diff - 12 if diff > 0 else diff + 12
                note.pitch += diff
                note.pitch = max(0, min(127, note.pitch))
    return midi_data


def apply_smart_inversions(midi_data):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda x: x.start)
        if not notes:
            continue
        chord_groups = []
        current_chord = [notes[0]]
        for i in range(1, len(notes)):
            if abs(notes[i].start - notes[i - 1].start) < 0.05:
                current_chord.append(notes[i])
            else:
                chord_groups.append(current_chord)
                current_chord = [notes[i]]
        chord_groups.append(current_chord)
        avg_pitch = chord_groups[0][0].pitch
        for chord in chord_groups:
            if len(chord) < 3:
                avg_pitch = sum(n.pitch for n in chord) / len(chord)
                continue
            chord.sort(key=lambda n: n.pitch)
            original_pitches = [n.pitch for n in chord]
            min_dist = float("inf")
            best_pitches = list(original_pitches)
            for i in range(3):
                current_pitches = list(original_pitches)
                if i == 1:
                    current_pitches[0] += 12
                elif i == 2:
                    current_pitches[0] += 12
                    current_pitches[1] += 12
                current_pitches.sort()
                current_avg = sum(current_pitches) / len(current_pitches)
                dist = abs(current_avg - avg_pitch)
                if dist < min_dist:
                    min_dist = dist
                    best_pitches = current_pitches
            for n, p in zip(chord, best_pitches, strict=False):
                n.pitch = p
            avg_pitch = sum(n.pitch for n in chord) / len(chord)
    return midi_data


def apply_legato_staccato(midi_data, mode="Legato", percentage=0.9):
    if mode not in LEGATO_MODES:
        raise ValueError(f"Mode must be one of {LEGATO_MODES}, got {mode!r}")
    if mode == "Staccato":
        percentage = _finite_number(percentage, "Percentage", 0, 1)
        if percentage == 0:
            raise ValueError("Percentage must be above 0")
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        inst.notes.sort(key=lambda x: x.start)
        for i in range(len(inst.notes) - 1):
            curr = inst.notes[i]
            next_n = inst.notes[i + 1]
            if mode == "Legato":
                if next_n.start - 0.02 > curr.start:
                    curr.end = next_n.start - 0.02
            elif mode == "Staccato":
                dur = curr.end - curr.start
                curr.end = curr.start + (dur * percentage)
    return midi_data


def apply_arpeggiator(midi_data, pattern="Up", rate="1/16", gate=0.9, bpm=120):
    bpm = validate_bpm(bpm)
    tempo = bpm
    q_note = 60.0 / tempo
    rates = {
        "1/4": q_note,
        "1/8": q_note / 2,
        "1/16": q_note / 4,
        "1/32": q_note / 8,
    }
    step = rates.get(rate, q_note / 4)

    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        sorted_notes = sorted(inst.notes, key=lambda x: x.start)
        if not sorted_notes:
            continue
        new_notes = []
        chord_map = {}
        for note in sorted_notes:
            quantized_start = round(note.start / step) * step
            if quantized_start not in chord_map:
                chord_map[quantized_start] = []
            chord_map[quantized_start].append(note)
        sorted_times = sorted(chord_map.keys())
        for t in sorted_times:
            notes = chord_map[t]
            if not notes:
                continue
            notes.sort(key=lambda n: n.pitch)
            seq = []
            if pattern == "Up":
                seq = notes
            elif pattern == "Down":
                seq = notes[::-1]
            elif pattern == "Up-Down":
                seq = notes[:]
                if len(notes) > 2:
                    seq.extend(notes[-2:0:-1])
            elif pattern == "Random":
                seq = notes[:]
                np.random.shuffle(seq)
            for i, note in enumerate(seq):
                start_time = t + (i * step)
                duration = step * gate
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=start_time,
                    end=start_time + duration,
                )
                new_notes.append(new_note)
        new_notes.sort(key=lambda x: x.start)
        inst.notes = new_notes
    return midi_data


def apply_scale_filter(midi_data, root, scale_name):
    if scale_name not in SCALES:
        return midi_data
    intervals = SCALES[scale_name]
    allowed_pitches = {(root + i) % 12 for i in intervals}
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if (note.pitch % 12) in allowed_pitches:
                continue
            for distance in range(1, 7):
                lower = note.pitch - distance
                upper = note.pitch + distance
                if lower >= 0 and (lower % 12) in allowed_pitches:
                    note.pitch = lower
                    break
                if upper <= 127 and (upper % 12) in allowed_pitches:
                    note.pitch = upper
                    break
    return midi_data


def apply_octave_transpose(midi_data, target_octave):
    if target_octave is None:
        return midi_data
    target_octave = _whole_in_range(target_octave, "Octave", -1, 9)
    for i in midi_data.instruments:
        if not i.is_drum:
            for note in i.notes:
                current_octave = math.floor(note.pitch / 12) - 1
                shift = (target_octave - current_octave) * 12
                note.pitch = max(0, min(127, note.pitch + shift))
    return midi_data


def normalize_note_velocities(midi_data, low=48, high=118):
    notes = [n for inst in midi_data.instruments for n in inst.notes]
    if not notes:
        return midi_data
    velocities = np.array([n.velocity for n in notes], dtype=float)
    floor = float(np.percentile(velocities, 5))
    ceiling = float(np.percentile(velocities, 95))
    span = ceiling - floor
    for note in notes:
        if span < 1.0:
            scaled = (low + high) / 2.0
        else:
            ratio = min(1.0, max(0.0, (note.velocity - floor) / span))
            scaled = low + ratio * (high - low)
        note.velocity = int(round(scaled))
    return midi_data


def process_velocity(midi_data, mode="Original", value=0.0):
    if mode not in VELOCITY_MODES:
        raise ValueError(f"Mode must be one of {VELOCITY_MODES}, got {mode!r}")
    if mode == "Original":
        return midi_data
    value = _finite_number(value, "Velocity value")
    for inst in midi_data.instruments:
        for note in inst.notes:
            if mode == "Fixed":
                note.velocity = int(value)
            elif mode == "Compressed":
                alpha = 0.3
                note.velocity = int(note.velocity * alpha + value * (1 - alpha))
            note.velocity = max(1, min(127, note.velocity))
    return midi_data


def split_by_pitch(midi_data, split_point=60):
    new_instruments = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            new_instruments.append(inst)
            continue
        low_notes = []
        high_notes = []
        for note in inst.notes:
            if note.pitch < split_point:
                low_notes.append(note)
            else:
                high_notes.append(note)
        if low_notes:
            inst_low = pretty_midi.Instrument(
                program=inst.program, is_drum=False, name=f"{inst.name} (Low)"
            )
            inst_low.notes = low_notes
            new_instruments.append(inst_low)
        if high_notes:
            inst_high = pretty_midi.Instrument(
                program=inst.program, is_drum=False, name=f"{inst.name} (High)"
            )
            inst_high.notes = high_notes
            new_instruments.append(inst_high)
    midi_data.instruments = new_instruments
    return midi_data


def apply_open_voicing(midi_data):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda x: x.start)
        chord_map = {}
        for n in notes:
            q = round(n.start, 2)
            if q not in chord_map:
                chord_map[q] = []
            chord_map[q].append(n)
        for t in chord_map:
            cluster = chord_map[t]
            if len(cluster) >= 3:
                cluster.sort(key=lambda x: x.pitch)
                if cluster[1].pitch + 12 <= 127:
                    cluster[1].pitch += 12
    return midi_data


def apply_midi_delay(midi_data, delay_time=0.25, feedback=0.6, repeats=3):
    delay_time = _finite_number(delay_time, "Delay time", 0, MAX_DELAY_SECONDS)
    feedback = _finite_number(feedback, "Feedback", 0, 1)
    repeats = _whole_in_range(repeats, "Repeats", 0, MAX_DELAY_REPEATS)
    new_instruments = []
    for inst in midi_data.instruments:
        new_inst = copy.deepcopy(inst)
        new_inst.name = f"{inst.name} (Delay)"
        original_notes = list(inst.notes)
        delayed_notes = []
        for i in range(1, repeats + 1):
            current_vel_factor = feedback**i
            current_time_shift = delay_time * i
            for note in original_notes:
                new_vel = int(note.velocity * current_vel_factor)
                if new_vel < 5:
                    continue
                delayed_notes.append(
                    pretty_midi.Note(
                        velocity=new_vel,
                        pitch=note.pitch,
                        start=note.start + current_time_shift,
                        end=note.end + current_time_shift,
                    )
                )
        new_inst.notes.extend(delayed_notes)
        new_instruments.append(inst)
        new_instruments.append(new_inst)
    midi_data.instruments = new_instruments
    return midi_data


def apply_density_thin(midi_data, prob=0.5):
    prob = _finite_number(prob, "Probability", 0, 1)
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        inst.notes = [n for n in inst.notes if np.random.random() > prob]
    return midi_data


def apply_groove_template(midi_data, template="MPC Swing", bpm=120.0):
    bpm = validate_bpm(bpm)
    swing_ratio = 0.0
    offset = 0.0
    if template == "MPC Swing":
        swing_ratio = 0.15
    elif template == "Laid Back":
        offset = 0.02
    elif template == "Rushing":
        offset = -0.015
    elif template == "Drunken":
        swing_ratio = 0.05
        offset = 0.01

    tempo = bpm

    sixteenth = 60 / tempo / 4
    for inst in midi_data.instruments:
        for note in inst.notes:
            start_grid = note.start / sixteenth
            swing_delta = 0
            if int(start_grid) % 2 != 0:
                swing_delta = sixteenth * swing_ratio
            total_delta = swing_delta + offset
            note.start += total_delta
            note.end += total_delta
            if template == "Drunken":
                jitter = (np.random.random() - 0.5) * 0.02
                note.start += jitter
                note.end += jitter
    return midi_data


def invert_velocity(midi_data):
    for inst in midi_data.instruments:
        for note in inst.notes:
            note.velocity = max(1, min(127, 128 - note.velocity))
    return midi_data


def apply_smart_loop(midi_data, target_bars=4, bpm=120.0):
    bpm = validate_bpm(bpm)
    q_note = 60 / bpm
    target_duration = target_bars * 4 * q_note
    actual_end = midi_data.get_end_time()
    if actual_end <= 0:
        return midi_data
    ratio = target_duration / actual_end
    for inst in midi_data.instruments:
        for note in inst.notes:
            note.start *= ratio
            note.end *= ratio
        for cc in inst.control_changes:
            cc.time *= ratio
    return midi_data


def apply_ratchet(midi_data, divisions=4):
    new_instruments = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            new_inst = copy.deepcopy(inst)
            new_inst.notes = []
            for note in inst.notes:
                dur = note.end - note.start
                sub_dur = dur / divisions
                for i in range(divisions):
                    t = note.start + (i * sub_dur)
                    new_inst.notes.append(
                        pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=t,
                            end=t + sub_dur * 0.9,
                        )
                    )
            new_instruments.append(new_inst)
        else:
            new_instruments.append(inst)
    midi_data.instruments = new_instruments
    return midi_data


def apply_microtuning(midi_data, amount=2000):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        sorted_notes = sorted(inst.notes, key=lambda x: x.start)
        pitch_bends = []
        for note in sorted_notes:
            bend = int((np.random.random() - 0.5) * amount)
            bend = max(-8192, min(8191, bend))
            pitch_bends.append(
                pretty_midi.PitchBend(pitch=bend, time=note.start)
            )
        inst.pitch_bends.extend(pitch_bends)
    return midi_data


def apply_velocity_ramp(midi_data, start_v=40, end_v=127):
    total_start = min(
        (n.start for i in midi_data.instruments for n in i.notes), default=0
    )
    total_end = max(
        (n.end for i in midi_data.instruments for n in i.notes), default=1
    )
    span = total_end - total_start
    for inst in midi_data.instruments:
        for note in inst.notes:
            pos = (note.start - total_start) / span if span > 0 else 0
            note.velocity = int(start_v + (end_v - start_v) * pos)
    return midi_data


def apply_voice_leading(midi_data):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda x: x.start)
        if not notes:
            continue
        chord_groups = []
        current_chord = [notes[0]]
        for i in range(1, len(notes)):
            if abs(notes[i].start - notes[i - 1].start) < 0.05:
                current_chord.append(notes[i])
            else:
                chord_groups.append(current_chord)
                current_chord = [notes[i]]
        chord_groups.append(current_chord)
        for i in range(len(chord_groups) - 1):
            curr = chord_groups[i]
            next_c = chord_groups[i + 1]
            if len(curr) < 2 or len(next_c) < 2:
                continue
            curr_pitches = [n.pitch for n in curr]
            curr_avg = sum(curr_pitches) / len(curr_pitches)
            best_pitches = [n.pitch for n in next_c]
            min_dist = float("inf")
            [n.pitch % 12 for n in next_c]
            for shift in [-12, 0, 12]:
                test_pitches = [n.pitch + shift for n in next_c]
                test_avg = sum(test_pitches) / len(test_pitches)
                dist = abs(test_avg - curr_avg)
                if dist < min_dist:
                    min_dist = dist
                    best_pitches = test_pitches
            for n, p in zip(next_c, best_pitches, strict=False):
                n.pitch = max(0, min(127, p))
    return midi_data


_LFO_SHAPES = ("Sine", "Square", "Triangle", "Saw")
_LFO_TARGETS = ("Velocity", "CC11 (Expression)")


def _lfo_value(shape, freq, t):
    phase = (freq * t) % 1.0
    if shape == "Sine":
        return math.sin(2 * math.pi * freq * t)
    if shape == "Square":
        return 1.0 if math.sin(2 * math.pi * freq * t) > 0 else -1.0
    if shape == "Triangle":
        return 1.0 - 4.0 * abs(phase - 0.5)
    return 2.0 * phase - 1.0


def apply_lfo_modulation(
    midi_data, freq=1.0, amount=20, target="Velocity", shape="Sine"
):
    if shape not in _LFO_SHAPES:
        raise ValueError(
            f"Unknown LFO shape {shape!r}; expected one of {_LFO_SHAPES}"
        )
    if target not in _LFO_TARGETS:
        raise ValueError(
            f"Unknown LFO target {target!r}; expected one of {_LFO_TARGETS}"
        )
    total_start = min(
        (n.start for i in midi_data.instruments for n in i.notes), default=0
    )
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        if target == "Velocity":
            for n in inst.notes:
                t = n.start - total_start
                mod = _lfo_value(shape, freq, t)
                val_change = int(mod * amount)
                n.velocity = max(1, min(127, n.velocity + val_change))
        else:
            duration = inst.get_end_time()
            step = 0.05
            times = np.arange(0, duration, step)
            cc_events = []
            for t in times:
                mod = _lfo_value(shape, freq, t)
                val = max(0, min(127, int(64 + (mod * 63))))
                cc_events.append(pretty_midi.ControlChange(11, val, t))
            inst.control_changes.extend(cc_events)
    return midi_data


def generate_smart_harmony(
    midi_data, root_note_idx, scale_name, harmony_type="Triads"
):
    if scale_name not in SCALES:
        scale_name = "Major"
    intervals = SCALES[scale_name]
    scale_degrees = []
    for i in range(10):
        base = i * 12 + root_note_idx
        for interval in intervals:
            note = base + interval
            if note < 128:
                scale_degrees.append(note)
    scale_set = set(scale_degrees)
    scale_lookup = sorted(scale_set)
    harmony_inst = pretty_midi.Instrument(
        program=0, is_drum=False, name=f"Harmony ({harmony_type})"
    )
    source_notes = []
    for inst in midi_data.instruments:
        if not inst.is_drum:
            source_notes.extend(inst.notes)
    source_notes.sort(key=lambda x: x.start)
    OFFSET_3RD = 2
    OFFSET_5TH = 4
    OFFSET_7TH = 6
    for note in source_notes:
        closest_idx = min(
            range(len(scale_lookup)),
            key=lambda i: abs(scale_lookup[i] - note.pitch),
        )
        base_pitch = scale_lookup[closest_idx]
        if abs(base_pitch - note.pitch) > 1:
            continue
        chord_indices = [closest_idx]
        if harmony_type == "Triads":
            chord_indices.extend(
                [closest_idx + OFFSET_3RD, closest_idx + OFFSET_5TH]
            )
        elif harmony_type == "7ths":
            chord_indices.extend(
                [
                    closest_idx + OFFSET_3RD,
                    closest_idx + OFFSET_5TH,
                    closest_idx + OFFSET_7TH,
                ]
            )
        elif harmony_type == "Power (5th)":
            chord_indices.extend([closest_idx + OFFSET_5TH])
        elif harmony_type == "Octaves":
            chord_indices.extend([closest_idx - 7])
        for idx in chord_indices:
            if 0 <= idx < len(scale_lookup):
                pitch = scale_lookup[idx]
                if harmony_type != "Octaves" and pitch > note.pitch:
                    pitch -= 12
                if 21 <= pitch <= 108:
                    new_note = pretty_midi.Note(
                        velocity=max(40, int(note.velocity * 0.7)),
                        pitch=pitch,
                        start=note.start,
                        end=note.end,
                    )
                    harmony_inst.notes.append(new_note)
    midi_data.instruments.append(harmony_inst)
    return midi_data


def generate_cc_from_audio(midi_data, audio_path, cc_number=11, step_ms=20):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        hop_length = int(sr * step_ms / 1000)
        if hop_length < 1:
            logger.warning(
                "generate_cc_from_audio: computed hop_length=%d from sr=%s and "
                "step_ms=%s is not positive; clamping to 1 to avoid a crash.",
                hop_length,
                sr,
                step_ms,
            )
            hop_length = 1
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if rms.max() > 0:
            rms_norm = (rms / rms.max() * 127).astype(int)
        else:
            rms_norm = np.zeros_like(rms, dtype=int)
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )
        for inst in midi_data.instruments:
            if inst.is_drum:
                continue
            cc_events = []
            for t, val in zip(times, rms_norm, strict=False):
                cc = pretty_midi.ControlChange(
                    number=cc_number, value=val, time=t
                )
                cc_events.append(cc)
            inst.control_changes.extend(cc_events)
    except Exception:
        pass
    return midi_data


def _song_offset(midi_data, bpm):
    from korshmer.midi_generation import song_match

    return song_match.estimate_offset(midi_data, validate_bpm(bpm))


def _song_triads(midi_data, bpm):
    from korshmer.midi_generation import song_match

    bar = 240.0 / bpm
    end = midi_data.get_end_time()
    offset = song_match.estimate_offset(midi_data, bpm)
    aligned = song_match.shift_song(midi_data, offset)
    out = []
    for index, triad in enumerate(song_match.bar_triads(aligned, bpm)):
        if triad is not None:
            start = index * bar + offset
            out.append((start, min(start + bar, end), triad))
    return out


def generate_simple_bass(midi_data, bpm=120.0):
    bpm = validate_bpm(bpm)
    bass_inst = pretty_midi.Instrument(
        program=33, is_drum=False, name="Smart Bass"
    )

    q = 60 / bpm
    bars = _song_triads(midi_data, bpm)
    if not bars:
        return midi_data

    styles = ["Sustain", "Pulse", "Octaves", "Offbeat"]
    weights = [0.1, 0.5, 0.2, 0.2]

    for t_start, t_end, triad in bars:
        dur = t_end - t_start
        if dur <= 0:
            continue
        current_style = (
            "Sustain" if dur < q else np.random.choice(styles, p=weights)
        )
        bass_pitch = triad[0] + 36
        if bass_pitch < 28:
            bass_pitch += 12

        if current_style == "Sustain":
            bass_inst.notes.append(
                pretty_midi.Note(100, bass_pitch, t_start, t_end)
            )

        elif current_style == "Pulse":
            steps = int(dur / (q / 2))
            for k in range(steps):
                p_time = t_start + k * (q / 2)
                vel = 110 if k % 4 == 0 else 85
                bass_inst.notes.append(
                    pretty_midi.Note(
                        vel, bass_pitch, p_time, p_time + (q / 2) * 0.7
                    )
                )

        elif current_style == "Octaves":
            steps = int(dur / (q / 2))
            for k in range(steps):
                p_time = t_start + k * (q / 2)
                curr_p = bass_pitch if k % 2 == 0 else bass_pitch + 12
                vel = 105 if k % 2 == 0 else 95
                bass_inst.notes.append(
                    pretty_midi.Note(
                        vel, curr_p, p_time, p_time + (q / 2) * 0.8
                    )
                )

        elif current_style == "Offbeat":
            steps = int(dur / (q / 2))
            for k in range(steps):
                if k % 2 != 0:
                    p_time = t_start + k * (q / 2)
                    bass_inst.notes.append(
                        pretty_midi.Note(
                            100, bass_pitch, p_time, p_time + (q / 2) * 0.8
                        )
                    )

    midi_data.instruments.append(bass_inst)
    return midi_data


def add_drum_fill(midi_data, interval_bars=4, bpm=120.0):
    bpm = validate_bpm(bpm)
    bar_dur = (60 / bpm) * 4
    offset = _song_offset(midi_data, bpm)
    for inst in midi_data.instruments:
        if not inst.is_drum:
            continue
        max_time = inst.get_end_time()
        curr_bar = 1
        while offset + curr_bar * bar_dur < max_time:
            if curr_bar % interval_bars == 0:
                fill_end = offset + curr_bar * bar_dur
                fill_start = fill_end - (60 / bpm)
                inst.notes = [
                    n
                    for n in inst.notes
                    if not (fill_start <= n.start < fill_end)
                ]

                steps = 4
                step_len = (fill_end - fill_start) / steps
                for i in range(steps):
                    t = fill_start + (i * step_len)
                    vel = 90 + (i * 10)
                    inst.notes.append(
                        pretty_midi.Note(vel, 38, t, t + step_len / 2)
                    )
                inst.notes.append(
                    pretty_midi.Note(120, 49, fill_end - 0.05, fill_end + 0.5)
                )

            curr_bar += 1
        inst.notes.sort(key=lambda x: x.start)
    return midi_data


def generate_sustain_pedal(midi_data, threshold=0.1):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        cc_events = []
        notes = sorted(inst.notes, key=lambda x: x.start)
        if not notes:
            continue
        cc_events.append(pretty_midi.ControlChange(64, 127, notes[0].start))
        for i in range(len(notes) - 1):
            curr = notes[i]
            next_n = notes[i + 1]
            gap = next_n.start - curr.end
            if gap > threshold:
                cc_events.append(pretty_midi.ControlChange(64, 0, curr.end))
                cc_events.append(
                    pretty_midi.ControlChange(64, 127, next_n.start)
                )
        inst.control_changes.extend(cc_events)
    return midi_data


def add_ghost_notes(midi_data, prob=0.3, bpm=120.0):
    bpm = validate_bpm(bpm)
    step = 60 / bpm / 4
    offset = _song_offset(midi_data, bpm)
    for inst in midi_data.instruments:
        existing_grid = {round((n.start - offset) / step) for n in inst.notes}

        max_tick = int((inst.get_end_time() - offset) / step)

        new_ghosts = []
        for i in range(max_tick):
            if i in existing_grid:
                continue
            pos_in_bar = i % 16
            is_off_beat = i % 2 != 0
            current_prob = prob

            if is_off_beat:
                current_prob += 0.15
            elif pos_in_bar in [0, 4, 8, 12]:
                current_prob -= 0.2

            if random.random() < current_prob:
                t = offset + i * step
                velocity = random.randint(28, 42)
                new_ghosts.append(pretty_midi.Note(velocity, 38, t, t + 0.05))
        inst.notes.extend(new_ghosts)
        inst.notes.sort(key=lambda x: x.start)
    return midi_data


def _followed_progression(midi_data, bpm, root_idx, intervals):
    from korshmer.midi_generation import song_match

    bpm = validate_bpm(bpm)
    aligned = song_match.shift_song(
        midi_data, song_match.estimate_offset(midi_data, bpm)
    )
    analysis = song_match.analyze_song(aligned, bpm, root_idx, intervals)
    return song_match.progression_for(analysis, "Strong")


def generate_melody_algo(
    midi_data, scale_name, root_idx, _density=0.6, bpm=120.0
):
    if scale_name not in SCALES:
        scale_name = "Major"
    intervals = SCALES[scale_name]

    lead_prog = random.choice(
        INSTRUMENT_DB.get("synth_lead", [])
        + INSTRUMENT_DB.get("guitar", [])
        + INSTRUMENT_DB.get("brass", [])
    )[3]

    counter_prog = random.choice(
        INSTRUMENT_DB.get("strings", []) + INSTRUMENT_DB.get("woodwind", [])
    )[3]

    pluck_prog = random.choice(
        INSTRUMENT_DB.get("ethnic", []) + INSTRUMENT_DB.get("chromatic", [])
    )[3]
    pad_prog = random.choice(
        INSTRUMENT_DB.get("synth_pad", []) + INSTRUMENT_DB.get("organ", [])
    )[3]
    bass_prog = random.choice(
        INSTRUMENT_DB.get("bass", []) + INSTRUMENT_DB.get("synth_lead", [])
    )[3]
    arp_prog = random.choice(
        INSTRUMENT_DB.get("keys", []) + INSTRUMENT_DB.get("guitar", [])
    )[3]
    fx_prog = random.choice(
        INSTRUMENT_DB.get("vocals", []) + INSTRUMENT_DB.get("brass", [])
    )[3]

    inst_lead = pretty_midi.Instrument(
        program=lead_prog, is_drum=False, name="Layer 1: Main Hook"
    )
    inst_counter = pretty_midi.Instrument(
        program=counter_prog, is_drum=False, name="Layer 2: Counter"
    )
    inst_pluck = pretty_midi.Instrument(
        program=pluck_prog, is_drum=False, name="Layer 3: Pluck"
    )
    inst_pad = pretty_midi.Instrument(
        program=pad_prog, is_drum=False, name="Layer 4: Atmosphere"
    )
    inst_bass = pretty_midi.Instrument(
        program=bass_prog, is_drum=False, name="Layer 5: Bass"
    )
    inst_arp = pretty_midi.Instrument(
        program=arp_prog, is_drum=False, name="Layer 6: Arp"
    )
    inst_fx = pretty_midi.Instrument(
        program=fx_prog, is_drum=False, name="Layer 7: FX"
    )

    tempo = bpm
    q = 60 / tempo
    step_16 = q / 4

    prog_degrees = [5, 3, 0, 4]
    followed = _followed_progression(midi_data, bpm, root_idx, intervals)
    offset = _song_offset(midi_data, bpm)

    full_scale = []
    for oct in range(1, 8):
        full_scale.extend([root_idx + i + (oct * 12) for i in intervals])

    total_len = midi_data.get_end_time()
    if total_len <= 0:
        total_len = 16

    def generate_rhythm_bar(density_factor):
        rhythm = []
        rem = q * 4
        while rem > 0:
            options = [q / 4, q / 2, q * 0.75, q]
            weights = [0.4, 0.4, 0.15, 0.05]

            d = random.choices(options, weights=weights, k=1)[0]
            if d > rem:
                d = rem

            is_strong_beat = rem % q == 0
            play = random.random() < (
                density_factor + (0.2 if is_strong_beat else -0.1)
            )

            rhythm.append({"dur": d, "play": play})
            rem -= d
        return rhythm

    rhythm_A = generate_rhythm_bar(0.7)
    rhythm_B = generate_rhythm_bar(0.8)
    rhythm_C = generate_rhythm_bar(0.5)

    curr_bar = 0
    while offset + curr_bar * q * 4 < total_len:
        bar_start = offset + curr_bar * q * 4

        if followed is not None and curr_bar < len(followed):
            chord_degree_idx = followed[curr_bar]
        else:
            chord_degree_idx = prog_degrees[curr_bar % len(prog_degrees)]
        chord_pcs = [
            (root_idx + intervals[(chord_degree_idx + step) % len(intervals)])
            % 12
            for step in (0, 2, 4)
        ]
        chord_root_val = chord_pcs[0]

        chord_tones = [n for n in full_scale if (n % 12) in chord_pcs]
        if not chord_tones:
            chord_tones = full_scale

        pad_notes = [
            48 + (pc - chord_root_val) % 12 + chord_root_val for pc in chord_pcs
        ]
        for p in pad_notes:
            inst_pad.notes.append(
                pretty_midi.Note(55, p, bar_start, bar_start + q * 4)
            )

        bass_note = chord_root_val + 24
        if bass_note < 28:
            bass_note += 12
        bass_pattern = (
            [1, 0, 1, 0, 1, 1, 0, 1]
            if curr_bar % 2 == 0
            else [1, 0, 1, 0, 1, 0, 1, 0]
        )
        for i, hit in enumerate(bass_pattern):
            if hit:
                t = bar_start + (i * (q / 2))
                inst_bass.notes.append(
                    pretty_midi.Note(
                        110 if i == 0 else 95, bass_note, t, t + q / 2 * 0.8
                    )
                )

        for i in range(16):
            if i % 3 == 0 or random.random() > 0.6:
                t = bar_start + (i * step_16)
                p = random.choice([n for n in chord_tones if 48 <= n <= 72])
                inst_arp.notes.append(pretty_midi.Note(60, p, t, t + step_16))

        for i in range(16):
            if (i % 3 == 0) and random.random() > 0.3:
                t = bar_start + (i * step_16)
                p = random.choice([n for n in chord_tones if 60 <= n <= 84])
                inst_pluck.notes.append(pretty_midi.Note(70, p, t, t + 0.1))

        phrase_idx = curr_bar % 4
        if phrase_idx == 0:
            current_rhythm = rhythm_A
        elif phrase_idx == 1:
            current_rhythm = rhythm_A
        elif phrase_idx == 2:
            current_rhythm = rhythm_B
        else:
            current_rhythm = rhythm_C

        current_time_in_bar = 0
        last_pitch = -1

        for event in current_rhythm:
            t = bar_start + current_time_in_bar

            if event["play"]:
                min_pitch = 60
                max_pitch = 81

                is_strong = current_time_in_bar % q == 0

                if is_strong:
                    options = [
                        n for n in chord_tones if min_pitch <= n <= max_pitch
                    ]
                else:
                    options = [
                        n for n in full_scale if min_pitch <= n <= max_pitch
                    ]

                if not options:
                    options = [72]

                if last_pitch != -1 and random.random() > 0.3:
                    pitch = min(options, key=lambda x: abs(x - last_pitch))
                    if random.random() > 0.8:
                        pitch = random.choice(options)
                else:
                    pitch = random.choice(options)

                last_pitch = pitch

                vel = random.randint(105, 127)
                dur = event["dur"] * 0.9

                inst_lead.notes.append(pretty_midi.Note(vel, pitch, t, t + dur))

                if event["dur"] >= q and random.random() > 0.5:
                    cnt_pitch = random.choice(
                        [n for n in chord_tones if 48 <= n <= 60]
                    )
                    inst_counter.notes.append(
                        pretty_midi.Note(80, cnt_pitch, t + q / 2, t + q)
                    )

            current_time_in_bar += event["dur"]

        if phrase_idx == 3:
            fx_note = random.choice([n for n in chord_tones if 36 <= n <= 60])
            inst_fx.notes.append(
                pretty_midi.Note(
                    90, fx_note, bar_start + q * 3.5, bar_start + q * 4
                )
            )

        curr_bar += 1

    midi_data.instruments.extend(
        [
            inst_lead,
            inst_counter,
            inst_pluck,
            inst_pad,
            inst_bass,
            inst_arp,
            inst_fx,
        ]
    )

    return midi_data


def _whole_in_range(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if not isinstance(value, (float, np.floating)) or not (
            math.isfinite(value) and value == int(value)
        ):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}, got {value}"
        )
    return value


def generate_euclidean_rhythm(
    midi_data, steps=16, hits=4, note_num=36, bpm=120.0
):
    steps = _whole_in_range(steps, "Steps", 1, 256)
    hits = _whole_in_range(hits, "Hits", 0, steps)
    note_num = _whole_in_range(note_num, "Note", 0, 127)
    tempo = validate_bpm(bpm)
    sixteenth = 60 / tempo / 4
    cycle_dur = sixteenth * steps
    pattern = [0] * steps
    if hits > 0:
        bucket = 0
        for i in range(steps):
            bucket += hits
            if bucket >= steps:
                bucket -= steps
                pattern[i] = 1
    drum_inst = pretty_midi.Instrument(
        program=0, is_drum=True, name=f"Euclidean {hits}/{steps}"
    )
    total_len = midi_data.get_end_time()
    current_bar = 0
    while current_bar * cycle_dur < total_len:
        offset = current_bar * cycle_dur
        for i, bit in enumerate(pattern):
            if bit:
                t = offset + (i * sixteenth)
                drum_inst.notes.append(
                    pretty_midi.Note(100, note_num, t, t + sixteenth / 2)
                )
        current_bar += 1
    midi_data.instruments.append(drum_inst)
    return midi_data


def generate_edm_pluck(path, bpm=120.0):
    bpm = validate_bpm(bpm)
    midi_data = pretty_midi.PrettyMIDI(path)
    inst = pretty_midi.Instrument(program=81, is_drum=False, name="EDM Pluck")

    q = 60 / bpm
    rate = q / 4

    all_notes = [
        n
        for inst in midi_data.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    if not all_notes:
        return midi_data
    all_notes.sort(key=lambda x: x.start)

    end_time = midi_data.get_end_time()
    curr_time = 0

    def get_chord_tones_at(t):
        active = [n.pitch % 12 for n in all_notes if n.start <= t <= n.end]
        if not active:
            return [0, 4, 7]
        return sorted(set(active))

    arp_index = 0
    while curr_time < end_time:
        tones = get_chord_tones_at(curr_time)
        base_octave = 5

        pitch_class = tones[arp_index % len(tones)]
        pitch = pitch_class + (base_octave * 12)

        if arp_index % 4 == 0:
            pitch += 12

        vel = 100 if (arp_index % 4 == 0) else 70

        duration = rate * 0.4

        inst.notes.append(
            pretty_midi.Note(vel, pitch, curr_time, curr_time + duration)
        )

        curr_time += rate
        arp_index += 1

    midi_data.instruments.append(inst)
    return _save_modified_midi(midi_data, path, "Pluck")


def generate_pad_layer(midi_data, bpm=120.0):
    bpm = validate_bpm(bpm)
    prog = np.random.choice([90, 91, 95, 51])
    pad_inst = pretty_midi.Instrument(
        program=int(prog), is_drum=False, name="Sidechain Pad"
    )

    bars = _song_triads(midi_data, bpm)
    if not bars:
        return midi_data

    def above(pitch, pc):
        pitch += 1
        while pitch % 12 != pc:
            pitch += 1
        return pitch

    for t_start, t_end, (root, third, fifth) in bars:
        if t_end - t_start < 0.2:
            continue
        low = 36 + root
        mid = low + 12
        pad_notes = [
            low,
            mid,
            above(mid, third),
            above(above(mid, third), fifth),
        ]
        for p in pad_notes:
            pad_inst.notes.append(pretty_midi.Note(80, p, t_start, t_end))

    tempo = bpm
    q = 60 / tempo

    resolution = 0.02
    curr_cc = 0
    end_time = midi_data.get_end_time()

    while curr_cc < end_time:
        beat_pos = (curr_cc % q) / q

        volume_curve = math.pow(beat_pos, 0.5)
        cc_val = int(30 + (97 * volume_curve))

        pad_inst.control_changes.append(
            pretty_midi.ControlChange(11, cc_val, curr_cc)
        )
        curr_cc += resolution

    midi_data.instruments.append(pad_inst)
    return midi_data


def generate_standard_beat(midi_data, style="Rock", bars=None, bpm=120.0):
    bpm = validate_bpm(bpm)
    tempo = bpm
    if not tempo or tempo <= 0:
        tempo = 120.0
    while tempo < 70:
        tempo *= 2
    while tempo > 145:
        tempo /= 2

    q = 60 / tempo
    sx = q / 4

    if bars is None:
        total_duration = midi_data.get_end_time()
        if total_duration <= 0:
            total_duration = 10
        bars = int(math.ceil(total_duration / (q * 4)))

    drum_inst = pretty_midi.Instrument(
        program=0, is_drum=True, name=f"Pro Drums {style}"
    )

    def add_hit(pitch, time, vel_base, dur=0.1):
        vel = int(max(1, min(127, vel_base + random.randint(-5, 5))))
        time_drift = random.uniform(-0.005, 0.005)
        drum_inst.notes.append(
            pretty_midi.Note(
                vel, pitch, time + time_drift, time + time_drift + dur
            )
        )

    styles_db = {
        "Rock": {
            "kick": [
                [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0],
            ],
            "snare": [
                [0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 2, 0, 3, 3],
            ],
            "hihat": [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
            "crash": True,
        },
        "Techno": {
            "kick": [[2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0]],
            "snare": [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            "hihat": [[0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0]],
            "crash": False,
        },
        "HipHop": {
            "kick": [
                [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0],
            ],
            "snare": [[0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]],
            "hihat": [[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]],
            "crash": True,
        },
        "Trap": {
            "kick": [[2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0]],
            "snare": [[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]],
            "hihat": [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
            "crash": True,
        },
    }

    current_style = styles_db.get(style, styles_db["Rock"])

    KICK = 36
    SNARE = 38
    CLAP = 39
    HH_CLOSED = 42
    HH_OPEN = 46
    CRASH = 49
    TOM_LO = 43
    TOM_HI = 50
    SNARE_NOTE = CLAP if style == "Techno" else SNARE
    if style == "Trap":
        SNARE_NOTE = 38

    for b in range(bars):
        bar_start_time = b * q * 4

        is_fill_bar = (b + 1) % 4 == 0 and style != "Techno"

        if is_fill_bar:
            add_hit(CRASH, bar_start_time, 110)
            add_hit(KICK, bar_start_time, 110)
            add_hit(SNARE_NOTE, bar_start_time + q, 110)

            fill_start = bar_start_time + q * 2
            notes = [SNARE_NOTE] * 2 + [TOM_HI] * 2 + [TOM_LO] * 4

            for i, note in enumerate(notes):
                t = fill_start + (i * sx)
                add_hit(note, t, 100 + i * 2)

        else:
            var_idx = b % len(current_style["kick"])

            kick_pat = current_style["kick"][var_idx]
            snare_pat = current_style["snare"][b % len(current_style["snare"])]
            hh_pat = current_style["hihat"][0]

            for step in range(16):
                time = bar_start_time + (step * sx)

                k_val = kick_pat[step]
                if k_val == 2:
                    add_hit(KICK, time, 115)
                elif k_val == 1:
                    add_hit(KICK, time, 95)

                s_val = snare_pat[step]
                if s_val == 2:
                    add_hit(SNARE_NOTE, time, 115)
                elif s_val == 3:
                    add_hit(SNARE_NOTE, time, 50)

                h_val = hh_pat[step]

                if style == "Trap" and step % 8 == 0 and random.random() > 0.6:
                    for r in range(4):
                        add_hit(HH_CLOSED, time + (r * sx / 4), 80 + r * 5)
                elif h_val > 0:
                    note = (
                        HH_OPEN
                        if style == "Techno" and h_val == 2
                        else HH_CLOSED
                    )
                    vel = 90 if step % 4 == 0 else 70
                    add_hit(note, time, vel)

            if b == 0 or b % 8 == 0:
                if current_style["crash"]:
                    add_hit(CRASH, bar_start_time, 100)

    midi_data.instruments.append(drum_inst)
    return midi_data


def generate_click_track(midi_data, bpm=120.0):
    bpm = validate_bpm(bpm)
    tempo = bpm
    end_time = midi_data.get_end_time()
    beat_dur = 60 / tempo
    click_inst = pretty_midi.Instrument(
        program=0, is_drum=True, name="Click Track"
    )
    offset = _song_offset(midi_data, bpm)
    t = offset % beat_dur
    bar_count = -round((offset - t) / beat_dur) % 4
    while t < end_time:
        pitch = 37 if bar_count % 4 == 0 else 31
        vel = 110 if bar_count % 4 == 0 else 90
        click_inst.notes.append(pretty_midi.Note(vel, pitch, t, t + 0.1))
        t += beat_dur
        bar_count += 1
    midi_data.instruments.append(click_inst)
    return midi_data


def generate_chord_progression(
    midi_data, root_name, scale_name, prog_type="Pop (I-V-vi-IV)", bpm=120.0
):
    if root_name == "Auto" or "Auto" in scale_name:
        return midi_data

    root_idx = NOTE_NAMES.index(root_name)
    intervals = SCALES.get(scale_name, SCALES["Major"])
    scale_notes = [(root_idx + i) % 12 for i in intervals]

    progs = {
        "Pop (I-V-vi-IV)": [0, 4, 5, 3],
        "Jazz (ii-V-I)": [1, 4, 0],
        "Blues (I-IV-V)": [0, 3, 4],
        "Sad (vi-IV-I-V)": [5, 3, 0, 4],
        "Canon (I-V-vi-iii)": [0, 4, 5, 2],
    }
    indices = progs.get(prog_type, [0, 4, 5, 3])

    prog_inst = pretty_midi.Instrument(
        program=4, is_drum=False, name=f"Prog {prog_type}"
    )

    tempo = bpm
    bar_dur = (60 / tempo) * 4
    current_time = 0

    for _ in range(4):
        for idx in indices:
            if idx < len(scale_notes):
                root_note = scale_notes[idx] + 48
                third_idx = (idx + 2) % len(scale_notes)
                fifth_idx = (idx + 4) % len(scale_notes)

                notes = [
                    root_note,
                    scale_notes[third_idx]
                    + (
                        48 if scale_notes[third_idx] >= scale_notes[idx] else 60
                    ),
                    scale_notes[fifth_idx]
                    + (
                        48 if scale_notes[fifth_idx] >= scale_notes[idx] else 60
                    ),
                ]

                for p in notes:
                    prog_inst.notes.append(
                        pretty_midi.Note(
                            75, p, current_time, current_time + bar_dur
                        )
                    )
            current_time += bar_dur

    midi_data.instruments.append(prog_inst)
    return midi_data


def generate_melody_continuation(midi_data, bars=4, bpm=120.0):
    bpm = validate_bpm(bpm)
    source_notes = []
    for inst in midi_data.instruments:
        if not inst.is_drum:
            source_notes.extend(inst.notes)
    if not source_notes:
        return midi_data
    source_notes.sort(key=lambda x: x.start)
    transitions = {}
    for i in range(len(source_notes) - 1):
        curr = source_notes[i]
        next_n = source_notes[i + 1]
        interval = next_n.pitch - curr.pitch
        dur = curr.end - curr.start
        gap = next_n.start - curr.start
        key = (interval, round(dur, 2))
        if key not in transitions:
            transitions[key] = []
        if i < len(source_notes) - 2:
            next_next = source_notes[i + 2]
            next_interval = next_next.pitch - next_n.pitch
            next_dur = next_n.end - next_n.start
            transitions[key].append((next_interval, next_dur, gap))

    last_note = source_notes[-1]
    current_pitch = last_note.pitch

    current_time = midi_data.get_end_time()
    if current_time < last_note.end:
        current_time = last_note.end

    tempo = bpm
    end_time = current_time + (bars * 4 * (60 / tempo))
    melody_inst = pretty_midi.Instrument(
        program=73, is_drum=False, name="Continuation"
    )
    transitions = {key: opts for key, opts in transitions.items() if opts}
    if not transitions:
        return midi_data
    min_gap = 60.0 / tempo / 16
    last_interval = 0
    if len(source_notes) > 1:
        last_interval = source_notes[-1].pitch - source_notes[-2].pitch
    last_dur = round(source_notes[-1].end - source_notes[-1].start, 2)
    state = (last_interval, last_dur)
    while current_time < end_time:
        options = transitions.get(state)
        if not options:
            state = list(transitions.keys())[
                np.random.randint(len(transitions))
            ]
            options = transitions[state]
        next_interval, next_dur, gap = options[np.random.randint(len(options))]
        new_pitch = max(0, min(127, current_pitch + next_interval))
        melody_inst.notes.append(
            pretty_midi.Note(
                velocity=np.random.randint(80, 110),
                pitch=new_pitch,
                start=current_time,
                end=current_time + next_dur,
            )
        )
        current_time += max(gap, min_gap)
        current_pitch = new_pitch
        state = (next_interval, round(next_dur, 2))
    midi_data.instruments.append(melody_inst)
    return midi_data


def generate_polyrhythms(midi_data, steps_a=3, steps_b=4, bpm=120.0):
    bpm = validate_bpm(bpm)
    tempo = bpm
    bar_duration = (60 / tempo) * 4
    inst_a = pretty_midi.Instrument(
        program=0, is_drum=True, name=f"Poly {steps_a}"
    )
    inst_b = pretty_midi.Instrument(
        program=0, is_drum=True, name=f"Poly {steps_b}"
    )
    total_bars = 4
    for b in range(total_bars):
        offset = b * bar_duration
        step_len_a = bar_duration / steps_a
        for i in range(steps_a):
            t = offset + (i * step_len_a)
            inst_a.notes.append(pretty_midi.Note(100, 36, t, t + 0.1))
        step_len_b = bar_duration / steps_b
        for i in range(steps_b):
            t = offset + (i * step_len_b)
            inst_b.notes.append(pretty_midi.Note(90, 38, t, t + 0.1))
    midi_data.instruments.extend([inst_a, inst_b])
    return midi_data


def simplify_chord_density(midi_data, max_voices=3):
    CHORD_WINDOW = 0.15
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        sorted_notes = sorted(inst.notes, key=lambda x: x.start)
        new_notes = []
        i = 0
        while i < len(sorted_notes):
            current_note = sorted_notes[i]
            chord_group = [current_note]
            j = i + 1
            while j < len(sorted_notes):
                next_note = sorted_notes[j]
                if abs(next_note.start - current_note.start) < CHORD_WINDOW:
                    chord_group.append(next_note)
                    j += 1
                else:
                    break
            if len(chord_group) > max_voices:
                chord_group.sort(key=lambda n: n.pitch)
                lowest = chord_group[0]
                highest = chord_group[-1]
                keep_list = [lowest]
                if lowest != highest:
                    keep_list.append(highest)
                slots_left = max_voices - len(keep_list)
                if slots_left > 0:
                    middle_notes = chord_group[1:-1]
                    middle_notes.sort(key=lambda n: n.velocity, reverse=True)
                    keep_list.extend(middle_notes[:slots_left])
                new_notes.extend(keep_list)
            else:
                new_notes.extend(chord_group)
            i = j
        new_notes.sort(key=lambda x: x.start)
        inst.notes = new_notes
    return midi_data


OUTLIER_MIN_NOTES = 20
OUTLIER_PERCENT = 3


def smart_instrument_selector(stem_name, midi_data):
    stem = stem_name.lower()
    if "drum" in stem:
        for i in midi_data.instruments:
            i.program = 0
            i.is_drum = True
            i.name = "Drums"
        return midi_data
    all_notes = [n for i in midi_data.instruments for n in i.notes]
    if not all_notes:
        return midi_data
    pitches = [n.pitch for n in all_notes]
    if len(pitches) >= OUTLIER_MIN_NOTES:
        stem_min = int(np.percentile(pitches, OUTLIER_PERCENT))
        stem_max = int(np.percentile(pitches, 100 - OUTLIER_PERCENT))
    else:
        stem_min, stem_max = min(pitches), max(pitches)
    stem_range = stem_max - stem_min
    search_categories = []
    if "bass" in stem:
        search_categories = ["bass", "synth_bass", "keys", "brass"]
    elif "guitar" in stem:
        search_categories = ["guitar", "keys", "organ", "synth_lead"]
    elif "piano" in stem or "keys" in stem:
        search_categories = ["keys", "chromatic", "guitar", "synth_pad"]
    elif "vocals" in stem:
        search_categories = ["vocals", "synth_lead", "woodwind", "strings"]
    else:
        search_categories = [
            "strings",
            "woodwind",
            "brass",
            "synth_pad",
            "keys",
        ]
    candidates = []
    for cat in search_categories:
        if cat in INSTRUMENT_DB:
            candidates.extend(INSTRUMENT_DB[cat])
    best = None
    for name, inst_min, inst_max, prog in candidates:
        inst_range = inst_max - inst_min
        if stem_range > inst_range:
            continue
        if stem_min < inst_min:
            total_shift = -(-(inst_min - stem_min) // 12) * 12
        elif stem_max > inst_max:
            total_shift = -(-(stem_max - inst_max) // 12) * -12
        else:
            total_shift = 0
        if (
            stem_min + total_shift < inst_min
            or stem_max + total_shift > inst_max
        ):
            continue
        if best is None or abs(total_shift) < abs(best[0]):
            best = (total_shift, name, prog, inst_min, inst_max)
            if total_shift == 0:
                break
    if best is None:
        default_inst = pretty_midi.Instrument(
            program=0,
            is_drum=False,
            name=f"{stem_name.capitalize()} - Default Piano",
        )
        default_inst.notes = all_notes
        new_instruments = [default_inst]
    else:
        total_shift, name, prog, inst_min, inst_max = best
        current_notes = copy.deepcopy(all_notes)
        for n in current_notes:
            pitch = n.pitch + total_shift
            while pitch < inst_min and pitch + 12 <= inst_max:
                pitch += 12
            while pitch > inst_max and pitch - 12 >= inst_min:
                pitch -= 12
            n.pitch = max(0, min(127, pitch))
        new_inst = pretty_midi.Instrument(
            program=prog,
            is_drum=False,
            name=f"{stem_name.capitalize()} - {name}",
        )
        new_inst.notes = current_notes
        new_instruments = [new_inst]
    midi_data.instruments = new_instruments
    return midi_data


def fix_legato_and_flutter(midi_obj, min_duration=0.12):
    MIN_DURATION = min_duration
    GAP_THRESHOLD = 0.04
    FRAGMENT_DURATION = 0.1
    for inst in midi_obj.instruments:
        if inst.is_drum:
            continue
        sorted_notes = sorted(inst.notes, key=lambda x: x.start)
        new_notes = []
        if not sorted_notes:
            continue
        current_note = sorted_notes[0]
        for i in range(1, len(sorted_notes)):
            next_note = sorted_notes[i]
            gap = next_note.start - current_note.end
            is_fragment = (
                min(
                    current_note.end - current_note.start,
                    next_note.end - next_note.start,
                )
                < FRAGMENT_DURATION
            )
            if next_note.pitch == current_note.pitch and (
                gap < 0 or (gap < GAP_THRESHOLD and is_fragment)
            ):
                current_note.end = max(current_note.end, next_note.end)
            else:
                if (current_note.end - current_note.start) >= MIN_DURATION:
                    new_notes.append(current_note)
                current_note = next_note
        if (current_note.end - current_note.start) >= MIN_DURATION:
            new_notes.append(current_note)
        inst.notes = new_notes
    return midi_obj


def fit_beat_grid(
    onset_times,
    tempo_hint,
    grid_divisions=4,
    search=0.04,
    resolution=0.0001,
    min_onsets=16,
    min_alignment=0.4,
):
    times = np.asarray(onset_times, dtype=float)
    times = times[np.isfinite(times)]
    if len(times) < min_onsets:
        return None
    if not _is_positive_number(tempo_hint) or not _is_positive_number(
        grid_divisions
    ):
        return None
    steps = int(round(search / resolution))
    tempos = tempo_hint * (1.0 + resolution * np.arange(-steps, steps + 1))
    freqs = grid_divisions * tempos / 60.0
    phases = np.exp(2j * np.pi * np.outer(freqs, times))
    resultant = phases.mean(axis=1)
    best = int(np.argmax(np.abs(resultant)))
    alignment = float(np.abs(resultant[best]))
    if alignment < min_alignment:
        return None
    phase = float(np.angle(resultant[best]) / (2 * np.pi))
    return float(tempos[best]), phase / float(freqs[best]), alignment


def fit_beat_grid_or_best(onset_times, tempo_hint, grid_divisions=4):
    grid = fit_beat_grid(onset_times, tempo_hint, grid_divisions)
    if grid is not None:
        return grid
    return fit_beat_grid(
        onset_times,
        tempo_hint,
        grid_divisions,
        min_onsets=1,
        min_alignment=0.0,
    )


def _is_positive_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def validate_beat_grid_args(tempo, offset, grid_divisions, strength, swing):
    if not _is_positive_number(tempo):
        raise ValueError(
            f"Tempo must be a finite number above 0, got {tempo!r}"
        )
    validate_grid_division(grid_divisions)
    validate_quantize_strength(strength)
    for label, value in (("Offset", offset), ("Swing", swing)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must be a finite number, got {value!r}")


def apply_beat_grid(
    midi_obj, tempo, offset, grid_divisions=4, strength=1.0, swing=0.0
):
    validate_beat_grid_args(tempo, offset, grid_divisions, strength, swing)
    step_size = 60.0 / tempo / grid_divisions

    def get_grid_time(target_time):
        grid_index = round((target_time - offset) / step_size)
        grid_time = offset + grid_index * step_size
        if grid_index % 2 != 0:
            grid_time += step_size * 0.33 * swing
        return grid_time

    for inst in midi_obj.instruments:
        for note in inst.notes:
            duration = note.end - note.start
            start = (
                note.start * (1 - strength)
                + get_grid_time(note.start) * strength
            )
            note.start = max(0.0, start)
            note.end = note.start + duration
        by_pitch = {}
        for note in sorted(inst.notes, key=lambda n: n.start):
            by_pitch.setdefault(note.pitch, []).append(note)
        for notes in by_pitch.values():
            for current, following in zip(notes, notes[1:], strict=False):
                if current.end > following.start:
                    current.end = max(
                        current.start + 0.01, following.start - 0.005
                    )
    return midi_obj


GRID_PHASE_CONFIDENCE = 0.5


def _grid_origin(midi_obj, step):
    starts = []
    weights = []
    for inst in midi_obj.instruments:
        for note in inst.notes:
            starts.append(note.start)
            weights.append(max(float(note.velocity), 1.0))
    if not starts or step <= 0:
        return 0.0
    angles = 2 * math.pi * (np.asarray(starts) % step) / step
    weights = np.asarray(weights)
    cos_sum = float(np.sum(weights * np.cos(angles)))
    sin_sum = float(np.sum(weights * np.sin(angles)))
    if math.hypot(cos_sum, sin_sum) / float(np.sum(weights)) < (
        GRID_PHASE_CONFIDENCE
    ):
        return 0.0
    phase = (math.atan2(sin_sum, cos_sum) % (2 * math.pi)) / (2 * math.pi)
    return phase * step


def smart_quantize(
    midi_obj,
    detected_tempo,
    beat_times,
    grid_divisions=4,
    strength=1.0,
    swing=0.0,
):
    quarter_note_duration = 60.0 / detected_tempo
    step_size = quarter_note_duration / grid_divisions
    if beat_times is not None and len(beat_times) > 0:
        start_offset = beat_times[0]
    else:
        start_offset = _grid_origin(midi_obj, step_size)

    def get_grid_time(target_time):
        relative_time = target_time - start_offset
        grid_index = round(relative_time / step_size)
        if grid_index % 2 != 0:
            swing_offset = (step_size * 0.33) * swing
            final_grid_time = (
                start_offset + (grid_index * step_size) + swing_offset
            )
        else:
            final_grid_time = start_offset + (grid_index * step_size)
        return final_grid_time

    for inst in midi_obj.instruments:
        for note in inst.notes:
            target_start = get_grid_time(note.start)
            note.start = note.start * (1 - strength) + target_start * strength
            target_end = get_grid_time(note.end)
            note.end = note.end * (1 - strength) + target_end * strength
            if note.end <= note.start:
                note.end = note.start + step_size
    return midi_obj


DRUM_PIECES = (
    ("Kick", (35, 36)),
    ("Snare", (37, 38, 39, 40)),
    ("Toms", (41, 43, 45, 47, 48, 50)),
    ("HiHat", (42, 44, 46)),
    ("Cymbals", (49, 51, 52, 53, 55, 57, 59)),
)


def _split_drum_kit(midi_data, stem_name, tempo):
    pieces = {name: [] for name, _ in DRUM_PIECES}
    pieces["Percussion"] = []
    for inst in midi_data.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            target = "Percussion"
            for name, pitches in DRUM_PIECES:
                if note.pitch in pitches:
                    target = name
                    break
            pieces[target].append(note)
    result = []
    for name, notes in pieces.items():
        if not notes:
            continue
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(
            program=0, is_drum=True, name=f"Drums - {name}"
        )
        inst.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        pm.instruments.append(inst)
        result.append((f"{stem_name}_{name}.mid", pm))
    return result


def smart_decompose_stem(midi_data, stem_name, tempo, gm_mapping=False):
    all_notes = [n for i in midi_data.instruments for n in i.notes]
    if "drum" in stem_name.lower():
        if not all_notes:
            return [(f"{stem_name}.mid", midi_data)]
        if any(i.is_drum and i.notes for i in midi_data.instruments):
            return _split_drum_kit(midi_data, stem_name, tempo)
        mapping = GM_DRUM_MAP if gm_mapping else None
        drum_components = {
            "Kick": {
                "notes": [],
                "midi_note": mapping["Kick"] if mapping else 36,
                "range": (0, 45),
            },
            "Toms": {
                "notes": [],
                "midi_note": mapping["Tom Low"] if mapping else 45,
                "range": (45, 55),
            },
            "Snare": {
                "notes": [],
                "midi_note": mapping["Snare"] if mapping else 38,
                "range": (55, 70),
            },
            "HiHat": {
                "notes": [],
                "midi_note": mapping["HiHat Closed"] if mapping else 42,
                "range": (70, 128),
            },
        }
        for note in all_notes:
            for _name, data in drum_components.items():
                low, high = data["range"]
                if low <= note.pitch < high:
                    p = data["midi_note"]
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=p,
                        start=note.start,
                        end=note.end,
                    )
                    data["notes"].append(new_note)
                    break
        res_drums = []
        for name, data in drum_components.items():
            if data["notes"]:
                pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                inst = pretty_midi.Instrument(
                    program=0, is_drum=True, name=f"Drums - {name}"
                )
                inst.notes = data["notes"]
                pm.instruments.append(inst)
                res_drums.append((f"{stem_name}_{name}.mid", pm))
        return res_drums if res_drums else [(f"{stem_name}.mid", midi_data)]
    if len(all_notes) < MIN_NOTES_FOR_SPLIT:
        return [(f"{stem_name}.mid", midi_data)]
    pitches = np.array([n.pitch for n in all_notes]).reshape(-1, 1)
    if pitches.max() - pitches.min() < 12:
        return [(f"{stem_name}.mid", midi_data)]
    try:
        kmeans = _get_sklearn_kmeans()(
            n_clusters=SPLIT_CLUSTERS, n_init=10, random_state=42
        ).fit(pitches)
        centers = kmeans.cluster_centers_.flatten()
        label_map = {old: new for new, old in enumerate(np.argsort(centers))}
        layers = [
            pretty_midi.PrettyMIDI(initial_tempo=tempo)
            for _ in range(SPLIT_CLUSTERS)
        ]
        base_inst = midi_data.instruments[0]
        for l in layers:
            l.instruments.append(
                pretty_midi.Instrument(
                    base_inst.program, base_inst.is_drum, base_inst.name
                )
            )
        for n, lbl in zip(all_notes, kmeans.labels_, strict=False):
            layers[label_map[lbl]].instruments[0].notes.append(n)
        res = []
        for i, l in enumerate(layers):
            if l.instruments[0].notes:
                l = smart_instrument_selector(stem_name, l)
                res.append((f"{stem_name}_{i}.mid", l))
        return res
    except Exception:
        return [(f"{stem_name}.mid", midi_data)]


def merge_same_instruments(midi_data):
    midi_data.instruments = merge_by_instrument(midi_data.instruments)
    return midi_data


def remap_drum_midi(midi_data, target_map_name):
    if target_map_name not in DRUM_MAPPINGS:
        return midi_data
    mapping = DRUM_MAPPINGS[target_map_name]
    for inst in midi_data.instruments:
        if not inst.is_drum:
            continue
        new_notes = []
        for note in inst.notes:
            if note.pitch in mapping:
                note.pitch = mapping[note.pitch]
                new_notes.append(note)
            else:
                new_notes.append(note)
        inst.notes = new_notes
    return midi_data


def merge_glitch_notes(midi_data, threshold=0.05, global_clean=True):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        inst.notes.sort(key=lambda x: (x.start, x.pitch))
        merged = []
        if not inst.notes:
            continue

        curr = inst.notes[0]
        for i in range(1, len(inst.notes)):
            nxt = inst.notes[i]
            if nxt.pitch == curr.pitch and nxt.start < (curr.end - 0.01):
                curr.end = max(curr.end, nxt.end)
                curr.velocity = max(curr.velocity, nxt.velocity)
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)
        inst.notes = merged

    if global_clean:
        all_notes_meta = []
        for inst in midi_data.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                all_notes_meta.append(
                    {
                        "note": note,
                        "inst": inst,
                        "center": note.start + (note.end - note.start) / 2,
                    }
                )

        all_notes_meta.sort(key=lambda x: (x["note"].start, x["note"].pitch))

        notes_to_remove = set()

        for i in range(len(all_notes_meta) - 1):
            curr = all_notes_meta[i]["note"]
            all_notes_meta[i]["inst"]

            for j in range(i + 1, min(i + 10, len(all_notes_meta))):
                nxt = all_notes_meta[j]["note"]
                all_notes_meta[j]["inst"]

                if nxt.start > curr.start + threshold:
                    break

                if (
                    curr.pitch == nxt.pitch
                    and abs(curr.start - nxt.start) < threshold
                ):
                    if curr.velocity >= nxt.velocity:
                        notes_to_remove.add(id(nxt))
                    else:
                        notes_to_remove.add(id(curr))

        for inst in midi_data.instruments:
            inst.notes = [n for n in inst.notes if id(n) not in notes_to_remove]

    return midi_data
