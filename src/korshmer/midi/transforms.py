import bisect
import copy
import math
import numbers
import random


def _validate_whole_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def transpose_midi(midi_data, semitones):
    semitones = _validate_whole_number(semitones, "Semitones")
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = max(0, min(127, n.pitch + semitones))
    return midi_data


def lofi_velocity(midi_data, bits=4):
    bits = _validate_whole_number(bits, "Bits")
    if not 1 <= bits <= 7:
        raise ValueError(f"Bits must be between 1 and 7, got {bits}")
    levels = 2**bits
    step = 127 / levels
    for inst in midi_data.instruments:
        for n in inst.notes:
            q = round(n.velocity / step) * step
            n.velocity = int(max(1, min(127, q)))
    return midi_data


def generate_legato(midi_data):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        starts = sorted({n.start for n in inst.notes})
        for n in inst.notes:
            idx = bisect.bisect_right(starts, n.start)
            if idx < len(starts) and n.end < starts[idx]:
                n.end = starts[idx]
    return midi_data


STRUM_DIRECTIONS = ("Down", "Up")
STRUM_CURVES = ("Linear", "Accelerate")


def _validate_strum_delay(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Strum delay must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"Strum delay must be a finite number of 0 or more, got {value!r}"
        )
    return result


def apply_strumming(
    midi_data, delay_sec=0.03, direction="Down", curve="Linear"
):
    delay_sec = _validate_strum_delay(delay_sec)
    if direction not in STRUM_DIRECTIONS:
        raise ValueError(
            f"Strum direction must be one of {STRUM_DIRECTIONS}, "
            f"got {direction!r}"
        )
    if curve not in STRUM_CURVES:
        raise ValueError(
            f"Strum curve must be one of {STRUM_CURVES}, got {curve!r}"
        )
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda x: x.start)
        chord_map = {}
        for n in notes:
            q = round(n.start * 20) / 20.0
            if q not in chord_map:
                chord_map[q] = []
            chord_map[q].append(n)
        for t in chord_map:
            cluster = chord_map[t]
            if len(cluster) > 1:
                cluster.sort(key=lambda x: x.pitch, reverse=(direction == "Up"))
                len(cluster)
                for i, n in enumerate(cluster):
                    factor = i
                    if curve == "Accelerate":
                        factor = i * i * 0.5
                    shift = factor * delay_sec
                    n.start += shift
                    n.end += shift
        inst.notes.sort(key=lambda x: x.start)
    return midi_data


def apply_crescendo(midi_data, start_vel=None, end_vel=None):
    if start_vel is None:
        start_vel = 40
    if end_vel is None:
        end_vel = 120
    start_vel = _clamp_velocity_bound(start_vel)
    end_vel = _clamp_velocity_bound(end_vel)

    total_start = min(
        (n.start for i in midi_data.instruments for n in i.notes), default=0
    )
    total_end = max(
        (n.end for i in midi_data.instruments for n in i.notes), default=1
    )
    duration = total_end - total_start
    if duration == 0:
        duration = 1
    for inst in midi_data.instruments:
        for n in inst.notes:
            pos = (n.start - total_start) / duration
            new_vel = start_vel + (end_vel - start_vel) * pos
            n.velocity = int(max(1, min(127, new_vel)))
    return midi_data


def apply_retrograde(midi_data):
    total_end = max(
        (
            t
            for i in midi_data.instruments
            for t in (
                [n.end for n in i.notes]
                + [c.time for c in i.control_changes]
                + [b.time for b in i.pitch_bends]
            )
        ),
        default=0,
    )
    for inst in midi_data.instruments:
        for n in inst.notes:
            old_start = n.start
            old_end = n.end
            n.start = total_end - old_end
            n.end = total_end - old_start
        for event in inst.control_changes:
            event.time = total_end - event.time
        for event in inst.pitch_bends:
            event.time = total_end - event.time
        inst.control_changes.sort(key=lambda x: x.time)
        inst.pitch_bends.sort(key=lambda x: x.time)
    return midi_data


def apply_mirror(midi_data, center_pitch=60):
    center_pitch = _validate_whole_number(center_pitch, "Center pitch")
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = max(0, min(127, center_pitch - (n.pitch - center_pitch)))
    return midi_data


def apply_drop2(midi_data):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda x: x.start)
        chord_map = {}
        for n in notes:
            q = round(n.start * 20) / 20.0
            if q not in chord_map:
                chord_map[q] = []
            chord_map[q].append(n)
        for t in chord_map:
            cluster = chord_map[t]
            if len(cluster) >= 3:
                cluster.sort(key=lambda x: x.pitch)
                cluster[-2].pitch = max(0, cluster[-2].pitch - 12)
    return midi_data


def limit_pitch_range(
    midi_data, min_pitch=None, max_pitch=None, min_p=None, max_p=None
):

    if min_pitch is not None:
        min_val = min_pitch
    elif min_p is not None:
        min_val = min_p
    else:
        min_val = 0

    if max_pitch is not None:
        max_val = max_pitch
    elif max_p is not None:
        max_val = max_p
    else:
        max_val = 127

    min_val = _validate_whole_number(min_val, "Lowest pitch")
    max_val = _validate_whole_number(max_val, "Highest pitch")
    if min_val > max_val:
        raise ValueError(
            f"Lowest pitch {min_val} is above the highest pitch {max_val}"
        )

    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes_to_remove = []
        for n in inst.notes:
            if n.pitch < min_val or n.pitch > max_val:
                notes_to_remove.append(n)
        for n in notes_to_remove:
            inst.notes.remove(n)
    return midi_data


def randomize_velocity(midi_data, amount=None, min_v=None, max_v=None):
    if amount is not None:
        amount = _validate_whole_number(amount, "Velocity amount")
        if amount < 0:
            raise ValueError(
                f"Velocity amount must be 0 or more, got {amount!r}"
            )
        for inst in midi_data.instruments:
            for n in inst.notes:
                variation = random.randint(-amount, amount)
                n.velocity = max(1, min(127, n.velocity + variation))
    else:
        min_vel = _clamp_velocity_bound(min_v if min_v is not None else 1)
        max_vel = _clamp_velocity_bound(max_v if max_v is not None else 127)
        _check_velocity_order(min_vel, max_vel)
        for inst in midi_data.instruments:
            for n in inst.notes:
                n.velocity = random.randint(min_vel, max_vel)
    return midi_data


def extract_rhythm(midi_data, pitch=60):
    pitch = _validate_whole_number(pitch, "Rhythm pitch")
    if not 0 <= pitch <= 127:
        raise ValueError(f"Rhythm pitch must be 0 to 127, got {pitch!r}")
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = pitch
    return midi_data


def _validate_stretch_factor(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            import numpy as np

            numeric = isinstance(value, (np.integer, np.floating))
        except ImportError:
            numeric = False
        if not numeric:
            raise ValueError(
                f"Stretch factor must be a positive number, got {value!r}"
            )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(
            f"Stretch factor must be a finite number above 0, got {value!r}"
        )
    return result


def time_stretch(midi_data, factor=None, ratio=None):
    stretch_factor = _validate_stretch_factor(
        factor if factor is not None else (ratio if ratio is not None else 1.0)
    )
    midi_data = copy.deepcopy(midi_data)
    for inst in midi_data.instruments:
        for n in inst.notes:
            n.start *= stretch_factor
            n.end *= stretch_factor
        for cc in inst.control_changes:
            cc.time *= stretch_factor
        for bend in inst.pitch_bends:
            bend.time *= stretch_factor
    for event_list in (
        midi_data.time_signature_changes,
        midi_data.key_signature_changes,
        midi_data.lyrics,
    ):
        for event in event_list:
            event.time *= stretch_factor
    return midi_data


def _clamp_velocity_bound(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Velocity bound must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"Velocity bound must be finite, got {value!r}")
    return int(max(1, min(127, round(value))))


def _check_velocity_order(min_vel, max_vel):
    if min_vel > max_vel:
        raise ValueError(
            f"Lowest velocity {min_vel} is above the highest velocity {max_vel}"
        )


def compress_velocity(midi_data, ratio=None, min_vel=40, max_vel=110):
    if ratio is not None:
        if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
            raise ValueError(
                f"Compression ratio must be a number, got {ratio!r}"
            )
        if not math.isfinite(ratio):
            raise ValueError(f"Compression ratio must be finite, got {ratio!r}")
    if ratio is None:
        min_vel = _clamp_velocity_bound(min_vel)
        max_vel = _clamp_velocity_bound(max_vel)
        _check_velocity_order(min_vel, max_vel)
    if ratio is not None:
        for inst in midi_data.instruments:
            for n in inst.notes:
                midpoint = 64
                distance = n.velocity - midpoint
                n.velocity = int(max(1, min(127, midpoint + distance * ratio)))
    else:
        for inst in midi_data.instruments:
            for n in inst.notes:
                old_range = 127 - 1
                new_range = max_vel - min_vel
                norm = (n.velocity - 1) / old_range
                n.velocity = int(min_vel + norm * new_range)
    return midi_data


def double_octave(midi_data, shift_octaves=-1):
    shift_octaves = _validate_whole_number(shift_octaves, "Octave shift")
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        extra_notes = []
        shift = 12 * shift_octaves
        for n in inst.notes:
            target = n.pitch + shift
            if shift == 0 or not 0 <= target <= 127:
                continue
            new_note = copy.deepcopy(n)
            new_note.pitch = target
            extra_notes.append(new_note)
        inst.notes.extend(extra_notes)
    return midi_data


_MONO_CLUSTER_WINDOW = 0.05


def force_monophonic(midi_data, mode="High"):
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        clusters = []
        for n in notes:
            if (
                clusters
                and n.start - clusters[-1][0].start < _MONO_CLUSTER_WINDOW
            ):
                clusters[-1].append(n)
            else:
                clusters.append([n])
        keep_notes = []
        for cluster in clusters:
            if mode == "High":
                chosen = max(cluster, key=lambda x: x.pitch)
            else:
                chosen = min(cluster, key=lambda x: x.pitch)
            keep_notes.append(chosen)
        for current, following in zip(keep_notes, keep_notes[1:], strict=False):
            if current.end > following.start:
                current.end = following.start
        inst.notes = keep_notes
    return midi_data
