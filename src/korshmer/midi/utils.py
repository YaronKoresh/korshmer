import copy
import json
import os

import numpy as np
import pretty_midi
from PIL import Image, ImageOps

from korshmer.utils.track_layout import merge_by_instrument


def remove_short_notes(midi_data, min_duration=None, min_ms=50):
    midi_data = copy.deepcopy(midi_data)
    min_sec = min_duration if min_duration is not None else (min_ms / 1000.0)
    for inst in midi_data.instruments:
        inst.notes = [n for n in inst.notes if (n.end - n.start) >= min_sec]
    return midi_data


def generate_piano_roll_image(midi_or_path, output_path=None):
    if isinstance(midi_or_path, str):
        pm = pretty_midi.PrettyMIDI(midi_or_path)
    else:
        pm = midi_or_path

    piano_roll = pm.get_piano_roll(fs=100)
    if piano_roll.size == 0:
        return None
    piano_roll = np.flipud(piano_roll)
    min_nonzero = 0
    max_nonzero = 127
    rows_with_notes = np.any(piano_roll > 0, axis=1)
    if np.any(rows_with_notes):
        y_indices = np.where(rows_with_notes)[0]
        min_nonzero = y_indices[0]
        max_nonzero = y_indices[-1]
    margin = 5
    start_row = max(0, min_nonzero - margin)
    end_row = min(128, max_nonzero + margin)
    cropped = piano_roll[start_row:end_row, :]
    norm = 255 * (cropped > 0)
    img_data = norm.astype(np.uint8)
    img = Image.fromarray(img_data, mode="L")
    if img.width > 3000:
        img = img.resize((3000, img.height), resample=Image.Resampling.BOX)
    img = img.resize((img.width, img.width // 3), resample=Image.NEAREST)
    img = img.convert("RGB")
    img = ImageOps.colorize(img.convert("L"), black="#0f0c29", white="#00f2ff")
    img.save(output_path)
    return output_path


def remove_empty_bars(midi_data, gap_threshold=2.0):
    midi_data = copy.deepcopy(midi_data)
    notes = []
    for inst in midi_data.instruments:
        notes.extend(inst.notes)
    if not notes:
        return midi_data
    notes.sort(key=lambda x: x.start)
    shift_map = []
    latest_end = notes[0].end
    for i in range(len(notes) - 1):
        latest_end = max(latest_end, notes[i].end)
        gap = notes[i + 1].start - latest_end
        if gap > gap_threshold:
            shift_amount = gap - 0.5
            shift_map.append((notes[i + 1].start, shift_amount))

    def event_shift(time):
        total = 0.0
        for resume, amount in shift_map:
            cut_from = resume - amount
            total += min(amount, max(0.0, time - cut_from))
        return total

    for inst in midi_data.instruments:
        for note in inst.notes:
            total_shift = sum(amt for t, amt in shift_map if note.start >= t)
            note.start -= total_shift
            note.end -= total_shift
        for event in (*inst.control_changes, *inst.pitch_bends):
            event.time -= event_shift(event.time)
    for event_list in (
        midi_data.time_signature_changes,
        midi_data.key_signature_changes,
        midi_data.lyrics,
    ):
        for event in event_list:
            event.time -= event_shift(event.time)
    return midi_data


def export_to_json(midi_or_path, output_path=None, bpm=120.0):
    if isinstance(midi_or_path, str):
        pm = pretty_midi.PrettyMIDI(midi_or_path)
    else:
        pm = midi_or_path

    out = {"bpm": bpm, "tracks": []}
    for inst in pm.instruments:
        track_data = {
            "name": str(inst.name),
            "program": int(inst.program),
            "is_drum": bool(inst.is_drum),
            "notes": [],
        }
        for n in inst.notes:
            track_data["notes"].append(
                {
                    "pitch": int(n.pitch),
                    "start": float(n.start),
                    "end": float(n.end),
                    "velocity": int(n.velocity),
                }
            )
        out["tracks"].append(track_data)

    if output_path is None:
        return json.dumps(out, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return output_path


def save_midi(midi_data, original_path, suffix):
    dir_name = os.path.dirname(original_path)
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    new_name = f"{base_name}_{suffix}.mid"
    new_path = os.path.join(dir_name, new_name)
    midi_data.instruments = merge_by_instrument(
        midi_data.instruments, keep_shared_names=True
    )
    midi_data.write(new_path)
    return new_path
