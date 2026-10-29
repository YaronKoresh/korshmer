import copy

import pretty_midi

TARGETS = ("Everything", "Chords", "Lead")
CHORD_SHARE = 0.3
CHORD_WINDOW = 0.05
REPEAT_GAP = 0.04
LEGATO_GAP = 0.12
MIN_NOTE = 0.01


def classify(inst):
    notes = sorted(inst.notes, key=lambda n: n.start)
    if len(notes) < 2:
        return "lead"
    stacked = 0
    for i, note in enumerate(notes):
        near = (i > 0 and note.start - notes[i - 1].start < CHORD_WINDOW) or (
            i + 1 < len(notes)
            and notes[i + 1].start - note.start < CHORD_WINDOW
        )
        stacked += 1 if near else 0
    return "chords" if stacked / len(notes) >= CHORD_SHARE else "lead"


def _merge_repeats(notes):
    merged = []
    by_pitch = {}
    for note in sorted(notes, key=lambda n: (n.pitch, n.start)):
        last = by_pitch.get(note.pitch)
        if last is not None and note.start - last.end < REPEAT_GAP:
            last.end = max(last.end, note.end)
            last.velocity = max(last.velocity, note.velocity)
            continue
        by_pitch[note.pitch] = note
        merged.append(note)
    return sorted(merged, key=lambda n: (n.start, n.pitch))


def _close_lead(notes):
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    adjusted = 0
    for note, nxt in zip(notes, notes[1:], strict=False):
        if nxt.start - note.start < MIN_NOTE:
            continue
        gap = nxt.start - note.end
        if gap < 0 or gap <= LEGATO_GAP:
            if abs(note.end - nxt.start) > 1e-9:
                note.end = nxt.start
                adjusted += 1
    return notes, adjusted


def _smooth_chords(instruments):
    from korshmer.midi_algorithms import apply_voice_leading

    before = [[n.pitch for n in inst.notes] for inst in instruments]
    holder = pretty_midi.PrettyMIDI()
    holder.instruments = instruments
    apply_voice_leading(holder)
    moved = 0
    for pitches, inst in zip(before, instruments, strict=True):
        moved += sum(
            1 for a, n in zip(pitches, inst.notes, strict=True) if a != n.pitch
        )
    return moved


def polish_midi(midi_data, target="Everything", min_ms=50):
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    pm = copy.deepcopy(midi_data)
    min_sec = max(0.0, float(min_ms)) / 1000.0
    report = {
        "short_removed": 0,
        "repeats_merged": 0,
        "chord_notes_moved": 0,
        "lead_notes_joined": 0,
        "chord_tracks": 0,
        "lead_tracks": 0,
    }
    chord_insts = []
    for inst in pm.instruments:
        if inst.is_drum or not inst.notes:
            continue
        kind = classify(inst)
        if (target == "Chords" and kind != "chords") or (
            target == "Lead" and kind != "lead"
        ):
            continue
        before = len(inst.notes)
        inst.notes = [n for n in inst.notes if n.end - n.start >= min_sec]
        report["short_removed"] += before - len(inst.notes)
        before = len(inst.notes)
        inst.notes = _merge_repeats(inst.notes)
        report["repeats_merged"] += before - len(inst.notes)
        if kind == "chords":
            report["chord_tracks"] += 1
            chord_insts.append(inst)
        else:
            report["lead_tracks"] += 1
            inst.notes, joined = _close_lead(inst.notes)
            report["lead_notes_joined"] += joined
    if chord_insts:
        report["chord_notes_moved"] = _smooth_chords(chord_insts)
    return pm, report
