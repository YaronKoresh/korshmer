import math
from dataclasses import dataclass, replace

import numpy as np
import pretty_midi

from korshmer.constants import SCALES

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)

HARMONIC_OFFSETS = (12, 19, 24)
ONSET_WINDOW = 0.08
OVERLAP_SHARE = 0.6

PITCH_RANGES = {
    "bass": (24, 72),
    "vocals": (36, 96),
    "guitar": (36, 96),
    "piano": (21, 108),
}


@dataclass(frozen=True)
class CleanupOptions:
    enabled: bool = True
    merge_reattacks: bool = False
    reattack_gap: float = 0.06
    reattack_ratio: float = 0.9
    merge_runs: bool = True
    run_gap: float = 0.12
    run_touch: float = 0.02
    run_min: int = 3
    run_velocity_rise: float = 1.2
    run_onset_window: float = 0.05
    remove_ghosts: bool | None = None
    ghost_ratio: float = 0.75
    ghost_offsets: tuple = HARMONIC_OFFSETS
    remove_sub_octaves: bool = True
    sub_octave_ratio: float = 0.9
    remove_neighbours: bool = False
    neighbour_ratio: float = 0.6
    velocity_floor: float = 0.0
    snap_to_key: bool = False
    min_key_confidence: float = 0.8
    keep_strong_share: float = 0.9
    keep_long_seconds: float = 0.6
    min_note: float = 0.06
    polyphony_cap: int = 0
    fix_octave_jumps: bool = False
    jump_window: int = 5
    pitch_range: tuple = ()
    close_gaps: float = 0.0
    quantize_division: int = 16
    quantize_tolerance: float = 0.06
    quantize_confidence: float = 0.75
    quantize_ends: bool = True
    smooth_velocity: float = 0.0


def options_for_cleanup(stem_name, base=None):
    base = base or CleanupOptions()
    lower = stem_name.lower()
    for key in ("bass", "vocal", "guitar", "piano"):
        if key in lower:
            name = "vocals" if key == "vocal" else key
            changes = {"pitch_range": PITCH_RANGES[name]}
            if name == "bass":
                changes["remove_sub_octaves"] = False
                if base.remove_ghosts is None:
                    changes["remove_ghosts"] = True
            if name in ("bass", "vocals"):
                changes["merge_runs"] = False
            return replace(base, **changes)
    return base


def _all_notes(midi):
    return [
        n for inst in midi.instruments if not inst.is_drum for n in inst.notes
    ]


def key_profile(notes):
    profile = np.zeros(12)
    for n in notes:
        profile[n.pitch % 12] += (n.end - n.start) * (n.velocity / 127.0)
    return profile


def estimate_key_confidence(notes):
    profile = key_profile(notes)
    if profile.sum() <= 0 or np.std(profile) == 0:
        return 0, "Major", 0.0
    best = (-2.0, 0, "Major")
    for name, ref in (("Major", MAJOR_PROFILE), ("Minor", MINOR_PROFILE)):
        for root in range(12):
            corr = float(np.corrcoef(profile, np.roll(ref, root))[0, 1])
            if corr > best[0]:
                best = (corr, root, name)
    return best[1], best[2], best[0]


def estimate_key(notes):
    root, mode, _ = estimate_key_confidence(notes)
    return root, mode


def _scale_classes(root, mode):
    if mode in SCALES:
        steps = SCALES[mode]
    else:
        steps = MAJOR_STEPS if mode == "Major" else MINOR_STEPS
    return {(root + s) % 12 for s in steps}


def _by_pitch(notes):
    out = {}
    for n in notes:
        out.setdefault(n.pitch, []).append(n)
    return out


def _covered_by(note, by_pitch, offsets, ratio):
    for offset in offsets:
        for other in by_pitch.get(note.pitch - offset, ()):
            if other is note or abs(other.start - note.start) > ONSET_WINDOW:
                continue
            overlap = min(note.end, other.end) - max(note.start, other.start)
            shorter = min(note.end - note.start, other.end - other.start)
            if (
                shorter > 0
                and overlap >= OVERLAP_SHARE * shorter
                and note.velocity < ratio * other.velocity
            ):
                return True
    return False


def _drop_covered(midi, offsets, ratio):
    by_pitch = _by_pitch(_all_notes(midi))
    removed = 0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        kept = [
            n
            for n in inst.notes
            if not _covered_by(n, by_pitch, offsets, ratio)
        ]
        removed += len(inst.notes) - len(kept)
        inst.notes = kept
    return removed


def remove_ghost_notes(midi, ratio=0.75, offsets=HARMONIC_OFFSETS):
    return _drop_covered(midi, offsets, ratio)


def remove_sub_octave_notes(midi, ratio=0.75):
    return _drop_covered(midi, (-12,), ratio)


def remove_neighbour_notes(midi, ratio=0.6):
    return _drop_covered(midi, (1, -1), ratio)


def drop_quiet_notes(midi, share):
    notes = _all_notes(midi)
    if share <= 0 or not notes:
        return 0
    floor = share * float(np.median([n.velocity for n in notes]))
    removed = 0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        kept = [n for n in inst.notes if n.velocity >= floor]
        removed += len(inst.notes) - len(kept)
        inst.notes = kept
    return removed


def snap_weak_notes(midi, allowed, options):
    notes = _all_notes(midi)
    if not notes:
        return 0
    median_velocity = float(np.median([n.velocity for n in notes]))
    weight = np.zeros(12)
    for n in notes:
        weight[n.pitch % 12] += n.end - n.start
    moved = 0
    for n in notes:
        if (n.pitch % 12) in allowed:
            continue
        strong = n.velocity >= options.keep_strong_share * median_velocity
        long = n.end - n.start >= options.keep_long_seconds
        if strong and long:
            continue
        best = None
        for distance in (1, 2):
            candidates = [
                p
                for p in (n.pitch - distance, n.pitch + distance)
                if 0 <= p <= 127 and (p % 12) in allowed
            ]
            if candidates:
                best = max(candidates, key=lambda p: weight[p % 12])
                break
        if best is not None:
            n.pitch = best
            moved += 1
    return moved


def merge_fragments(inst, gap, min_note, ratio=None):
    merged = []
    last_by_pitch = {}
    for n in sorted(inst.notes, key=lambda n: (n.pitch, n.start)):
        last = last_by_pitch.get(n.pitch)
        if last is not None:
            distance = n.start - last.end
            join = distance < 0 or (
                distance < gap
                and (ratio is None or n.velocity <= ratio * last.velocity)
            )
            if join:
                last.end = max(last.end, n.end)
                last.velocity = max(last.velocity, n.velocity)
                continue
        last_by_pitch[n.pitch] = n
        merged.append(n)
    before = len(inst.notes)
    inst.notes = sorted(
        (n for n in merged if n.end - n.start >= min_note),
        key=lambda n: (n.start, n.pitch),
    )
    return before - len(inst.notes)


def merge_note_runs(
    inst, gap, min_run, rise, touch=0.02, onsets=None, window=0.05
):
    if min_run < 2:
        return 0
    attacks = np.sort(np.asarray([] if onsets is None else onsets, float))

    def attack_between(a, b):
        index = np.searchsorted(attacks, a.end - window, side="left")
        return index < len(attacks) and attacks[index] <= b.start + window

    by_pitch = {}
    for n in sorted(inst.notes, key=lambda n: n.start):
        by_pitch.setdefault(n.pitch, []).append(n)
    drop = set()
    for notes in by_pitch.values():
        run = [notes[0]]
        for n in notes[1:] + [None]:
            if (
                n is not None
                and touch <= n.start - run[-1].end < gap
                and n.velocity <= rise * run[-1].velocity
                and not attack_between(run[-1], n)
            ):
                run.append(n)
                continue
            if len(run) >= min_run:
                head = run[0]
                head.end = max(r.end for r in run)
                head.velocity = max(r.velocity for r in run)
                drop.update(id(r) for r in run[1:])
            run = [n]
    inst.notes = [n for n in inst.notes if id(n) not in drop]
    return len(drop)


def split_notes_at_onsets(midi, onset_times, min_part=0.08, gap=0.0):
    cuts = np.sort(np.asarray(onset_times, dtype=float))
    added = 0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        notes = []
        for n in inst.notes:
            lo = np.searchsorted(cuts, n.start + min_part, side="left")
            hi = np.searchsorted(cuts, n.end - min_part, side="right")
            points = [n.start, *cuts[lo:hi].tolist(), n.end]
            for index, (a, b) in enumerate(
                zip(points, points[1:], strict=False)
            ):
                end = b - gap if index < len(points) - 2 else b
                notes.append(pretty_midi.Note(n.velocity, n.pitch, a, end))
            added += len(points) - 2
        inst.notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    return added


def drop_notes_outside_regions(midi, regions, margin=0.25):
    spans = []
    for a, b in sorted(
        (float(a) - margin, float(b) + margin) for a, b in regions
    ):
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    removed = 0
    for inst in midi.instruments:
        kept = []
        for n in inst.notes:
            span = next((s for s in spans if s[0] <= n.start <= s[1]), None)
            if span is None:
                removed += 1
                continue
            n.end = max(min(n.end, span[1]), n.start + 0.01)
            kept.append(n)
        inst.notes = kept
    return removed


def clamp_pitch_range(inst, low, high):
    before = len(inst.notes)
    inst.notes = [n for n in inst.notes if low <= n.pitch <= high]
    return before - len(inst.notes)


def correct_octave_jumps(inst, window=5):
    notes = sorted(inst.notes, key=lambda n: n.start)
    pitches = [n.pitch for n in notes]
    moved = 0
    half = max(1, window // 2)
    for i, n in enumerate(notes):
        around = pitches[max(0, i - half) : i] + pitches[i + 1 : i + 1 + half]
        if len(around) < 2:
            continue
        median = float(np.median(around))
        shift = int(round((median - n.pitch) / 12.0)) * 12
        if shift and abs(n.pitch - median) >= 12:
            new = n.pitch + shift
            if 0 <= new <= 127:
                n.pitch = new
                pitches[i] = new
                moved += 1
    return moved


def cap_polyphony(inst, cap):
    if cap <= 0:
        return 0
    notes = sorted(inst.notes, key=lambda n: (n.start, -n.velocity))
    kept = []
    active = []
    for n in notes:
        active = [a for a in active if a.end > n.start]
        if len(active) < cap:
            active.append(n)
            kept.append(n)
            continue
        weakest = min(active, key=lambda a: a.velocity)
        if n.velocity > weakest.velocity:
            weakest.end = max(weakest.start, n.start)
            active.remove(weakest)
            active.append(n)
            kept.append(n)
    removed = len(inst.notes) - len(kept)
    inst.notes = sorted(
        (n for n in kept if n.end > n.start), key=lambda n: (n.start, n.pitch)
    )
    return removed


def close_note_gaps(inst, max_gap):
    if max_gap <= 0:
        return 0
    notes = sorted(inst.notes, key=lambda n: n.start)
    changed = 0
    for a, b in zip(notes, notes[1:], strict=False):
        if 0 < b.start - a.end <= max_gap:
            a.end = b.start
            changed += 1
    return changed


def grid_confidence(times, step):
    if len(times) < 4 or step <= 0:
        return 0.0, 0.0
    angles = 2 * np.pi * (np.asarray(times) % step) / step
    vector = np.mean(np.exp(1j * angles))
    phase = (np.angle(vector) % (2 * np.pi)) / (2 * np.pi) * step
    return float(phase), float(np.abs(vector))


def quantize_light(midi, tempo, division, tolerance, confidence, ends=True):
    notes = _all_notes(midi)
    if division <= 0 or tempo <= 0 or not notes:
        return 0
    step = (60.0 / tempo) * 4.0 / division
    tolerance = min(tolerance, 0.45 * step)
    phase, strength = grid_confidence([n.start for n in notes], step)
    if strength < confidence:
        return 0

    def snap(t):
        nearest = round((t - phase) / step) * step + phase
        return nearest if abs(nearest - t) <= tolerance else t

    def snap_end(t):
        before = math.floor((t - phase) / step + 1e-6) * step + phase
        return before if t - before <= tolerance else t

    moved = 0
    for n in notes:
        start = snap(n.start)
        end = snap_end(n.end) if ends else n.end
        if end <= start:
            end = n.end
        if start != n.start or end != n.end:
            moved += 1
        n.start, n.end = max(0.0, start), max(0.0, end)
    return moved


def smooth_note_velocity(inst, strength):
    strength = float(np.clip(strength, 0.0, 1.0))
    notes = sorted(inst.notes, key=lambda n: n.start)
    if strength == 0 or len(notes) < 3:
        return 0
    velocities = np.array([n.velocity for n in notes], dtype=float)
    for i, n in enumerate(notes):
        lo, hi = max(0, i - 2), min(len(notes), i + 3)
        local = float(np.median(velocities[lo:hi]))
        n.velocity = int(
            np.clip(
                round((1 - strength) * velocities[i] + strength * local), 1, 127
            )
        )
    return len(notes)


def merge_note_variants(midis, tolerance=0.08, need=2, *, keep_seconds=0.4):
    first = midis[0]
    tempi = first.get_tempo_changes()[1]
    result = pretty_midi.PrettyMIDI(
        initial_tempo=float(tempi[0]) if len(tempi) else 120.0
    )
    others = [_by_pitch(_all_notes(other)) for other in midis[1:]]
    velocities = [n.velocity for n in _all_notes(first)]
    median = float(np.median(velocities)) if velocities else 0.0

    def keeps(n):
        votes = 1 + sum(
            any(
                abs(m.start - n.start) <= tolerance
                for m in table.get(n.pitch, ())
            )
            for table in others
        )
        return votes >= need or (
            keep_seconds is not None
            and n.end - n.start >= keep_seconds
            and n.velocity > median
        )

    for inst in first.instruments:
        copy = _copy_instrument(inst)
        if not inst.is_drum:
            copy.notes = [
                pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
                for n in inst.notes
                if keeps(n)
            ]
        result.instruments.append(copy)
    return result


def clean_transcription(midi, key=None, options=None, onsets=None):
    options = options or CleanupOptions()
    tempi = midi.get_tempo_changes()[1]
    tempo = float(tempi[0]) if len(tempi) else 120.0
    result = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    result.instruments = [_copy_instrument(inst) for inst in midi.instruments]
    report = {}
    if not options.enabled or not _all_notes(result):
        return result, report
    steps = []
    if options.merge_reattacks:
        steps.append(("fragments_merged", _merge_all, (options,)))
    if options.merge_runs:
        steps.append(("runs_merged", _merge_runs, (options, onsets)))
    if options.remove_ghosts:
        steps.append(
            (
                "ghosts_removed",
                lambda m, o: remove_ghost_notes(
                    m, o.ghost_ratio, o.ghost_offsets
                ),
                (options,),
            )
        )
    if options.remove_sub_octaves:
        steps.append(
            (
                "sub_octaves_removed",
                lambda m, o: remove_sub_octave_notes(m, o.sub_octave_ratio),
                (options,),
            )
        )
    if options.remove_neighbours:
        steps.append(
            (
                "neighbours_removed",
                lambda m, o: remove_neighbour_notes(m, o.neighbour_ratio),
                (options,),
            )
        )
    if options.velocity_floor > 0:
        steps.append(
            (
                "quiet_removed",
                lambda m, o: drop_quiet_notes(m, o.velocity_floor),
                (options,),
            )
        )
    for label, func, args in steps:
        report[label] = func(result, *args)
    if options.pitch_range:
        low, high = options.pitch_range
        report["out_of_range"] = sum(
            clamp_pitch_range(i, low, high)
            for i in result.instruments
            if not i.is_drum
        )
    if options.fix_octave_jumps:
        report["octave_jumps"] = sum(
            correct_octave_jumps(i, options.jump_window)
            for i in result.instruments
            if not i.is_drum
        )
    if options.snap_to_key:
        notes = _all_notes(result)
        if key is not None:
            root, mode, confidence = key[0], key[1], 1.0
        else:
            root, mode, confidence = estimate_key_confidence(notes)
        report["key"] = (root, mode)
        report["key_confidence"] = confidence
        if confidence >= options.min_key_confidence:
            report["notes_snapped"] = snap_weak_notes(
                result, _scale_classes(root, mode), options
            )
    if options.polyphony_cap > 0:
        report["polyphony_trimmed"] = sum(
            cap_polyphony(i, options.polyphony_cap)
            for i in result.instruments
            if not i.is_drum
        )
    if options.close_gaps > 0:
        report["gaps_closed"] = sum(
            close_note_gaps(i, options.close_gaps)
            for i in result.instruments
            if not i.is_drum
        )
    if options.quantize_division > 0:
        report["quantized"] = quantize_light(
            result,
            tempo,
            options.quantize_division,
            options.quantize_tolerance,
            options.quantize_confidence,
            options.quantize_ends,
        )
    if options.smooth_velocity > 0:
        for i in result.instruments:
            if not i.is_drum:
                smooth_note_velocity(i, options.smooth_velocity)
    for inst in result.instruments:
        if not inst.is_drum:
            inst.notes = [
                n for n in inst.notes if n.end - n.start >= options.min_note
            ]
    return result, report


def _merge_all(midi, options):
    return sum(
        merge_fragments(
            inst, options.reattack_gap, options.min_note, options.reattack_ratio
        )
        for inst in midi.instruments
        if not inst.is_drum
    )


def _merge_runs(midi, options, onsets):
    return sum(
        merge_note_runs(
            inst,
            options.run_gap,
            options.run_min,
            options.run_velocity_rise,
            options.run_touch,
            onsets,
            options.run_onset_window,
        )
        for inst in midi.instruments
        if not inst.is_drum
    )


def _copy_instrument(inst):
    out = pretty_midi.Instrument(
        program=inst.program, is_drum=inst.is_drum, name=inst.name
    )
    out.notes = [
        pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
        for n in inst.notes
    ]
    out.pitch_bends = list(inst.pitch_bends)
    out.control_changes = list(inst.control_changes)
    return out
