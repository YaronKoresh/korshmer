import os

import pretty_midi

from korshmer.analysis import export_scale_stats, suggest_next_chord
from korshmer.constants import GRID_RESOLUTIONS, NOTE_NAMES
from korshmer.midi import (
    apply_crescendo,
    apply_drop2,
    apply_mirror,
    apply_retrograde,
    apply_strumming,
    compress_velocity,
    double_octave,
    export_to_json,
    extract_rhythm,
    force_monophonic,
    generate_legato,
    limit_pitch_range,
    lofi_velocity,
    randomize_velocity,
    remove_empty_bars,
    remove_short_notes,
    time_stretch,
    transpose_midi,
)
from korshmer.midi_algorithms import (
    _save_modified_midi,
    add_drum_fill,
    add_ghost_notes,
    apply_arpeggiator,
    apply_density_thin,
    apply_groove_template,
    apply_humanize,
    apply_legato_staccato,
    apply_lfo_modulation,
    apply_microtuning,
    apply_midi_delay,
    apply_open_voicing,
    apply_ratchet,
    apply_scale_snap,
    apply_smart_inversions,
    apply_smart_loop,
    apply_velocity_ramp,
    apply_voice_leading,
    generate_chord_progression,
    generate_click_track,
    generate_euclidean_rhythm,
    generate_melody_algo,
    generate_melody_continuation,
    generate_pad_layer,
    generate_polyrhythms,
    generate_simple_bass,
    generate_smart_harmony,
    generate_standard_beat,
    invert_velocity,
    merge_glitch_notes,
    remap_drum_midi,
    smart_quantize,
    split_by_pitch,
)
from korshmer.utils.midi_loading import load_pretty_midi
from korshmer.utils.track_layout import merge_by_instrument


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


def process_smart_harmony(midi_path, root_name, scale_name, harmony_type):
    pm = load_pretty_midi(midi_path)
    if root_name == "Auto" or scale_name == "Auto (Detect)":
        return None
    root_idx = NOTE_NAMES.index(root_name)
    pm = generate_smart_harmony(pm, root_idx, scale_name, harmony_type)
    return _save_modified_midi(pm, midi_path, f"Harmony_{harmony_type}")


def process_arpeggiator(midi_path, pattern, rate, bpm=120.0):
    pm = load_pretty_midi(midi_path)
    pm = apply_arpeggiator(pm, pattern, rate, bpm=bpm)
    return _save_modified_midi(pm, midi_path, f"Arp_{pattern}")


def process_humanize(midi_path, time_amt, vel_amt):
    pm = load_pretty_midi(midi_path)
    pm = apply_humanize(pm, time_amt, vel_amt)
    return _save_modified_midi(pm, midi_path, "Humanized")


def process_scale_snap(
    midi_path,
    root_name,
    scale_name,
    grid_str="1/8",
    strength=1.0,
    swing=0.0,
    bpm=120.0,
):
    pm = load_pretty_midi(midi_path)

    if root_name != "Auto" and "Auto" not in scale_name:
        try:
            root_idx = NOTE_NAMES.index(root_name)
            pm = apply_scale_snap(pm, root_idx, scale_name)
        except ValueError:
            pass

    if strength > 0:
        grid_div = GRID_RESOLUTIONS.get(grid_str, 2)

        pm = smart_quantize(
            pm,
            detected_tempo=bpm,
            beat_times=None,
            grid_divisions=grid_div,
            strength=strength,
            swing=swing,
        )

    return _save_modified_midi(pm, midi_path, "Snap")


def process_inversions(midi_path):
    pm = load_pretty_midi(midi_path)
    pm = apply_smart_inversions(pm)
    return _save_modified_midi(pm, midi_path, "Inversions")


def process_legato_staccato(midi_path, mode, val):
    pm = load_pretty_midi(midi_path)
    pm = apply_legato_staccato(pm, mode, val)
    return _save_modified_midi(pm, midi_path, mode)


def process_split_pitch(midi_path, split_point):
    pm = load_pretty_midi(midi_path)
    pm = split_by_pitch(pm, split_point)
    return _save_modified_midi(pm, midi_path, "Split", one_track_each=False)


def process_velocity_compress(midi_path, min_v, max_v):
    pm = load_pretty_midi(midi_path)
    pm = compress_velocity(pm, min_vel=min_v, max_vel=max_v)
    return _save_modified_midi(pm, midi_path, "Comp")


def process_drum_remap(midi_path, map_name):
    pm = load_pretty_midi(midi_path)
    pm = remap_drum_midi(pm, map_name)
    return _save_modified_midi(pm, midi_path, f"Map_{map_name.split()[0]}")


def process_doubling(midi_path, octave):
    pm = load_pretty_midi(midi_path)
    pm = double_octave(pm, octave)
    return _save_modified_midi(pm, midi_path, "Double")


def get_chord_suggestion(midi_path):
    pm = load_pretty_midi(midi_path)
    return suggest_next_chord(pm)


def process_bass_gen(path, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_simple_bass(pm, bpm)
    return _save_modified_midi(pm, path, "BassGen")


def process_drum_fill(path, bars, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = add_drum_fill(pm, bars, bpm)
    return _save_modified_midi(pm, path, "Fills")


def process_sustain(path):
    pm = load_pretty_midi(path)
    pm = generate_sustain_pedal(pm)
    return _save_modified_midi(pm, path, "Sustain")


def process_ghosts(path, prob, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = add_ghost_notes(pm, prob, bpm)
    return _save_modified_midi(pm, path, "Ghost")


def process_open_voice(path):
    pm = load_pretty_midi(path)
    pm = apply_open_voicing(pm)
    return _save_modified_midi(pm, path, "Open")


def process_mono(path, mode):
    pm = load_pretty_midi(path)
    pm = force_monophonic(pm, mode)
    return _save_modified_midi(pm, path, f"Mono{mode}")


def process_polish(path, target="Everything", min_ms=50):
    from korshmer.midi.polish import polish_midi

    pm = load_pretty_midi(path)
    pm, _report = polish_midi(pm, target, min_ms)
    return _save_modified_midi(pm, path, f"Polish_{target}")


def process_variation(path, strength="Medium", seed=0):
    from korshmer.midi.variation import create_variation

    pm = load_pretty_midi(path)
    pm, _report = create_variation(pm, strength, seed)
    return _save_modified_midi(pm, path, f"Variation_{strength}_{seed}")


def process_cleanup(path, ms):
    pm = load_pretty_midi(path)
    pm = remove_short_notes(pm, min_ms=ms)
    return _save_modified_midi(pm, path, "Clean")


def process_transpose(path, semitones):
    pm = load_pretty_midi(path)
    pm = transpose_midi(pm, semitones)
    return _save_modified_midi(pm, path, f"Trans{semitones}")


def process_lofi(path):
    pm = load_pretty_midi(path)
    pm = lofi_velocity(pm)
    return _save_modified_midi(pm, path, "LoFi")


def process_legato(path):
    pm = load_pretty_midi(path)
    pm = generate_legato(pm)
    return _save_modified_midi(pm, path, "Legato")


def process_strum(path, speed, direction, curve="Linear"):
    pm = load_pretty_midi(path)
    pm = apply_strumming(pm, speed, direction, curve)
    return _save_modified_midi(pm, path, "Strum")


def process_crescendo(path, v_start, v_end):
    pm = load_pretty_midi(path)
    pm = apply_crescendo(pm, v_start, v_end)
    return _save_modified_midi(pm, path, "Cresc")


def process_retrograde(path):
    pm = load_pretty_midi(path)
    pm = apply_retrograde(pm)
    return _save_modified_midi(pm, path, "Retro")


def process_mirror(path, center):
    pm = load_pretty_midi(path)
    pm = apply_mirror(pm, center)
    return _save_modified_midi(pm, path, "Mirror")


def process_drop2(path):
    pm = load_pretty_midi(path)
    pm = apply_drop2(pm)
    return _save_modified_midi(pm, path, "Drop2")


def process_limit_range(path, min_p, max_p):
    pm = load_pretty_midi(path)
    pm = limit_pitch_range(pm, min_p, max_p)
    return _save_modified_midi(pm, path, "Limit")


def process_rand_vel(path, min_v, max_v):
    pm = load_pretty_midi(path)
    pm = randomize_velocity(pm, min_v=min_v, max_v=max_v)
    return _save_modified_midi(pm, path, "RandVel")


def process_rhythm(path):
    pm = load_pretty_midi(path)
    pm = extract_rhythm(pm)
    return _save_modified_midi(pm, path, "Rhythm")


def process_speed(path, ratio):
    pm = load_pretty_midi(path)
    pm = time_stretch(pm, ratio)
    suffix = "Fast" if ratio < 1 else "Slow"
    return _save_modified_midi(pm, path, suffix)


def process_melody_gen(path, root, scale, bpm=120.0):
    if root == "Auto" or "Auto" in scale:
        return None
    pm = load_pretty_midi(path)
    root_idx = NOTE_NAMES.index(root)
    pm = generate_melody_algo(pm, scale, root_idx, bpm=bpm)
    return _save_modified_midi(pm, path, "Melody")


def process_euclidean(path, steps, hits, note, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_euclidean_rhythm(pm, steps, hits, note, bpm=bpm)
    return _save_modified_midi(pm, path, f"Euc_{hits}_{steps}")


def process_midi_delay(path, time, feed):
    pm = load_pretty_midi(path)
    pm = apply_midi_delay(pm, time, feed)
    return _save_modified_midi(pm, path, "Delay")


def process_density(path, amount):
    pm = load_pretty_midi(path)
    pm = apply_density_thin(pm, amount)
    return _save_modified_midi(pm, path, "Thin")


def process_pad(path, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_pad_layer(pm, bpm=bpm)
    return _save_modified_midi(pm, path, "Pad")


def process_drum_pattern(path, style, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_standard_beat(pm, style, bpm=bpm)
    return _save_modified_midi(pm, path, f"Beat_{style}")


def process_remove_silence(path):
    pm = load_pretty_midi(path)
    pm = remove_empty_bars(pm)
    return _save_modified_midi(pm, path, "NoGap")


def process_click(path, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_click_track(pm, bpm=bpm)
    return _save_modified_midi(pm, path, "Click")


def process_json_export(path, bpm=120.0):
    out_path = os.path.splitext(path)[0] + ".json"
    return export_to_json(path, out_path, bpm)


def process_progression(path, root, scale, ptype, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_chord_progression(pm, root, scale, ptype, bpm=bpm)
    return _save_modified_midi(pm, path, "Prog")


def process_groove(path, template, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = apply_groove_template(pm, template, bpm=bpm)
    return _save_modified_midi(pm, path, "Groove")


def process_merge_glitch(path):
    pm = load_pretty_midi(path)
    pm = merge_glitch_notes(pm)
    return _save_modified_midi(pm, path, "DeGlitch")


def process_invert_vel(path):
    pm = load_pretty_midi(path)
    pm = invert_velocity(pm)
    return _save_modified_midi(pm, path, "InvVel")


def process_stats(path):
    dir_name = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(dir_name, f"{base_name}_Stats.txt")
    return export_scale_stats(load_pretty_midi(path), out_path)


def process_chord_detection(midi_path):
    from korshmer.midi_processing import ChordDetector

    pm = load_pretty_midi(midi_path)
    detector = ChordDetector()
    chords = detector.detect_chords(pm)
    dir_name = os.path.dirname(midi_path)
    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    out_path = os.path.join(dir_name, f"{base_name}_ChordDetection.txt")
    with open(out_path, "w") as f:
        f.write(f"Chord Detection Results for: {base_name}\n")
        f.write("=" * 50 + "\n\n")
        for start, chord in chords:
            f.write(f"{start:8.2f}s  {chord.get_name()}\n")
    return out_path


ARPEGGIO_RATES = {"1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125}
CHORD_WINDOW_SECONDS = 0.05


def _file_tempo(pm):
    _times, tempi = pm.get_tempo_changes()
    return float(tempi[0]) if len(tempi) else 120.0


def process_advanced_quantize(midi_path, strength=1.0, grid_div=4):
    from korshmer.midi_processing import AdvancedQuantizer
    from korshmer.midi_processing.quantization import Note

    pm = load_pretty_midi(midi_path)
    quantizer = AdvancedQuantizer(bpm=_file_tempo(pm))
    quantizer.quantize_strength = strength
    division = 1.0 / grid_div
    for inst in pm.instruments:
        notes = [Note(n.start, n.end, n.pitch, n.velocity) for n in inst.notes]
        inst.notes = [
            pretty_midi.Note(
                velocity=q.velocity,
                pitch=q.pitch,
                start=q.start_time,
                end=q.end_time,
            )
            for q in quantizer.quantize_notes(notes, division)
        ]
    return _save_modified_midi(pm, midi_path, "AdvQuant")


def _chords_by_start(notes):
    chords = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if chords and note.start - chords[-1][0].start < CHORD_WINDOW_SECONDS:
            chords[-1].append(note)
        else:
            chords.append([note])
    return chords


def process_arpeggiator_advanced(midi_path, pattern="up", rate="1/16"):
    from korshmer.midi_processing import Arpeggiator, ArpPattern

    if rate not in ARPEGGIO_RATES:
        raise ValueError(
            f"Unknown arpeggio rate {rate!r}; "
            f"use one of {', '.join(ARPEGGIO_RATES)}"
        )
    try:
        arp_pattern = ArpPattern(str(pattern).lower().replace("-", "_"))
    except ValueError:
        raise ValueError(
            f"Unknown arpeggio pattern {pattern!r}; use one of "
            f"{', '.join(p.value for p in ArpPattern)}"
        ) from None

    pm = load_pretty_midi(midi_path)
    arp = Arpeggiator(bpm=_file_tempo(pm))
    arp.pattern = arp_pattern
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        chords = _chords_by_start(inst.notes)
        arped = []
        for index, chord in enumerate(chords):
            start = chord[0].start
            end = max(n.end for n in chord)
            if index + 1 < len(chords):
                end = min(end, chords[index + 1][0].start)
            if len(chord) < 2 or end <= start:
                arped.extend(chord)
                continue
            velocity = max(n.velocity for n in chord)
            for note in arp.generate_arpeggio(
                [n.pitch for n in chord],
                end - start,
                ARPEGGIO_RATES[rate],
                velocity,
            ):
                arped.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=start + note.start_time,
                        end=start + note.start_time + note.duration,
                    )
                )
        inst.notes = sorted(arped, key=lambda n: n.start)
    return _save_modified_midi(pm, midi_path, f"Arp_{arp_pattern.value}")


def process_scale_detection(midi_path):
    from korshmer.midi_processing import ScaleDetector

    pm = load_pretty_midi(midi_path)
    notes = sorted(
        (n for inst in pm.instruments if not inst.is_drum for n in inst.notes),
        key=lambda n: n.start,
    )
    scales = ScaleDetector().detect_scale_from_notes([n.pitch for n in notes])
    dir_name = os.path.dirname(midi_path)
    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    out_path = os.path.join(dir_name, f"{base_name}_ScaleAnalysis.txt")
    with open(out_path, "w") as f:
        f.write(f"Scale Analysis for: {base_name}\n")
        f.write("=" * 50 + "\n\n")
        if not scales:
            f.write("No matching scale found.\n")
        for scale in scales:
            f.write(f"{scale.get_name()}  ({scale.get_formula()})\n")
    return out_path


def process_smart_loop(path, bars, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = apply_smart_loop(pm, bars, bpm)
    return _save_modified_midi(pm, path, f"Loop_{bars}Bar")


def process_ratchet(path, div):
    pm = load_pretty_midi(path)
    pm = apply_ratchet(pm, div)
    return _save_modified_midi(pm, path, f"Ratchet_x{div}")


def process_microtuning(path, amt):
    pm = load_pretty_midi(path)
    pm = apply_microtuning(pm, amt)
    return _save_modified_midi(pm, path, "Detune")


def process_vel_ramp(path, v1, v2):
    pm = load_pretty_midi(path)
    pm = apply_velocity_ramp(pm, v1, v2)
    return _save_modified_midi(pm, path, "Ramp")


def process_melody_continuation(
    path, bars, root, scale, grid, quant, swing, bpm=120.0
):
    pm = load_pretty_midi(path)
    pm = generate_melody_continuation(pm, bars, bpm=bpm)
    if root != "Auto" and "Auto" not in scale:
        try:
            root_idx = NOTE_NAMES.index(root)
            pm = apply_scale_snap(pm, root_idx, scale)
        except ValueError:
            pass
    if quant > 0:
        grid_div = GRID_RESOLUTIONS.get(grid, 2)
        pm = smart_quantize(
            pm,
            detected_tempo=bpm,
            beat_times=None,
            grid_divisions=grid_div,
            strength=quant,
            swing=swing,
        )
    return _save_modified_midi(pm, path, "AI_Cont")


def process_polyrhythm(path, s1, s2, bpm=120.0):
    pm = load_pretty_midi(path)
    pm = generate_polyrhythms(pm, s1, s2, bpm=bpm)
    return _save_modified_midi(pm, path, f"Poly_{s1}_{s2}")


def process_voice_leading(path):
    pm = load_pretty_midi(path)
    pm = apply_voice_leading(pm)
    return _save_modified_midi(pm, path, "VoiceLead")


def process_lfo(path, freq, amt):
    pm = load_pretty_midi(path)
    pm = apply_lfo_modulation(pm, freq, amt, target="Velocity")
    return _save_modified_midi(pm, path, "LFO")


def process_song_builder(
    root,
    scale,
    style,
    bpm,
    structure,
    progression=None,
    seed=None,
    parts=None,
    bars_per_section=8,
):
    from korshmer.midi_generation import build_song

    pm = build_song(
        root,
        scale,
        style=style,
        bpm=bpm,
        structure=structure,
        progression=progression,
        seed=seed,
        parts=parts,
        bars_per_section=bars_per_section,
    )
    pm.instruments = merge_by_instrument(pm.instruments)
    if not pm.instruments:
        return None
    tag = f"{style}_{root}{scale}".replace(" ", "")
    return _save_modified_midi(pm, "Song", f"{tag}_{seed}")
