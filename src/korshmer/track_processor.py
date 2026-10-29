import concurrent.futures
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pretty_midi

from korshmer.analysis import detect_key_and_mode
from korshmer.audio_processing import (
    analyse_tempo,
    run_demucs,
    trim_audio_silence,
)
from korshmer.audio_processing.drum_transcription import (
    MAX_VELOCITY as DRUM_MAX_VELOCITY,
    MIN_VELOCITY as DRUM_MIN_VELOCITY,
    transcribe_drums,
)
from korshmer.audio_processing.music_regions import (
    cut_outside_regions,
    find_music_regions_in_file,
)
from korshmer.audio_processing.separation_engine import separate_with_label
from korshmer.audio_processing.stem_session import (
    build_session_from_processing,
    collect_stems,
    normalize_detected_tempo,
)
from korshmer.audio_processing.transcription_prep import (
    ModelOptions,
    onset_times as audio_onset_times,
    prepare_for_transcription,
    variants_for_stem,
)
from korshmer.constants import (
    DATA_DIR,
    GRID_RESOLUTIONS,
    NOTE_NAMES,
    STEM_FILTER_PROGRESS,
    STEM_TYPE_PATTERNS,
)
from korshmer.lazy_imports import _get_basic_pitch_predict
from korshmer.midi.adaptive_snap import (
    finest_for,
    shared_beat_offset,
    snap_adaptive,
    supports,
)
from korshmer.midi.transcription_cleanup import (
    clean_transcription,
    drop_notes_outside_regions,
    merge_note_variants,
    options_for_cleanup,
    split_notes_at_onsets,
)
from korshmer.midi_algorithms import (
    apply_beat_grid,
    apply_octave_transpose,
    apply_scale_filter,
    fit_beat_grid_or_best,
    fix_legato_and_flutter,
    generate_cc_from_audio,
    normalize_note_velocities,
    process_velocity,
    simplify_chord_density,
    smart_decompose_stem,
    smart_instrument_selector,
)
from korshmer.patches import _apply_demucs_patches, _apply_torchaudio_patches
from korshmer.utils.track_layout import merge_by_instrument
from korshmer.utils.worker_count import compute_worker_count

BASS_SPLIT_MIN_PART = 0.12
BASS_SPLIT_GAP = 0.04
MIDI_STEM_TYPES = ("drums", "bass", "other", "vocals", "guitar", "piano")
VOCAL_PROGRAM = 52
VOCAL_TRACK_NAME = "Vocals"
KEPT_TRACK_NAMES = frozenset({VOCAL_TRACK_NAME})
MUSIC_ONLY_NAME = "music_only.wav"
NO_NOTES_TEXT = "Done! No notes were found to export."


def song_grid_offset(midis, tempo):
    from korshmer.midi_generation.song_match import estimate_offset

    merged = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for midi_obj in midis:
        merged.instruments.extend(midi_obj.instruments)
    try:
        return round(float(estimate_offset(merged, tempo)), 4)
    except ValueError:
        return None


def process_track(
    audio_file,
    target_octave,
    root_note_name,
    scale_type,
    grid_str,
    quantize_strength=1.0,
    swing=0.0,
    velocity_mode="Original",
    velocity_value=0.0,
    selected_stems=None,
    export_format="Merged",
    enable_gm_drums=True,
    enable_cc_gen=False,
    export_wav_32bit=False,
    progress_callback=None,
    check_stop_callback=None,
    prep_options=None,
    cleanup_options=None,
    model_options=None,
    music_options=None,
    separation_options=None,
):
    generated_files = []

    def check_stop():
        if check_stop_callback and check_stop_callback():
            raise InterruptedError("Stopped by user")

    def progress(val, desc):
        if progress_callback and callable(progress_callback):
            progress_callback(val, desc)
        else:
            pass

    file_name = str(Path(audio_file).stem)
    if selected_stems is None:
        selected_stems = ["drums", "bass", "other", "vocals", "guitar", "piano"]
    progress(0.05, desc="Setting up...")
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_work_dir = os.path.join(DATA_DIR, "_temp_work")
    os.makedirs(temp_work_dir, exist_ok=True)
    job_dir = os.path.join(temp_work_dir, file_name)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    os.makedirs(job_dir, exist_ok=True)
    progress(0.15, desc="Preprocessing...")
    check_stop()
    music_file = None
    music_regions = []
    kept_regions = []
    music_cut = None
    try:
        if music_options is not None:
            music = find_music_regions_in_file(
                audio_file,
                music_options,
                progress=lambda fraction, text: progress(
                    0.15 + 0.05 * fraction, desc=text
                ),
            )
            progress(0.2, desc=f"Music detection: {music.reason}")
            if music.applied:
                try:
                    timeline = music.cut_timeline()
                    music_file = cut_outside_regions(
                        audio_file,
                        os.path.join(job_dir, MUSIC_ONLY_NAME),
                        timeline,
                        music_options.fade_s,
                    )
                    music_regions = [
                        [new, new + end - begin] for begin, end, new in timeline
                    ]
                    kept_regions = [list(region) for region in music.regions]
                    music_cut = {
                        "timeline": [list(entry) for entry in timeline],
                        "source": MUSIC_ONLY_NAME,
                    }
                except Exception as exc:
                    print(f"Cutting non-music failed: {exc}", file=sys.stderr)
                    progress(
                        0.2, desc="Music detection skipped: kept the whole file"
                    )
            check_stop()
        trimmed_audio_file = trim_audio_silence(
            music_file or audio_file, job_dir
        )
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    try:
        tempo = float(
            normalize_detected_tempo(analyse_tempo(trimmed_audio_file))
        )
        detected_root_idx, detected_mode = detect_key_and_mode(
            trimmed_audio_file
        )
        final_root_idx = detected_root_idx
        final_scale_mode = detected_mode
        if root_note_name != "Auto":
            try:
                final_root_idx = NOTE_NAMES.index(root_note_name)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Processing failed: {str(e)}") from e
        if scale_type != "Auto (Detect)":
            final_scale_mode = scale_type
        check_stop()
        progress(0.25, desc="Demucs Separation...")
        separation_label = "htdemucs_6s"
        engine_stems_path = None
        separation_source = music_file or (
            audio_file if separation_options is not None else trimmed_audio_file
        )
        try:
            _apply_demucs_patches()
            _apply_torchaudio_patches()
            if separation_options is None:
                run_demucs(separation_source, job_dir)
            else:
                engine_stems_path, separation_label = separate_with_label(
                    separation_source,
                    job_dir,
                    separation_options,
                    progress=lambda fraction, text: progress(
                        0.25 + 0.1 * fraction, desc=text
                    ),
                    should_stop=lambda: bool(
                        check_stop_callback and check_stop_callback()
                    ),
                )
        except InterruptedError:
            raise
        except Exception as e:
            traceback.print_exc()
            message = str(e)
            if message.startswith("Demucs failed"):
                raise RuntimeError(message) from e
            raise RuntimeError(f"Demucs failed: {e}") from e
        check_stop()

        model_root = os.path.join(job_dir, "htdemucs_6s")
        stems_path = engine_stems_path or model_root
        if engine_stems_path is None and os.path.exists(model_root):
            subdirs = [
                d
                for d in os.listdir(model_root)
                if os.path.isdir(os.path.join(model_root, d))
            ]
            if subdirs:
                stems_path = os.path.join(model_root, subdirs[0])

        if not os.path.exists(stems_path):
            for root, _dirs, files in os.walk(job_dir):
                if any(f.endswith(".wav") for f in files):
                    stems_path = root
                    break

        all_stem_files = [
            f
            for f in os.listdir(stems_path)
            if f.endswith((".wav", ".mp3")) and "trimmed" not in f
        ]
        now = datetime.now()
        timestamp_string = (
            now.strftime("%Y_%m_%d__%H_%M_%S")
            + f"__{str(now.timestamp()).replace('.', '')}"
        )
        output_root_folder = os.path.join(DATA_DIR, file_name, timestamp_string)
        progress(STEM_FILTER_PROGRESS, desc="Exporting stems...")
        exported_stems = collect_stems(
            stems_path,
            os.path.join(output_root_folder, "Stems"),
            float32=export_wav_32bit,
        )
        if not exported_stems:
            raise RuntimeError(
                "No stems found after separation. Check Demucs output."
            )
        generated_files.extend(exported_stems.values())
        if music_cut is not None:
            try:
                shutil.copy2(
                    music_file,
                    os.path.join(output_root_folder, MUSIC_ONLY_NAME),
                )
            except OSError as exc:
                print(
                    f"Saving the cut source failed, A/B will play the "
                    f"original: {exc}",
                    file=sys.stderr,
                )
        midi_outputs = []
        session_grid_offset = None

        def write_manifest():
            try:
                build_session_from_processing(
                    audio_file,
                    output_root_folder,
                    tempo,
                    NOTE_NAMES[final_root_idx % len(NOTE_NAMES)],
                    final_scale_mode,
                    exported_stems,
                    midi_outputs,
                    grid_offset=session_grid_offset,
                    music_regions=kept_regions,
                    music_cut=music_cut,
                    separation=separation_label,
                ).save()
            except Exception as manifest_error:
                print(
                    f"Stem manifest write failed: {manifest_error}",
                    file=sys.stderr,
                )

        stem_files = []

        for stem in all_stem_files:
            stem_lower = stem.lower()

            stem_type = None
            for stype, patterns in STEM_TYPE_PATTERNS.items():
                if any(pattern in stem_lower for pattern in patterns):
                    stem_type = stype
                    break

            if stem_type and stem_type in selected_stems:
                if stem_type in ("guitar", "piano"):
                    if (
                        "guitar" not in selected_stems
                        and "piano" not in selected_stems
                    ):
                        continue

                if stem_type in MIDI_STEM_TYPES:
                    stem_files.append(stem)
                    progress(
                        STEM_FILTER_PROGRESS,
                        desc=f"Added MIDI-convertible instrument: {stem_type}",
                    )
                else:
                    progress(
                        STEM_FILTER_PROGRESS,
                        desc=f"Skipping non-MIDI instrument: {stem_type}",
                    )

        tmp_midis = []
        grid_div = GRID_RESOLUTIONS.get(grid_str, 4)
        loop_start_pct, loop_end_pct = 0.40, 0.80
        total_range = loop_end_pct - loop_start_pct
        num_stems = len(stem_files)

        if num_stems == 0:
            progress(0.95, desc="No MIDI-convertible stems selected.")
            write_manifest()
            progress(1.0, desc="Done!")
            return generated_files

        def process_stem_worker(idx, stem_name):
            try:
                check_stop()
                prog = loop_start_pct + (total_range * (idx / num_stems))
                trans_prog = prog + (total_range / num_stems * 0.25)
                progress(trans_prog, desc=f"Transcribing: {stem_name}...")
                s_path = os.path.join(stems_path, stem_name)
                b_name = os.path.splitext(stem_name)[0]

                is_vocal_stem = "vocal" in stem_name.lower()
                is_drum_stem = "drum" in stem_name.lower()
                variant_paths = []
                if not is_drum_stem:
                    variant_paths = [
                        prepare_for_transcription(
                            s_path,
                            os.path.join(job_dir, "_prep"),
                            variant,
                            tag=f"_{i}",
                        )
                        for i, variant in enumerate(
                            variants_for_stem(stem_name, prep_options)[
                                : 1 if is_vocal_stem else None
                            ]
                        )
                    ]
                    model_path = variant_paths[0]

                if is_drum_stem:
                    progress(
                        trans_prog + 0.02,
                        desc=f"Detecting drum hits: {stem_name}...",
                    )
                    m_dat = transcribe_drums(s_path, tempo)
                    progress(
                        trans_prog + 0.15,
                        desc=f"Drum hits detected: {stem_name}",
                    )
                elif is_vocal_stem:
                    try:
                        from korshmer.audio_processing.vocal_melody import (
                            transcribe_vocal_melody,
                        )

                        progress(
                            trans_prog,
                            desc=f"Detecting vocal notes: {stem_name}...",
                        )
                        vocal_notes = transcribe_vocal_melody(
                            s_path,
                            should_stop=lambda: bool(
                                check_stop_callback and check_stop_callback()
                            ),
                        )

                        m_dat = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                        vocal_instrument = pretty_midi.Instrument(
                            program=0, name=VOCAL_TRACK_NAME
                        )

                        for v_note in vocal_notes:
                            midi_note = pretty_midi.Note(
                                velocity=v_note.velocity,
                                pitch=v_note.pitch,
                                start=v_note.start,
                                end=v_note.end,
                            )
                            vocal_instrument.notes.append(midi_note)

                        m_dat.instruments.append(vocal_instrument)
                        progress(
                            trans_prog + 0.1,
                            desc=f"Vocal notes detected: {len(vocal_notes)}",
                        )
                    except InterruptedError:
                        raise
                    except Exception as e:
                        print(
                            f"Vocal note detection failed, using vocal-optimized transcription: {e}",
                            file=sys.stderr,
                        )
                        progress(
                            trans_prog + 0.02,
                            desc=f"Fallback vocal transcription: {stem_name}...",
                        )
                        predict = _get_basic_pitch_predict()
                        _, m_dat, _ = predict(
                            model_path,
                            midi_tempo=tempo,
                            onset_threshold=0.4,
                            frame_threshold=0.2,
                            minimum_note_length=40,
                            minimum_frequency=80,
                            maximum_frequency=1000,
                        )
                        progress(
                            trans_prog + 0.15,
                            desc=f"Vocal transcription complete: {stem_name}",
                        )
                else:
                    progress(
                        trans_prog + 0.02,
                        desc=f"AI Transcription in progress: {stem_name}...",
                    )
                    predict = _get_basic_pitch_predict()
                    model_kwargs = (
                        model_options or ModelOptions.for_stem(stem_name)
                    ).as_kwargs()
                    results = [
                        predict(path, midi_tempo=tempo, **model_kwargs)[1]
                        for path in variant_paths
                    ]
                    m_dat = (
                        results[0]
                        if len(results) == 1
                        else merge_note_variants(results)
                    )
                    progress(
                        trans_prog + 0.15,
                        desc=f"AI Transcription complete: {stem_name}",
                    )

                arr_prog = trans_prog + (total_range / num_stems * 0.25)
                progress(arr_prog, desc=f"Arranging: {stem_name}...")
                drum_check = "drum" in stem_name.lower()
                if music_regions:
                    drop_notes_outside_regions(m_dat, music_regions)
                stem_onsets = (
                    audio_onset_times(variant_paths[0]) if variant_paths else []
                )
                if "bass" in stem_name.lower() and variant_paths:
                    split_notes_at_onsets(
                        m_dat,
                        stem_onsets,
                        min_part=BASS_SPLIT_MIN_PART,
                        gap=BASS_SPLIT_GAP,
                    )
                if not drum_check:
                    m_dat, _ = clean_transcription(
                        m_dat,
                        options=options_for_cleanup(stem_name, cleanup_options),
                        onsets=stem_onsets,
                    )
                if (
                    not drum_check
                    and scale_type != "Auto (Detect)"
                    and final_scale_mode != "Chromatic"
                ):
                    m_dat = apply_scale_filter(
                        m_dat, final_root_idx, final_scale_mode
                    )
                if target_octave != "Original" and not drum_check:
                    m_dat = apply_octave_transpose(m_dat, int(target_octave))
                if velocity_mode == "Original":
                    m_dat = (
                        normalize_note_velocities(
                            m_dat,
                            low=DRUM_MIN_VELOCITY,
                            high=DRUM_MAX_VELOCITY,
                        )
                        if drum_check
                        else normalize_note_velocities(m_dat)
                    )
                m_dat = process_velocity(m_dat, velocity_mode, velocity_value)
                if enable_cc_gen and not drum_check:
                    m_dat = generate_cc_from_audio(m_dat, s_path)
                if is_vocal_stem:
                    stem_splits = [(f"{b_name}.mid", m_dat)]
                else:
                    stem_splits = smart_decompose_stem(
                        m_dat, b_name, tempo, gm_mapping=enable_gm_drums
                    )
                processed = []
                for f_n, m_obj in stem_splits:
                    if (
                        len(m_obj.instruments) == 1
                        and not m_obj.instruments[0].is_drum
                    ):
                        m_obj = smart_instrument_selector(b_name, m_obj)
                    if is_vocal_stem:
                        for inst in m_obj.instruments:
                            inst.name = VOCAL_TRACK_NAME
                            if inst.program == 0 and not inst.is_drum:
                                inst.program = VOCAL_PROGRAM
                    processed.append((f_n, m_obj, drum_check, s_path))
                return processed
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(f"Processing failed: {str(e)}") from e

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(
                1, min(compute_worker_count("compute"), len(stem_files))
            )
        ) as executor:
            futures = [
                executor.submit(process_stem_worker, i, stem)
                for i, stem in enumerate(stem_files)
            ]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    tmp_midis.extend(res)

        progress(0.95, desc="Merging & Exporting...")

        beat_grid = None
        if quantize_strength > 0:
            onset_times = [
                note.start
                for _, m, _, _ in tmp_midis
                for inst in m.instruments
                for note in inst.notes
            ]
            beat_grid = fit_beat_grid_or_best(
                onset_times, tempo, grid_divisions=grid_div
            )
            if beat_grid is not None:
                tempo = round(beat_grid[0], 2)

        shared_offset = None
        if beat_grid is not None and supports(grid_div):
            shared_offset = shared_beat_offset(
                [m for _, m, _, _ in tmp_midis],
                beat_grid[0],
                beat_grid[1],
                grid_div,
            )
        snap_totals = {}

        def report_snapping():
            if snap_totals:
                summary = ", ".join(
                    f"{label}: {count}" for label, count in snap_totals.items()
                )
                progress(0.99, desc=f"Snapped notes to the grid ({summary})")

        def polish_midi(midi_obj):
            midi_obj = fix_legato_and_flutter(midi_obj, min_duration=0.25)
            midi_obj = simplify_chord_density(midi_obj, max_voices=4)
            if beat_grid is not None and supports(grid_div):
                counts = snap_adaptive(
                    midi_obj,
                    beat_grid[0],
                    shared_offset,
                    finest=finest_for(grid_div),
                    strength=quantize_strength,
                    swing=swing,
                    phase_div=1,
                    snap_drums=True,
                )
                for label, count in counts.items():
                    snap_totals[label] = snap_totals.get(label, 0) + count
            elif beat_grid is not None:
                midi_obj = apply_beat_grid(
                    midi_obj,
                    beat_grid[0],
                    beat_grid[1],
                    grid_divisions=grid_div,
                    strength=quantize_strength,
                    swing=swing,
                )
            return midi_obj

        output_folder = os.path.join(output_root_folder, "MIDI")
        os.makedirs(output_folder, exist_ok=True)
        if export_format == "Separate":
            polished = []
            for fname, m, _, _ in tmp_midis:
                if not any(inst.notes for inst in m.instruments):
                    continue
                m.instruments = merge_by_instrument(
                    m.instruments,
                    keep_names=KEPT_TRACK_NAMES
                    | {inst.name for inst in m.instruments if inst.is_drum},
                )
                m = polish_midi(m)
                polished.append(m)
                out_path = os.path.join(output_folder, fname)
                header = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                header.instruments = m.instruments
                header.write(out_path)
                generated_files.append(out_path)
                midi_outputs.append(out_path)
            if not polished:
                progress(0.99, desc=NO_NOTES_TEXT)
            session_grid_offset = song_grid_offset(polished, tempo)
            write_manifest()
            report_snapping()
            progress(1.0, desc="Done!" if polished else NO_NOTES_TEXT)
            return generated_files
        else:
            pretty_midi.PrettyMIDI(initial_tempo=tempo)
            all_instruments = merge_by_instrument(
                [inst for _, m, _, _ in tmp_midis for inst in m.instruments],
                keep_names=KEPT_TRACK_NAMES,
            )
            MAX_CHANNELS = 16

            polished = []

            def save_part(instruments_chunk, suffix=""):
                pm_part = pretty_midi.PrettyMIDI(initial_tempo=tempo)
                pm_part.instruments = instruments_chunk
                pm_part = polish_midi(pm_part)
                polished.append(pm_part)
                out_name = f"{file_name}{suffix}.mid"
                out_path = os.path.join(output_folder, out_name)
                pm_part.write(out_path)
                return out_path

            if not all_instruments:
                progress(0.99, desc=NO_NOTES_TEXT)
            elif len(all_instruments) <= MAX_CHANNELS:
                saved_path = save_part(all_instruments)
                generated_files.append(saved_path)
                midi_outputs.append(saved_path)
            else:
                for i in range(0, len(all_instruments), MAX_CHANNELS):
                    chunk = all_instruments[i : i + MAX_CHANNELS]
                    part_num = (i // MAX_CHANNELS) + 1
                    saved_path = save_part(chunk, suffix=f"_Part{part_num}")
                    generated_files.append(saved_path)
                    midi_outputs.append(saved_path)
            session_grid_offset = song_grid_offset(polished, tempo)
            write_manifest()
            report_snapping()
            progress(1.0, desc="Done!" if polished else NO_NOTES_TEXT)
            return generated_files
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Processing failed: {str(e)}") from e
    finally:
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
            except Exception:
                pass
