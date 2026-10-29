import os
import shutil
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import track_processor
from korshmer.audio_processing.transcription_prep import ModelOptions
from korshmer.midi.transcription_cleanup import (
    CleanupOptions,
    clean_transcription,
    drop_notes_outside_regions,
    merge_note_variants,
    options_for_cleanup,
)
from tests.fixtures import transcription_eval as ev
from tests.fixtures.transcription_fixtures import to_pretty_midi

SR = 22050
STEM_NAMES = ("other", "vocals", "drums")


def _write_stems(output_dir, names):
    target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
    os.makedirs(target, exist_ok=True)
    t = np.arange(SR * 2) / SR
    tone = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
    for name in names:
        sf.write(os.path.join(target, f"{name}.wav"), tone, SR)


def _fake_predict(calls):
    def predict(path, **kwargs):
        calls.append((os.path.basename(str(path)), kwargs))
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        for index in range(4):
            start = 0.2 + index * 0.5
            inst.notes.append(pretty_midi.Note(90, 57, start, start + 0.3))
        midi.instruments.append(inst)
        return None, midi, []

    return predict


def _run_track(tmp_path, monkeypatch, selected, predict, melody, progress):
    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (9, "Minor")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )

    def fake_trim(path, job_dir):
        target = os.path.join(job_dir, "temp_trimmed.wav")
        shutil.copy2(path, target)
        return target

    monkeypatch.setattr(track_processor, "trim_audio_silence", fake_trim)
    monkeypatch.setattr(
        track_processor,
        "run_demucs",
        lambda _audio, output_dir: _write_stems(output_dir, STEM_NAMES),
    )
    monkeypatch.setattr(
        track_processor, "_get_basic_pitch_predict", lambda: predict
    )
    monkeypatch.setattr(
        "korshmer.audio_processing.vocal_melody.transcribe_vocal_melody",
        melody,
    )
    song = tmp_path / "Tune.wav"
    sf.write(song, np.zeros(SR, dtype="float32"), SR)
    return track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=0.0,
        selected_stems=selected,
        export_format="Separate",
        progress_callback=lambda value, desc="": progress.append(desc),
    )


def _sung_notes(_path, options=None, progress=None, should_stop=None):
    return [
        SimpleNamespace(pitch=62 + index, velocity=90, start=s, end=s + 0.3)
        for index, s in enumerate((0.2, 0.7, 1.2))
    ]


def _midi_notes(generated, stem):
    paths = [p for p in generated if os.path.basename(p) == f"{stem}.mid"]
    assert paths, f"{stem}.mid was not written"
    return [
        n
        for inst in pretty_midi.PrettyMIDI(paths[0]).instruments
        for n in inst.notes
    ]


def test_other_and_vocals_stems_are_transcribed_when_selected(
    tmp_path, monkeypatch
):
    calls, progress = [], []
    generated = _run_track(
        tmp_path,
        monkeypatch,
        ["other", "vocals"],
        _fake_predict(calls),
        _sung_notes,
        progress,
    )

    assert {name.split("_prep")[0] for name, _ in calls} == {"other"}
    assert len(calls) == 2
    assert len(_midi_notes(generated, "other")) >= 4
    assert len(_midi_notes(generated, "vocals")) == 3
    assert "Vocal notes detected: 3" in progress
    assert not any("Skipping" in text for text in progress)


def test_deselected_stems_are_still_skipped(tmp_path, monkeypatch):
    calls, progress, sung = [], [], []

    def melody(path, options=None, progress=None, should_stop=None):
        sung.append(path)
        return _sung_notes(path)

    generated = _run_track(
        tmp_path, monkeypatch, ["other"], _fake_predict(calls), melody, progress
    )

    assert sung == []
    assert {name.split("_prep")[0] for name, _ in calls} == {"other"}
    names = {os.path.basename(p) for p in generated if p.endswith(".mid")}
    assert names == {"other.mid"}


def test_vocal_branch_uses_the_melody_extractor_and_falls_back_to_basic_pitch(
    tmp_path, monkeypatch
):
    calls, progress = [], []
    generated = _run_track(
        tmp_path,
        monkeypatch,
        ["vocals"],
        _fake_predict(calls),
        _sung_notes,
        progress,
    )
    assert calls == []
    assert [n.pitch for n in _midi_notes(generated, "vocals")]

    def broken(_path, options=None, progress=None, should_stop=None):
        raise RuntimeError("no pitch")

    calls, progress = [], []
    (tmp_path / "fallback").mkdir()
    generated = _run_track(
        tmp_path / "fallback",
        monkeypatch,
        ["vocals"],
        _fake_predict(calls),
        broken,
        progress,
    )
    assert len(calls) == 1 and calls[0][0].startswith("vocals")
    assert calls[0][1]["maximum_frequency"] == 1000
    assert len(_midi_notes(generated, "vocals")) >= 3


def test_a_stop_in_the_vocal_extractor_is_not_a_fallback(tmp_path, monkeypatch):
    calls = []

    def stopped(_path, options=None, progress=None, should_stop=None):
        assert should_stop() is False
        raise InterruptedError("Stopped by user")

    with pytest.raises(RuntimeError, match="Stopped by user"):
        _run_track(
            tmp_path,
            monkeypatch,
            ["vocals"],
            _fake_predict(calls),
            stopped,
            [],
        )
    assert calls == []


def _repeated_16ths(count=8, step=0.125, length=0.085):
    return to_pretty_midi(
        [(48, k * step, k * step + length, 90) for k in range(count)]
    )


def test_repeated_notes_with_audio_attacks_stay_separate():
    midi = _repeated_16ths()
    attacks = [k * 0.125 for k in range(8)]

    merged, _ = clean_transcription(midi)
    kept, report = clean_transcription(midi, onsets=attacks)
    held, _ = clean_transcription(midi, onsets=[0.0])

    assert len(merged.instruments[0].notes) < 8
    assert len(kept.instruments[0].notes) == 8
    assert "runs_merged" in report and report["runs_merged"] == 0
    assert len(held.instruments[0].notes) < 8


def _vote_inputs(shift):
    first = to_pretty_midi([(60, 0.0, 0.2, 90), (64, 1.0, 1.2, 90)])
    second = to_pretty_midi([(60, shift, shift + 0.2, 80)])
    return first, second


def test_vote_tolerance_covers_a_note_shifted_by_0_07_s():
    first, second = _vote_inputs(0.07)

    wide = merge_note_variants([first, second])
    narrow = merge_note_variants([first, second], tolerance=0.06)

    assert [n.pitch for n in wide.instruments[0].notes] == [60]
    assert narrow.instruments[0].notes == []


def test_vote_tolerance_stops_short_of_a_0_10_s_shift():
    first, second = _vote_inputs(0.10)

    assert merge_note_variants([first, second]).instruments[0].notes == []


def _first_only_vote(**kwargs):
    first = to_pretty_midi(
        [
            (60, 0.0, 0.8, 100),
            (62, 1.0, 1.8, 70),
            (64, 2.0, 2.2, 100),
            (65, 3.0, 3.2, 70),
        ]
    )
    second = to_pretty_midi([(72, 5.0, 5.2, 80)])
    merged = merge_note_variants([first, second], **kwargs)
    return [n.pitch for n in merged.instruments[0].notes]


def test_first_only_note_survives_only_when_long_and_louder_than_median():
    assert _first_only_vote() == [60]
    assert _first_only_vote(keep_seconds=None) == []
    assert _first_only_vote(keep_seconds=0.1) == [60, 64]


def test_first_only_vote_without_notes_does_not_fail():
    empty = to_pretty_midi([])
    second = to_pretty_midi([(72, 5.0, 5.2, 80)])

    merged = merge_note_variants([empty, second])

    assert merged.instruments[0].notes == []


def test_octave_doubling_survives_unless_clearly_weaker():
    doubled = to_pretty_midi([(48, 0.0, 1.0, 85), (60, 0.0, 1.0, 90)])
    weak = to_pretty_midi([(48, 0.0, 1.0, 40), (60, 0.0, 1.0, 100)])

    kept, _ = clean_transcription(doubled)
    dropped, _ = clean_transcription(weak)

    assert sorted(n.pitch for n in kept.instruments[0].notes) == [48, 60]
    assert [n.pitch for n in dropped.instruments[0].notes] == [60]


def test_sub_octave_default_drops_a_lower_note_under_0_9_of_the_upper():
    midi = to_pretty_midi([(48, 0.0, 1.0, 80), (60, 0.0, 1.0, 90)])

    cleaned, report = clean_transcription(midi)

    assert [n.pitch for n in cleaned.instruments[0].notes] == [60]
    assert report["sub_octaves_removed"] == 1


def test_weak_octave_ghost_is_removed_in_a_bass_stem():
    midi = to_pretty_midi([(48, 0.0, 1.0, 100), (60, 0.0, 1.0, 70)])

    cleaned, report = clean_transcription(
        midi, options=options_for_cleanup("bass.wav")
    )

    assert [n.pitch for n in cleaned.instruments[0].notes] == [48]
    assert report["ghosts_removed"] == 1


def test_octave_doubling_at_0_7_velocity_survives_in_other_stems():
    midi = to_pretty_midi([(48, 0.0, 1.0, 100), (60, 0.0, 1.0, 70)])

    for stem in ("guitar.wav", "piano.wav", "other.wav"):
        cleaned, report = clean_transcription(
            midi, options=options_for_cleanup(stem)
        )
        assert sorted(n.pitch for n in cleaned.instruments[0].notes) == [
            48,
            60,
        ], stem
        assert "ghosts_removed" not in report


def test_ghost_removal_follows_an_explicit_choice_for_every_stem():
    assert CleanupOptions().remove_ghosts is None
    assert options_for_cleanup("bass.wav").remove_ghosts
    assert not options_for_cleanup("vocals.wav").remove_ghosts
    assert not options_for_cleanup("guitar.wav").remove_ghosts

    on = options_for_cleanup("guitar.wav", CleanupOptions(remove_ghosts=True))
    off = options_for_cleanup("bass.wav", CleanupOptions(remove_ghosts=False))
    assert on.remove_ghosts is True
    assert off.remove_ghosts is False


def test_other_stems_get_softer_thresholds_than_the_general_ones():
    other = ModelOptions.for_stem("other.wav")
    general = ModelOptions()

    assert other.onset_threshold < general.onset_threshold
    assert other.frame_threshold < general.frame_threshold
    assert other.maximum_frequency == general.maximum_frequency


def test_vocal_stems_do_not_merge_repeated_notes():
    assert not options_for_cleanup("vocals.wav").merge_runs
    assert options_for_cleanup("guitar.wav").merge_runs


def _note_midi(spec):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(90, p, s, e) for p, s, e in spec]
    midi.instruments.append(inst)
    return midi


def test_notes_in_overlapping_widened_regions_are_not_cut_early():
    midi = _note_midi([(60, 10.1, 15.0)])

    removed = drop_notes_outside_regions(
        midi, [(0.0, 10.0), (10.2, 20.0)], margin=0.25
    )

    note = midi.instruments[0].notes[0]
    assert removed == 0
    assert (note.start, note.end) == (10.1, 15.0)


def test_overlapping_regions_still_trim_at_the_merged_end():
    midi = _note_midi([(60, 10.1, 30.0), (62, 20.2, 21.0), (64, 20.4, 21.0)])

    removed = drop_notes_outside_regions(
        midi, [(10.2, 20.0), (0.0, 10.0)], margin=0.25
    )

    notes = midi.instruments[0].notes
    assert removed == 1
    assert [(n.pitch, n.end) for n in notes] == [(60, 20.25), (62, 20.25)]


@pytest.mark.skipif(
    not ev.model_available(), reason="basic_pitch model is not installed"
)
def test_default_pipeline_keeps_fixture_precision():
    results = ev.evaluate(cleanup=CleanupOptions())

    assert results["mean"]["f1"] >= 0.9
    assert results["mean"]["precision"] >= 0.88
    assert results["bass_a_minor"]["f1"] >= 0.9
    assert results["guitar_g_major"]["f1"] >= 0.99
