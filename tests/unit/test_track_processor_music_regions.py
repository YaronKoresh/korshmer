import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import track_processor
from korshmer.audio_processing.music_regions import (
    MusicRegionOptions,
    MusicRegions,
)
from korshmer.audio_processing.separation_engine import SeparationOptions
from korshmer.audio_processing.stem_session import MANIFEST_NAME
from korshmer.midi.transcription_cleanup import drop_notes_outside_regions

SR = 22050
SECONDS = 3
REGION = (1.0, 1.8)


def _midi(starts, drum=False):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=drum)
    for start in starts:
        inst.notes.append(pretty_midi.Note(90, 57, start, start + 0.3))
    midi.instruments.append(inst)
    return midi


def test_drop_notes_outside_regions_removes_residue_and_trims_ends():
    midi = _midi([0.2, 0.7, 1.2, 1.7, 2.5])
    midi.instruments[0].notes[3].end = 4.0

    removed = drop_notes_outside_regions(midi, [REGION], margin=0.25)

    notes = midi.instruments[0].notes
    assert removed == 3
    assert [round(n.start, 2) for n in notes] == [1.2, 1.7]
    assert notes[1].end == REGION[1] + 0.25


def test_drop_notes_outside_regions_keeps_the_margin_and_filters_drums():
    midi = _midi([0.8, 0.7], drum=True)

    removed = drop_notes_outside_regions(midi, [REGION], margin=0.25)

    assert removed == 1
    assert [n.start for n in midi.instruments[0].notes] == [0.8]


def test_drop_notes_outside_regions_handles_several_regions():
    midi = _midi([0.5, 5.2, 9.0])

    removed = drop_notes_outside_regions(midi, [(0.0, 1.0), (5.0, 6.0)])

    assert removed == 1
    assert [n.start for n in midi.instruments[0].notes] == [0.5, 5.2]


def _fake_predict(path, **_kwargs):
    midi = _midi([0.2, 0.7, 1.2, 1.7, 2.4])
    return None, midi, []


def _setup(tmp_path, monkeypatch, stems=("other",), predict=_fake_predict):
    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (9, "Minor")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )
    monkeypatch.setattr(
        track_processor, "_get_basic_pitch_predict", lambda: predict
    )
    monkeypatch.setattr(
        track_processor,
        "find_music_regions_in_file",
        lambda _path, _options, progress=None: MusicRegions(
            (REGION,), float(SECONDS), True, "test"
        ),
    )
    monkeypatch.setattr(
        track_processor,
        "trim_audio_silence",
        lambda path, job_dir: shutil.copy2(
            path, os.path.join(job_dir, "temp_trimmed.wav")
        ),
    )
    heard = {}

    def fake_demucs(audio_path, output_dir):
        heard["source"] = audio_path
        heard["audio"], _ = sf.read(audio_path)
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        for name in stems:
            sf.write(os.path.join(target, f"{name}.wav"), heard["audio"], SR)

    monkeypatch.setattr(track_processor, "run_demucs", fake_demucs)
    t = np.arange(SR * SECONDS) / SR
    song = tmp_path / "Tune.wav"
    sf.write(song, (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype("float32"), SR)
    return heard, str(song)


def _process(song, stems=("other",), export="Separate", **kwargs):
    progress = []
    generated = track_processor.process_track(
        song,
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=0.0,
        selected_stems=list(stems),
        export_format=export,
        progress_callback=lambda _value, desc="": progress.append(desc),
        **kwargs,
    )
    return generated, progress


def _manifest(midi_path):
    path = os.path.join(
        os.path.dirname(os.path.dirname(midi_path)), MANIFEST_NAME
    )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_process_track_cuts_non_music_and_gates_stem_notes(
    tmp_path, monkeypatch
):
    heard, song = _setup(tmp_path, monkeypatch)

    generated, _ = _process(song, music_options=MusicRegionOptions())

    assert os.path.basename(heard["source"]) == "music_only.wav"
    audio = np.abs(heard["audio"])
    assert len(audio) == round((REGION[1] - REGION[0]) * SR)
    assert audio[int(0.2 * SR) : int(0.6 * SR)].max() > 0.1
    midi_path = next(p for p in generated if p.endswith("other.mid"))
    starts = [
        n.start
        for inst in pretty_midi.PrettyMIDI(midi_path).instruments
        for n in inst.notes
    ]
    assert starts
    assert all(0.0 <= start <= 1.05 for start in starts)
    manifest = _manifest(midi_path)
    assert manifest["music_regions"] == [list(REGION)]
    assert manifest["music_cut"] == {
        "timeline": [[REGION[0], REGION[1], 0.0]],
        "source": "music_only.wav",
    }
    cut_copy = os.path.join(
        os.path.dirname(os.path.dirname(midi_path)), "music_only.wav"
    )
    assert os.path.isfile(cut_copy)


def test_a_cut_in_the_middle_closes_the_gap_in_audio_and_notes(
    tmp_path, monkeypatch
):
    heard, song = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        track_processor,
        "find_music_regions_in_file",
        lambda _path, _options, progress=None: MusicRegions(
            ((0.0, 1.0), (2.0, 3.0)), float(SECONDS), True, "test"
        ),
    )

    generated, _ = _process(song, music_options=MusicRegionOptions())

    assert len(heard["audio"]) == 2 * SR
    midi_path = next(p for p in generated if p.endswith("other.mid"))
    starts = sorted(
        round(n.start, 2)
        for inst in pretty_midi.PrettyMIDI(midi_path).instruments
        for n in inst.notes
    )
    assert starts == [0.2, 0.7, 1.2, 1.7]
    assert _manifest(midi_path)["music_cut"]["timeline"] == [
        [0.0, 1.0, 0.0],
        [2.0, 3.0, 1.0],
    ]


def test_a_failed_cut_keeps_the_whole_file_instead_of_failing(
    tmp_path, monkeypatch
):
    heard, song = _setup(tmp_path, monkeypatch)

    def unwritable(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(track_processor, "cut_outside_regions", unwritable)

    generated, progress = _process(song, music_options=MusicRegionOptions())

    assert os.path.basename(heard["source"]) == "temp_trimmed.wav"
    midi_path = next(p for p in generated if p.endswith("other.mid"))
    manifest = _manifest(midi_path)
    assert manifest["music_regions"] == []
    assert manifest["music_cut"] is None
    assert any("Music detection skipped" in text for text in progress)


def test_a_failed_copy_of_the_cut_source_keeps_the_timeline_in_the_manifest(
    tmp_path, monkeypatch
):
    _heard, song = _setup(tmp_path, monkeypatch)
    real_copy = shutil.copy2

    def copy(src, dst, **kwargs):
        if os.path.basename(dst) == "music_only.wav":
            raise OSError("disk full")
        return real_copy(src, dst, **kwargs)

    monkeypatch.setattr(track_processor.shutil, "copy2", copy)

    generated, _ = _process(song, music_options=MusicRegionOptions())

    midi_path = next(p for p in generated if p.endswith("other.mid"))
    manifest = _manifest(midi_path)
    assert manifest["music_cut"]["timeline"] == [[REGION[0], REGION[1], 0.0]]
    assert not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(midi_path)), "music_only.wav"
        )
    )


def test_a_stop_during_separation_is_not_reported_as_a_demucs_failure(
    tmp_path, monkeypatch
):
    _heard, song = _setup(tmp_path, monkeypatch)

    def stopped(*_args, **_kwargs):
        raise InterruptedError("Stopped by user")

    monkeypatch.setattr(track_processor, "separate_with_label", stopped)

    with pytest.raises(RuntimeError) as caught:
        _process(song, separation_options=SeparationOptions(quality="fast"))

    assert str(caught.value) == "Processing failed: Stopped by user"


def test_separate_export_says_so_when_no_notes_were_found(
    tmp_path, monkeypatch
):
    def silent(_path, **_kwargs):
        return None, pretty_midi.PrettyMIDI(), []

    _heard, song = _setup(tmp_path, monkeypatch, predict=silent)

    generated, progress = _process(song)

    assert not [p for p in generated if p.endswith(".mid")]
    assert "No notes were found" in progress[-1]


def test_merged_export_keeps_the_vocal_track_apart_from_the_piano(
    tmp_path, monkeypatch
):
    _heard, song = _setup(tmp_path, monkeypatch, stems=("other", "vocals"))
    monkeypatch.setattr(
        "korshmer.audio_processing.vocal_melody.transcribe_vocal_melody",
        lambda _path, **_kwargs: [
            SimpleNamespace(pitch=62, velocity=90, start=1.2, end=1.5)
        ],
    )

    def default_piano(stem_name, midi):
        inst = pretty_midi.Instrument(
            0, name=f"{stem_name.capitalize()} - Default Piano"
        )
        inst.notes = [n for i in midi.instruments for n in i.notes]
        midi.instruments = [inst]
        return midi

    monkeypatch.setattr(
        track_processor, "smart_instrument_selector", default_piano
    )

    generated, _ = _process(song, stems=("other", "vocals"), export="Merged")

    midi_path = next(p for p in generated if p.endswith("Tune.mid"))
    instruments = pretty_midi.PrettyMIDI(midi_path).instruments
    assert sorted(inst.program for inst in instruments) == [
        0,
        track_processor.VOCAL_PROGRAM,
    ]


def test_a_stop_after_the_cut_leaves_no_temporary_audio(tmp_path, monkeypatch):
    _heard, song = _setup(tmp_path, monkeypatch)
    checks = []

    def stop_second_time():
        checks.append(1)
        return len(checks) >= 2

    with pytest.raises(InterruptedError):
        _process(
            song,
            music_options=MusicRegionOptions(),
            check_stop_callback=stop_second_time,
        )

    assert not (tmp_path / "data" / "_temp_work" / "Tune").exists()
