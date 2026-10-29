import os
import shutil
from types import SimpleNamespace

import numpy as np
import pretty_midi
import soundfile as sf

from korshmer import track_processor
from korshmer.audio_processing import vocal_melody
from korshmer.audio_processing.transcription_prep import PrepOptions
from korshmer.midi.transcription_cleanup import CleanupOptions
from korshmer.utils.track_layout import DRUMS_NAME

SR = 22050
STEMS = ("piano", "guitar", "vocals", "drums")


def _stem_midi(pitch):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for k in range(8):
        inst.notes.append(pretty_midi.Note(90, pitch, k * 0.5, k * 0.5 + 0.4))
    midi.instruments.append(inst)
    return midi


def _predict(path, **_kwargs):
    name = os.path.basename(path)
    pitch = 60 if "piano" in name else 64
    return None, _stem_midi(pitch), []


def _transcribe_drums(_path, _tempo):
    midi = pretty_midi.PrettyMIDI()
    kit = pretty_midi.Instrument(0, is_drum=True)
    for k, pitch in enumerate((36, 38, 42, 36, 38, 42)):
        kit.notes.append(pretty_midi.Note(100, pitch, k * 0.5, k * 0.5 + 0.1))
    midi.instruments.append(kit)
    return midi


def _vocal_notes(_path, **_kwargs):
    return [
        SimpleNamespace(velocity=90, pitch=67, start=k * 0.5, end=k * 0.5 + 0.4)
        for k in range(4)
    ]


def _same_instrument(stem_name, midi_data):
    for inst in midi_data.instruments:
        inst.program = 0
        inst.name = f"{stem_name.capitalize()} - Acoustic Grand Piano"
    return midi_data


def _export(
    tmp_path, monkeypatch, export, predict=_predict, stems=STEMS, **kwargs
):
    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (0, "Major")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )
    monkeypatch.setattr(
        track_processor, "_get_basic_pitch_predict", lambda: predict
    )
    monkeypatch.setattr(track_processor, "transcribe_drums", _transcribe_drums)
    monkeypatch.setattr(vocal_melody, "transcribe_vocal_melody", _vocal_notes)
    monkeypatch.setattr(
        track_processor, "smart_instrument_selector", _same_instrument
    )
    monkeypatch.setattr(
        track_processor,
        "trim_audio_silence",
        lambda path, job_dir: shutil.copy2(
            path, os.path.join(job_dir, "temp_trimmed.wav")
        ),
    )
    audio = (0.4 * np.sin(2 * np.pi * 220.0 * np.arange(3 * SR) / SR)).astype(
        "float32"
    )

    def fake_demucs(_audio, output_dir):
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        for name in stems:
            sf.write(os.path.join(target, f"{name}.wav"), audio, SR)

    monkeypatch.setattr(track_processor, "run_demucs", fake_demucs)
    song = tmp_path / "Tune.wav"
    sf.write(song, audio, SR)
    generated = track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=0.0,
        selected_stems=list(stems),
        export_format=export,
        prep_options=PrepOptions(enabled=False),
        cleanup_options=CleanupOptions(enabled=False),
        **kwargs,
    )
    return [path for path in generated if path.endswith(".mid")]


def test_merged_export_has_one_track_per_instrument(tmp_path, monkeypatch):
    midis = _export(tmp_path, monkeypatch, "Merged")

    assert len(midis) == 1
    instruments = pretty_midi.PrettyMIDI(midis[0]).instruments
    assert sorted(i.name for i in instruments) == [
        DRUMS_NAME,
        "Grand Piano",
        "Vocals",
    ]
    by_name = {i.name: i for i in instruments}
    assert by_name[DRUMS_NAME].is_drum
    assert len(by_name[DRUMS_NAME].notes) == 6
    assert len(by_name["Grand Piano"].notes) == 16


def test_separate_export_still_names_each_file_by_its_instrument(
    tmp_path, monkeypatch
):
    midis = _export(tmp_path, monkeypatch, "Separate")

    names = {
        os.path.basename(path): [
            i.name for i in pretty_midi.PrettyMIDI(path).instruments
        ]
        for path in midis
    }
    assert names["piano.mid"] == ["Grand Piano"]
    assert names["guitar.mid"] == ["Grand Piano"]
    assert names["vocals.mid"] == ["Vocals"]
    assert names["drums_Kick.mid"] == [f"{DRUMS_NAME} - Kick"]
    assert names["drums_Snare.mid"] == [f"{DRUMS_NAME} - Snare"]
    assert names["drums_HiHat.mid"] == [f"{DRUMS_NAME} - HiHat"]


def _chord_predict(_path, **_kwargs):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for k in range(4):
        for pitch in range(48, 80, 4):
            inst.notes.append(
                pretty_midi.Note(90, pitch, k * 1.0, k * 1.0 + 0.9)
            )
    midi.instruments.append(inst)
    return None, midi, []


def test_dense_chords_end_up_on_one_capped_track(tmp_path, monkeypatch):
    midis = _export(
        tmp_path,
        monkeypatch,
        "Merged",
        predict=_chord_predict,
        stems=("piano",),
    )

    instruments = pretty_midi.PrettyMIDI(midis[0]).instruments
    assert [i.name for i in instruments] == ["Grand Piano"]
    notes = instruments[0].notes
    assert notes
    for note in notes:
        sounding = [n for n in notes if n.start <= note.start < n.end]
        assert len(sounding) <= 6
