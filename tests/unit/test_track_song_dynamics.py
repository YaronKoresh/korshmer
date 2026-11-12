import os

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import track_processor

SR = 22050
SECONDS = 12.0
STEMS = {"bass": (-6.0, 110.0, 45), "other": (-18.0, 220.0, 57)}


def _tone(level_db, frequency):
    t = np.arange(int(SECONDS * SR)) / SR
    wave = np.sin(2 * np.pi * frequency * t) * 10 ** (level_db / 20)
    return wave.astype(np.float32)


def _separate(_source, job_dir):
    folder = os.path.join(job_dir, "htdemucs_6s", "song")
    os.makedirs(folder)
    for name, (level_db, frequency, _) in STEMS.items():
        sf.write(
            os.path.join(folder, f"{name}.wav"), _tone(level_db, frequency), SR
        )


def _transcribe(path, midi_tempo=120, **_options):
    name = next(n for n in STEMS if n in os.path.basename(path))
    midi = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
    inst = pretty_midi.Instrument(0)
    for start in np.arange(0.5, SECONDS - 1.0, 0.5):
        inst.notes.append(
            pretty_midi.Note(90, STEMS[name][2], start, start + 0.4)
        )
    midi.instruments.append(inst)
    return None, midi, None


def _mean_velocities(files):
    means = {}
    for path in files:
        if not path.endswith(".mid"):
            continue
        name = next(n for n in STEMS if n in os.path.basename(path).lower())
        notes = [
            n
            for inst in pretty_midi.PrettyMIDI(path).instruments
            for n in inst.notes
        ]
        means[name] = np.mean([n.velocity for n in notes])
    return means


def _convert(tmp_path, monkeypatch, **options):
    source = tmp_path / "song.wav"
    sf.write(source, sum(_tone(db, hz) for db, hz, _ in STEMS.values()), SR)
    monkeypatch.setattr(track_processor, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(track_processor, "run_demucs", _separate)
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )
    monkeypatch.setattr(
        track_processor, "_get_basic_pitch_predict", lambda: _transcribe
    )
    files = track_processor.process_track(
        str(source),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=0.0,
        selected_stems=list(STEMS),
        export_format="Separate",
        **options,
    )
    return _mean_velocities(files)


def test_backing_stem_is_exported_softer_than_the_dominant_stem(
    tmp_path, monkeypatch
):
    means = _convert(tmp_path, monkeypatch)
    assert means["bass"] > 75
    assert means["other"] == pytest.approx(means["bass"] / 2, rel=0.15)


def test_fixed_velocity_mode_still_exports_one_level(tmp_path, monkeypatch):
    means = _convert(
        tmp_path, monkeypatch, velocity_mode="Fixed", velocity_value=70
    )
    assert means == {"bass": 70, "other": 70}
