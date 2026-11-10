import os
import shutil

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import track_processor
from korshmer.audio_processing.transcription_prep import PrepOptions
from korshmer.midi.adaptive_snap import snap_adaptive
from korshmer.midi.transcription_cleanup import CleanupOptions

TEMPO = 120.0
BEAT = 60.0 / TEMPO
RATE = 22050


def test_a_run_of_sixteenth_notes_keeps_every_note_on_its_own_step():
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=0)
    for beat_start in (0.01, 0.26, 0.51, 0.74):
        inst.notes.append(
            pretty_midi.Note(
                90, 64, beat_start * BEAT, (beat_start + 0.2) * BEAT
            )
        )
    pm.instruments.append(inst)

    counts = snap_adaptive(pm, TEMPO, 0.0, finest=4)

    starts = sorted(n.start / BEAT for n in inst.notes)
    assert starts == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert counts["1/16"] == 4


def test_conversion_keeps_eighth_notes_of_a_melodic_stem(tmp_path, monkeypatch):
    eighths = [0.5 + k * BEAT / 2 for k in range(16)]
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * np.arange(5 * RATE) / RATE)

    def predict(_path, **_kwargs):
        midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
        piano = pretty_midi.Instrument(program=0)
        for k, start in enumerate(eighths):
            piano.notes.append(
                pretty_midi.Note(90, 60 + k % 5, start, start + 0.2)
            )
        midi.instruments.append(piano)
        return None, midi, []

    def fake_demucs(_audio, output_dir):
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        sf.write(os.path.join(target, "piano.wav"), tone, RATE)

    def fake_trim(path, job_dir):
        return shutil.copy2(path, os.path.join(job_dir, "temp_trimmed.wav"))

    for name, value in (
        ("DATA_DIR", tmp_path / "data"),
        ("analyse_tempo", lambda _p: TEMPO),
        ("detect_key_and_mode", lambda _p: (0, "Major")),
        ("_apply_demucs_patches", lambda: None),
        ("_apply_torchaudio_patches", lambda: None),
        ("_get_basic_pitch_predict", lambda: predict),
        ("run_demucs", fake_demucs),
        ("trim_audio_silence", fake_trim),
    ):
        monkeypatch.setattr(track_processor, name, value)
    song = tmp_path / "Tune.wav"
    sf.write(song, tone, RATE)

    generated = track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        selected_stems=["piano"],
        prep_options=PrepOptions(enabled=False),
        cleanup_options=CleanupOptions(enabled=False),
    )

    midis = [p for p in generated if p.endswith(".mid")]
    assert midis
    starts = sorted(
        n.start
        for path in midis
        for inst in pretty_midi.PrettyMIDI(path).instruments
        for n in inst.notes
    )
    assert starts == pytest.approx(eighths, abs=0.01)
