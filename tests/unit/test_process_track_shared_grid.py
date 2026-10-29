import json
import os
import shutil

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import track_processor
from korshmer.audio_processing.stem_session import MANIFEST_NAME
from korshmer.audio_processing.transcription_prep import PrepOptions
from korshmer.midi.adaptive_snap import SWING_SHARE, find_beat_offset
from korshmer.midi.transcription_cleanup import CleanupOptions
from tests.fixtures.drum_loops import SAMPLE_RATE, make_drum_loop

TEMPO = 120.0
BEAT = 60.0 / TEMPO
FIRST_BEAT = 0.37
BEATS = 64
TOLERANCE = 0.03
SWING = 1.0


def _setup(tmp_path, monkeypatch, audio, stems, predict=None):
    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: TEMPO)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (0, "Major")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )
    if predict is not None:
        monkeypatch.setattr(
            track_processor, "_get_basic_pitch_predict", lambda: predict
        )
    monkeypatch.setattr(
        track_processor,
        "trim_audio_silence",
        lambda path, job_dir: shutil.copy2(
            path, os.path.join(job_dir, "temp_trimmed.wav")
        ),
    )

    def fake_demucs(_audio, output_dir):
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        for name in stems:
            sf.write(os.path.join(target, f"{name}.wav"), audio, SAMPLE_RATE)

    monkeypatch.setattr(track_processor, "run_demucs", fake_demucs)
    song = tmp_path / "Tune.wav"
    sf.write(song, audio, SAMPLE_RATE)
    return str(song)


def _export(song, stems, swing=0.0, **kwargs):
    generated = track_processor.process_track(
        song,
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=1.0,
        swing=swing,
        selected_stems=list(stems),
        export_format="Separate",
        **kwargs,
    )
    return [path for path in generated if path.endswith(".mid")]


def _manifest(midi_path):
    path = os.path.join(
        os.path.dirname(os.path.dirname(midi_path)), MANIFEST_NAME
    )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _starts(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    return [n.start for inst in midi.instruments for n in inst.notes]


def _circular(value, period):
    return abs((value + period / 2) % period - period / 2)


def test_session_grid_offset_is_the_bar_position_of_the_first_onset(
    tmp_path, monkeypatch
):
    shift = 0.137
    loop, _ = make_drum_loop(TEMPO, bars=8)
    audio = np.concatenate(
        [np.zeros(int(shift * SAMPLE_RATE), dtype=np.float32), loop]
    )
    song = _setup(tmp_path, monkeypatch, audio, ("drums",))

    midis = _export(song, ("drums",))

    assert midis
    tempo = pretty_midi.PrettyMIDI(midis[0]).get_tempo_changes()[1][0]
    bar = 240.0 / tempo
    first = min(start for path in midis for start in _starts(path))
    offset = _manifest(midis[0])["grid_offset"]
    assert offset is not None
    assert first > 0.1
    assert _circular(offset - first, bar) < TOLERANCE


def _line(starts, length, pitch):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for start in starts:
        inst.notes.append(
            pretty_midi.Note(90, pitch, start, start + length * BEAT)
        )
    midi.instruments.append(inst)
    return midi


def _on_beats():
    return _line([FIRST_BEAT + k * BEAT for k in range(BEATS)], 0.9, 48)


def _off_beats():
    starts = [FIRST_BEAT + (k + 0.5) * BEAT for k in range(1, BEATS, 2)]
    return _line(starts, 0.4, 64)


def _predict_by_stem(path, **_kwargs):
    stem = _on_beats() if "piano" in os.path.basename(path) else _off_beats()
    return None, stem, []


def test_stems_with_different_beats_are_snapped_to_one_shared_grid(
    tmp_path, monkeypatch
):
    assert find_beat_offset(_on_beats(), TEMPO, 0.0, 4) != pytest.approx(
        find_beat_offset(_off_beats(), TEMPO, 0.0, 4), abs=0.01
    )
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * np.arange(3 * SAMPLE_RATE))
    stems = ("piano", "guitar")
    song = _setup(
        tmp_path,
        monkeypatch,
        tone.astype("float32"),
        stems,
        predict=_predict_by_stem,
    )

    midis = _export(
        song,
        stems,
        swing=SWING,
        prep_options=PrepOptions(enabled=False),
        cleanup_options=CleanupOptions(enabled=False),
    )

    by_name = {os.path.basename(path): _starts(path) for path in midis}
    assert sorted(by_name) == ["guitar.mid", "piano.mid"]
    written = pretty_midi.PrettyMIDI(midis[0]).get_tempo_changes()[1][0]
    beat = 60.0 / written
    step = beat / 4
    origin = min(by_name["piano.mid"])
    late = SWING_SHARE * SWING * beat / 2
    for name, expected in (("piano.mid", 0.0), ("guitar.mid", late)):
        assert by_name[name]
        for start in by_name[name]:
            assert _circular((start - origin) % step - expected, step) < 0.015
