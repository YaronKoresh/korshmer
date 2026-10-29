import numpy as np
import pretty_midi
import soundfile as sf

from korshmer.audio_processing.transcription_prep import onset_times
from korshmer.midi.transcription_cleanup import split_notes_at_onsets


def _midi(notes, drum=False):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=drum)
    inst.notes = [pretty_midi.Note(90, p, s, e) for p, s, e in notes]
    pm.instruments.append(inst)
    return pm


def test_a_note_with_an_attack_inside_is_cut_there():
    pm = _midi([(40, 0.0, 1.0)])

    added = split_notes_at_onsets(pm, [0.5])

    notes = pm.instruments[0].notes
    assert added == 1
    assert [(n.pitch, n.start, n.end) for n in notes] == [
        (40, 0.0, 0.5),
        (40, 0.5, 1.0),
    ]
    assert all(n.velocity == 90 for n in notes)


def test_attacks_at_the_edges_or_too_close_make_no_cut():
    pm = _midi([(40, 0.0, 1.0)])

    added = split_notes_at_onsets(pm, [0.0, 0.03, 0.97, 1.0, 2.0])

    assert added == 0
    assert len(pm.instruments[0].notes) == 1


def test_several_attacks_make_several_parts():
    pm = _midi([(40, 0.0, 1.0)])

    split_notes_at_onsets(pm, [0.25, 0.5, 0.75])

    assert [round(n.start, 2) for n in pm.instruments[0].notes] == [
        0.0,
        0.25,
        0.5,
        0.75,
    ]


def test_drums_and_empty_onsets_are_left_alone():
    drums = _midi([(36, 0.0, 1.0)], drum=True)
    plain = _midi([(40, 0.0, 1.0)])

    assert split_notes_at_onsets(drums, [0.5]) == 0
    assert split_notes_at_onsets(plain, []) == 0
    assert len(drums.instruments[0].notes) == 1
    assert len(plain.instruments[0].notes) == 1


def test_onset_times_finds_clicks_and_survives_a_bad_file(tmp_path):
    sr = 22050
    y = np.zeros(sr * 2, dtype="float32")
    t = np.arange(int(0.2 * sr)) / sr
    for start in (0.2, 0.7, 1.2):
        i = int(start * sr)
        y[i : i + len(t)] += 0.8 * np.sin(2 * np.pi * 110 * t) * np.exp(-t * 15)
    path = tmp_path / "bass.wav"
    sf.write(str(path), y, sr)

    found = onset_times(str(path))

    assert len(found) >= 3
    for expected in (0.2, 0.7, 1.2):
        assert min(abs(found - expected)) < 0.06
    assert len(onset_times(str(tmp_path / "missing.wav"))) == 0


def test_bass_stem_notes_are_cut_at_the_audio_attacks(tmp_path, monkeypatch):
    import os
    import shutil

    import korshmer.track_processor as track_processor

    sr = 22050
    t = np.arange(int(0.3 * sr)) / sr
    audio = np.zeros(sr * 4, dtype="float32")
    for start in (0.3, 1.0, 1.7, 2.4):
        i = int(start * sr)
        audio[i : i + len(t)] += (
            0.8 * np.sin(2 * np.pi * 82.4 * t) * np.exp(-t * 12)
        ).astype("float32")

    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (4, "Minor")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )

    def fake_trim(path, job_dir):
        target = os.path.join(job_dir, "temp_trimmed.wav")
        shutil.copy2(path, target)
        return target

    def fake_run_demucs(_audio, output_dir):
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        sf.write(os.path.join(target, "bass.wav"), audio, sr)

    def fake_predict(_path, **_kwargs):
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        inst.notes.append(pretty_midi.Note(90, 40, 0.3, 3.0))
        pm.instruments.append(inst)
        return None, pm, []

    monkeypatch.setattr(track_processor, "trim_audio_silence", fake_trim)
    monkeypatch.setattr(track_processor, "run_demucs", fake_run_demucs)
    monkeypatch.setattr(
        track_processor, "_get_basic_pitch_predict", lambda: fake_predict
    )
    song = tmp_path / "Tune.wav"
    sf.write(song, audio, sr)

    generated = track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        selected_stems=["bass"],
    )

    midis = [p for p in generated if p.endswith(".mid")]
    assert midis
    notes = [
        n
        for path in midis
        for i in pretty_midi.PrettyMIDI(path).instruments
        for n in i.notes
    ]
    assert len(notes) >= 4


def test_a_gap_keeps_the_parts_apart_for_later_flutter_fixes():
    from korshmer.midi_algorithms import fix_legato_and_flutter

    pm = _midi([(40, 0.0, 0.6)])

    split_notes_at_onsets(pm, [0.15, 0.3, 0.45], min_part=0.12, gap=0.04)
    notes = pm.instruments[0].notes

    assert [round(n.start, 2) for n in notes] == [0.0, 0.15, 0.3, 0.45]
    assert round(notes[0].end, 2) == 0.11
    assert notes[-1].end == 0.6
    fixed = fix_legato_and_flutter(pm, min_duration=0.06)
    assert len(fixed.instruments[0].notes) == 4
