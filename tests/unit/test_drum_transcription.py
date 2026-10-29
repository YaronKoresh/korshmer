import os
import shutil

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer.audio_processing import drum_transcription as dt
from korshmer.midi_algorithms import smart_decompose_stem
from tests.fixtures import drum_loops
from tests.fixtures.drum_loops import SAMPLE_RATE, make_drum_loop

LABEL_OF_PITCH = {
    36: "kick",
    38: "snare",
    42: "hat",
    46: "open_hat",
    49: "crash",
    41: "tom",
    45: "tom",
    47: "tom",
    50: "tom",
}
TOLERANCE = 0.04


def _match(hits, truth):
    used = set()
    found = {}
    for label, when in truth:
        ok = False
        for index, (seconds, pitch, _velocity, _kind) in enumerate(hits):
            if index in used:
                continue
            if (
                abs(seconds - when) <= TOLERANCE
                and LABEL_OF_PITCH.get(pitch) == label
            ):
                used.add(index)
                ok = True
                break
        counts = found.setdefault(label, [0, 0])
        counts[1] += 1
        counts[0] += ok
    return found, len(hits) - len(used)


@pytest.mark.parametrize("bpm", (120.0, 100.5, 87.35))
def test_a_plain_beat_is_found_hit_by_hit(bpm):
    audio, truth = make_drum_loop(bpm, bars=8)
    found, extra = _match(dt.find_hits(audio, SAMPLE_RATE), truth)
    for label, (hit, total) in found.items():
        assert hit / total >= 0.95, (label, hit, total)
    assert extra <= len(truth) * 0.05


def test_toms_and_open_hats_get_their_own_names():
    audio, truth = make_drum_loop(120.0, bars=8, fills=True, open_hats=True)
    found, extra = _match(dt.find_hits(audio, SAMPLE_RATE), truth)
    assert {"kick", "snare", "hat", "open_hat", "tom"} <= set(found)
    for label, (hit, total) in found.items():
        assert hit / total >= 0.9, (label, hit, total)
    assert extra <= len(truth) * 0.05


def test_kick_and_snare_played_together_are_both_found():
    sr = SAMPLE_RATE
    audio = np.zeros(sr * 8, dtype=np.float32)
    plan = [(0.5, "k"), (1.5, "ks"), (2.5, "k"), (3.5, "ks"), (4.5, "k")]
    plan += [(5.5, "s"), (6.5, "ks")]
    truth = []
    for when, parts in plan:
        start = int(when * sr)
        if "k" in parts:
            sound = drum_loops._kick(sr)
            audio[start : start + len(sound)] += sound
            truth.append(("kick", when))
        if "s" in parts:
            sound = drum_loops._snare(sr)
            audio[start : start + len(sound)] += sound * 0.8
            truth.append(("snare", when))
    audio /= float(np.max(np.abs(audio))) / 0.9
    found, _ = _match(dt.find_hits(audio, sr), truth)
    assert found["kick"][0] == found["kick"][1]
    assert found["snare"][0] == found["snare"][1]


def test_quiet_background_noise_does_not_hide_the_main_hits():
    audio, truth = make_drum_loop(118.0, bars=8)
    noise = np.random.default_rng(0).standard_normal(audio.size)
    noisy = audio + noise.astype(np.float32) * 10 ** (-50 / 20)
    found, extra = _match(dt.find_hits(noisy, SAMPLE_RATE), truth)
    for label, (hit, total) in found.items():
        assert hit / total >= 0.9, (label, hit, total)
    assert extra <= len(truth) * 0.1


def test_louder_hits_get_higher_velocity():
    audio, truth = make_drum_loop(120.0, bars=8)
    snare_times = [when for label, when in truth if label == "snare"]
    for when in snare_times[::2]:
        start = int((when - 0.02) * SAMPLE_RATE)
        audio[start : start + int(0.3 * SAMPLE_RATE)] *= 0.35
    hits = dt.find_hits(audio, SAMPLE_RATE)
    loud, soft = [], []
    for index, when in enumerate(snare_times):
        near = [h for h in hits if h[1] == 38 and abs(h[0] - when) < 0.04]
        assert near
        (soft if index % 2 == 0 else loud).append(near[0][2])
    assert np.mean(loud) > np.mean(soft) + 5


def test_midi_has_a_drum_track_with_valid_general_midi_notes():
    audio, truth = make_drum_loop(120.0, bars=4)
    midi = dt.drum_midi_from_audio(audio, SAMPLE_RATE, tempo=123.4)
    assert len(midi.instruments) == 1
    kit = midi.instruments[0]
    assert kit.is_drum
    assert midi.get_tempo_changes()[1][0] == pytest.approx(123.4)
    assert len(kit.notes) >= len(truth) * 0.95
    for note in kit.notes:
        assert 35 <= note.pitch <= 81
        assert 1 <= note.velocity <= 127
        assert note.end > note.start >= 0


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
        np.full(SAMPLE_RATE, np.nan, dtype=np.float32),
    ],
)
def test_silence_and_bad_audio_give_an_empty_drum_track(audio):
    midi = dt.drum_midi_from_audio(audio, SAMPLE_RATE)
    assert midi.instruments[0].is_drum
    assert midi.instruments[0].notes == []


def test_transcribe_drums_reads_a_file_at_another_sample_rate(tmp_path):
    audio, truth = make_drum_loop(120.0, bars=4)
    import librosa

    low_rate = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=32000)
    path = tmp_path / "drums.wav"
    sf.write(path, low_rate, 32000)
    midi = dt.transcribe_drums(path, tempo=120.0)
    assert len(midi.instruments[0].notes) >= len(truth) * 0.9


def test_drum_kit_is_split_by_piece_and_keeps_its_pitches():
    audio, _ = make_drum_loop(120.0, bars=4, fills=True, open_hats=True)
    midi = dt.drum_midi_from_audio(audio, SAMPLE_RATE)
    parts = smart_decompose_stem(midi, "drums", 120.0, gm_mapping=True)
    names = {name for name, _ in parts}
    assert {
        "drums_Kick.mid",
        "drums_Snare.mid",
        "drums_HiHat.mid",
    } <= names
    total = 0
    for name, part in parts:
        assert part.instruments[0].is_drum
        total += len(part.instruments[0].notes)
        pitches = {n.pitch for n in part.instruments[0].notes}
        if name == "drums_Kick.mid":
            assert pitches <= {35, 36}
        if name == "drums_Snare.mid":
            assert pitches <= {37, 38, 39, 40}
        if name == "drums_HiHat.mid":
            assert pitches <= {42, 44, 46}
    assert total == len(midi.instruments[0].notes)


def test_legacy_drum_split_still_maps_pitched_notes():
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for pitch in (30, 50, 60, 90):
        inst.notes.append(pretty_midi.Note(90, pitch, 0.0, 0.2))
    midi.instruments.append(inst)
    parts = smart_decompose_stem(midi, "drums", 120.0, gm_mapping=True)
    assert {name for name, _ in parts} == {
        "drums_Kick.mid",
        "drums_Toms.mid",
        "drums_Snare.mid",
        "drums_HiHat.mid",
    }


def test_process_track_writes_real_drum_hits(tmp_path, monkeypatch):
    import korshmer.track_processor as track_processor

    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (0, "Major")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )

    def fake_trim(path, job_dir):
        target = os.path.join(job_dir, "temp_trimmed.wav")
        shutil.copy2(path, target)
        return target

    audio, truth = make_drum_loop(120.0, bars=8)

    def fake_run_demucs(_audio, output_dir):
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        sf.write(os.path.join(target, "drums.wav"), audio, SAMPLE_RATE)

    monkeypatch.setattr(track_processor, "trim_audio_silence", fake_trim)
    monkeypatch.setattr(track_processor, "run_demucs", fake_run_demucs)
    song = tmp_path / "Tune.wav"
    sf.write(song, audio, SAMPLE_RATE)
    generated = track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        selected_stems=["drums"],
        quantize_strength=0.0,
    )
    midis = [p for p in generated if p.endswith(".mid")]
    assert midis
    notes = 0
    for path in midis:
        for inst in pretty_midi.PrettyMIDI(path).instruments:
            assert inst.is_drum
            notes += len(inst.notes)
    assert notes >= len(truth) * 0.9


@pytest.mark.parametrize(
    "rate",
    (0, -1, 8000, 15999, float("nan"), float("inf"), None, True, "44100"),
)
def test_find_hits_rejects_a_sample_rate_the_bands_cannot_fit(rate):
    audio, _ = make_drum_loop(120.0, bars=2)
    with pytest.raises(ValueError, match="Sample rate"):
        dt.find_hits(audio, rate)


@pytest.mark.parametrize("channels_first", (True, False))
def test_stereo_audio_gives_the_hits_of_its_mono_mix(channels_first):
    audio, _ = make_drum_loop(120.0, bars=4)
    stereo = np.stack([audio, audio], axis=0 if channels_first else 1)
    assert dt.find_hits(stereo, SAMPLE_RATE) == dt.find_hits(audio, SAMPLE_RATE)


def test_audio_with_more_than_two_dimensions_is_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        dt.find_hits(np.zeros((2, 2, 4096)), SAMPLE_RATE)


@pytest.mark.parametrize(
    "tempo", (float("nan"), float("inf"), 0, -5, None, "fast")
)
def test_midi_tempo_that_cannot_be_stored_falls_back_to_a_valid_one(
    tempo, tmp_path
):
    audio, _ = make_drum_loop(120.0, bars=2)
    midi = dt.drum_midi_from_audio(audio, SAMPLE_RATE, tempo=tempo)
    path = str(tmp_path / "drum_tempo_test.mid")
    midi.write(path)
    again = pretty_midi.PrettyMIDI(path)
    assert np.isfinite(again.get_tempo_changes()[1][0])
    assert again.get_tempo_changes()[1][0] > 0
