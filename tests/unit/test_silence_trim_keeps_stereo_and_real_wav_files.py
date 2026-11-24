import os

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import korshmer.audio
import korshmer.audio_processing
from korshmer import track_processor
from korshmer.audio_processing import basic_effects, core_analysis

SR = 22050


@pytest.fixture(
    params=[
        lambda: korshmer.audio_processing.trim_audio_silence,
        lambda: korshmer.audio.trim_audio_silence,
        lambda: core_analysis.trim_audio_silence,
        lambda: basic_effects.trim_audio_silence,
    ],
    ids=["audio_processing", "audio", "core_analysis", "basic_effects"],
)
def trim(request):
    return request.param()


def stereo_song(path, lead=0.5, tail=0.5):
    t = np.arange(SR) / SR
    left = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.25 * np.sin(2 * np.pi * 330.0 * t)
    body = np.stack([left, right], axis=1)
    quiet_before = np.zeros((int(lead * SR), 2))
    quiet_after = np.zeros((int(tail * SR), 2))
    sf.write(path, np.concatenate([quiet_before, body, quiet_after]), SR)
    return body


def test_a_stereo_song_is_still_stereo_after_the_silence_is_trimmed(
    trim, tmp_path
):
    source = tmp_path / "song.wav"
    body = stereo_song(source)
    trimmed, sr = sf.read(trim(str(source), str(tmp_path)))
    assert sr == SR
    assert trimmed.ndim == 2 and trimmed.shape[1] == 2
    assert abs(len(trimmed) - len(body)) < 0.1 * SR


def test_left_and_right_keep_their_own_sound_after_trimming(trim, tmp_path):
    source = tmp_path / "song.wav"
    stereo_song(source)
    trimmed, _ = sf.read(trim(str(source), str(tmp_path)))
    assert trimmed.ndim == 2
    left_level = np.sqrt(np.mean(trimmed[:, 0] ** 2))
    right_level = np.sqrt(np.mean(trimmed[:, 1] ** 2))
    assert left_level > 1.8 * right_level


def test_a_side_that_starts_later_is_not_cut_out_of_step(trim, tmp_path):
    source = tmp_path / "song.wav"
    t = np.arange(SR) / SR
    left = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    right = np.where(t > 0.5, 0.5 * np.sin(2 * np.pi * 330.0 * t), 0.0)
    silence = np.zeros((SR // 2, 2))
    sf.write(
        source, np.concatenate([silence, np.stack([left, right], axis=1)]), SR
    )
    trimmed, _ = sf.read(trim(str(source), str(tmp_path)))
    assert trimmed.ndim == 2
    assert np.abs(trimmed[: int(0.3 * SR), 1]).max() < 0.01
    assert np.abs(trimmed[: int(0.3 * SR), 0]).max() > 0.3


def test_a_mono_song_stays_mono_as_a_safeguard(trim, tmp_path):
    source = tmp_path / "mono.wav"
    t = np.arange(SR) / SR
    sf.write(source, 0.5 * np.sin(2 * np.pi * 220.0 * t), SR)
    trimmed, _ = sf.read(trim(str(source), str(tmp_path)))
    assert trimmed.ndim == 1


def test_a_file_that_cannot_be_decoded_is_never_given_a_wav_name(
    trim, tmp_path
):
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"ID3" + b"\0" * 200)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = trim(str(source), str(out_dir))
    assert not result.endswith(".wav")
    assert open(result, "rb").read() == source.read_bytes()
    assert not (out_dir / "temp_trimmed.wav").exists()


def test_a_trim_problem_is_logged_and_not_printed(trim, tmp_path, capsys):
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"ID3" + b"\0" * 200)
    trim(str(source), str(tmp_path))
    printed = capsys.readouterr()
    assert "Trim" not in printed.out


def test_trimming_sound_in_memory_still_works_as_a_safeguard(trim):
    t = np.arange(SR) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.concatenate([np.zeros(SR // 2), tone, np.zeros(SR // 2)])
    trimmed = trim(audio, SR)
    assert 0.8 * len(tone) < len(trimmed) < len(audio)


def test_stem_separation_receives_the_song_in_stereo(tmp_path, monkeypatch):
    monkeypatch.setattr(track_processor, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: 120.0)
    monkeypatch.setattr(
        track_processor, "detect_key_and_mode", lambda _p: (9, "Minor")
    )
    monkeypatch.setattr(track_processor, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(
        track_processor, "_apply_torchaudio_patches", lambda: None
    )

    def notes(path, **_kwargs):
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        inst.notes = [pretty_midi.Note(90, 60, 0.2, 0.4)]
        midi.instruments.append(inst)
        return None, midi, []

    monkeypatch.setattr(
        track_processor, "_get_basic_pitch_predict", lambda: notes
    )
    heard = {}

    def separate(audio_path, output_dir):
        audio, _ = sf.read(audio_path)
        heard["channels"] = 1 if audio.ndim == 1 else audio.shape[1]
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        sf.write(os.path.join(target, "other.wav"), audio, SR)

    monkeypatch.setattr(track_processor, "run_demucs", separate)
    song = tmp_path / "Tune.wav"
    stereo_song(song)
    track_processor.process_track(
        str(song),
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=0.0,
        selected_stems=["other"],
        export_format="Separate",
    )
    assert heard["channels"] == 2
