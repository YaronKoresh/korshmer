import os
import shutil
import time

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer import track_processor

SR = 22050
DAY = 24 * 60 * 60


def notes(path, **_kwargs):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(90, 60, 0.2, 0.4)]
    midi.instruments.append(inst)
    return None, midi, []


def song_in(folder):
    folder.mkdir()
    t = np.arange(SR) / SR
    path = folder / "Tune.wav"
    sf.write(path, (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype("float32"), SR)
    return str(path)


def convert(song, **options):
    return track_processor.process_track(
        song,
        "Original",
        "Auto",
        "Auto (Detect)",
        "1/16",
        quantize_strength=0.0,
        selected_stems=["other"],
        export_format="Separate",
        **options,
    )


@pytest.fixture
def work(tmp_path, monkeypatch):
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
        track_processor, "_get_basic_pitch_predict", lambda: notes
    )
    monkeypatch.setattr(
        track_processor,
        "trim_audio_silence",
        lambda path, job_dir: shutil.copy2(
            path, os.path.join(job_dir, "temp_trimmed.wav")
        ),
    )
    folders = []

    def separate(audio_path, output_dir):
        folders.append(output_dir)
        target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
        os.makedirs(target, exist_ok=True)
        audio, _ = sf.read(audio_path)
        sf.write(os.path.join(target, "other.wav"), audio, SR)

    monkeypatch.setattr(track_processor, "run_demucs", separate)
    return tmp_path / "data" / "_temp_work", folders, monkeypatch


def test_a_second_song_with_the_same_file_name_leaves_the_first_one_alone(
    tmp_path, work
):
    temp_work, folders, monkeypatch = work
    first = song_in(tmp_path / "album_one")
    second = song_in(tmp_path / "album_two")
    separate = track_processor.run_demucs
    survived = []

    def separate_while_another_song_converts(audio_path, output_dir):
        separate(audio_path, output_dir)
        if len(folders) == 1:
            convert(second)
            stem = os.path.join(
                output_dir, "htdemucs_6s", "temp_trimmed", "other.wav"
            )
            survived.append(os.path.isfile(stem))

    monkeypatch.setattr(
        track_processor, "run_demucs", separate_while_another_song_converts
    )
    assert convert(first)
    assert survived == [True]
    assert folders[0] != folders[1]
    assert os.listdir(temp_work) == []


def test_stopping_right_at_the_start_leaves_no_work_folder(tmp_path, work):
    temp_work, _folders, _monkeypatch = work
    song = song_in(tmp_path / "album")
    with pytest.raises(InterruptedError):
        convert(song, check_stop_callback=lambda: True)
    assert os.listdir(temp_work) == []


def test_a_work_folder_left_by_a_crash_days_ago_is_removed(tmp_path, work):
    temp_work, _folders, _monkeypatch = work
    stale = temp_work / "Old_song_abc"
    stale.mkdir(parents=True)
    (stale / "temp_trimmed.wav").write_bytes(b"x")
    long_ago = time.time() - 2 * DAY
    os.utime(stale, (long_ago, long_ago))
    assert convert(song_in(tmp_path / "album"))
    assert not stale.exists()


def test_a_recent_work_folder_of_another_job_is_kept_as_a_safeguard(
    tmp_path, work
):
    temp_work, _folders, _monkeypatch = work
    running = temp_work / "Other_song_xyz"
    running.mkdir(parents=True)
    assert convert(song_in(tmp_path / "album"))
    assert running.exists()
