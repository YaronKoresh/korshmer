import os
import shutil

import librosa
import numpy as np
import soundfile as sf

from .tempo_estimation import estimate_tempo
from .validation import (
    is_empty_audio,
    require_output_target,
    validate_audio_not_empty,
    validate_effect_integer,
    validate_effect_number,
    validate_pitch_shift_semitones,
)


def trim_audio_silence(audio_or_path, dir_name_or_sr=None, top_db=24):
    validate_effect_number(top_db, "Silence threshold", 1.0, 120.0)
    if isinstance(audio_or_path, np.ndarray):
        y = audio_or_path
        if is_empty_audio(y):
            return y.copy()
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed

    audio_path = audio_or_path
    dir_name = require_output_target(dir_name_or_sr, "Output folder")
    try:
        y, sr = librosa.load(audio_path, sr=None)
        validate_audio_not_empty(y)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        new_path = os.path.join(dir_name, "temp_trimmed.wav")
        with open(new_path, "wb") as f:
            sf.write(f, y_trimmed, sr)
        return new_path
    except Exception as e:
        print(f"Trim failed, using original: {e}")
        fallback_path = os.path.join(dir_name, "temp_trimmed.wav")
        shutil.copy2(audio_path, fallback_path)
        return fallback_path


def analyse_tempo(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        validate_audio_not_empty(y)
        return estimate_tempo(y, sr)
    except Exception as e:
        print(f"Tempo analysis failed: {e}")
        return None


def run_demucs(
    audio_file, output_dir, model_name="htdemucs_6s", *, shifts=1, overlap=0.1
):
    validate_effect_integer(shifts, "Demucs shifts", 0, 16)
    validate_effect_number(overlap, "Demucs overlap", 0.0, 0.99)
    import demucs.separate

    from korshmer.processing.device import get_device_manager

    device = get_device_manager().select_device()
    args = [
        "-n",
        model_name,
        "-d",
        device,
        "-o",
        output_dir,
        "--int24",
        "-j",
        "2",
        "--shifts",
        str(shifts),
        "--overlap",
        str(overlap),
        audio_file,
    ]
    try:
        demucs.separate.main(args)
    except SystemExit as exit_request:
        raise RuntimeError(
            f"Demucs failed: exit code {exit_request.code}"
        ) from exit_request


def pitch_shift_file(input_path, output_path, semitones=0, n_steps=None):

    shift = n_steps if n_steps is not None else semitones
    validate_pitch_shift_semitones(shift)

    y, sr = librosa.load(input_path, sr=None)
    validate_audio_not_empty(y)

    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
    sf.write(output_path, y_shifted, sr)


def estimate_meter(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30, sr=None)
        validate_audio_not_empty(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        pulse = librosa.beat.plp(
            onset_envelope=onset_env,
            sr=sr,
            tempo_min=tempo,
            tempo_max=tempo + 0.001,
        )
        beats_per_bar = 4
        if len(pulse) > 0:
            autocorr = librosa.autocorrelate(onset_env, max_size=10 * sr // 512)
            peaks = librosa.util.peak_pick(
                autocorr,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.5,
                wait=10,
            )
            if len(peaks) > 1:
                period = peaks[1] - peaks[0]
                if 0.5 < (period / (60.0 / tempo * sr / 512)) < 3.5:
                    if period % 3 == 0:
                        beats_per_bar = 3
        return beats_per_bar
    except Exception as e:
        print(f"Meter estimation failed: {e}")
        return None
