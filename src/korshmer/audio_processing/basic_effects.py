import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from korshmer.audio.loader import check_audio_file_channels

from .validation import (
    is_empty_audio,
    require_output_target,
    validate_effect_number,
    validate_sample_rate,
)

_FILTER_ALIASES = {
    "lowpass": "low",
    "highpass": "high",
    "low": "low",
    "high": "high",
}


def _validate_filter_type(filter_type):
    ftype = _FILTER_ALIASES.get(filter_type)
    if ftype is None:
        raise ValueError(
            f"Unknown filter_type: {filter_type!r}; use 'low' or 'high' "
            f"(or 'lowpass' or 'highpass')"
        )
    return ftype


def _is_file_path(obj):
    return isinstance(obj, (str, Path))


def apply_filter(
    audio_or_path,
    sample_rate_or_output=None,
    filter_type="lowpass",
    cutoff=1000,
    output_path=None,
):
    ftype = _validate_filter_type(filter_type)
    validate_effect_number(cutoff, "Cutoff", 0.0, 192000.0)
    if _is_file_path(audio_or_path):
        input_path = audio_or_path
        output_file = require_output_target(
            output_path
            if sample_rate_or_output is None
            else sample_rate_or_output,
            "Output file path",
        )

        check_audio_file_channels(input_path)
        y, sr = librosa.load(input_path, sr=None, mono=False)
        nyquist = 0.5 * sr
        norm_cutoff = cutoff / nyquist

        if norm_cutoff <= 0 or norm_cutoff >= 1:
            sf.write(output_file, y.T, sr)
            return

        b, a = signal.butter(5, norm_cutoff, btype=ftype, analog=False)
        if y.ndim == 1:
            y_filt = signal.filtfilt(b, a, y)
        else:
            y_filt = signal.filtfilt(b, a, y, axis=1)
        sf.write(output_file, y_filt.T, sr)
    else:
        audio = audio_or_path
        sample_rate = sample_rate_or_output

        if is_empty_audio(audio):
            return audio

        validate_sample_rate(sample_rate)
        nyquist = 0.5 * sample_rate
        norm_cutoff = cutoff / nyquist

        if norm_cutoff <= 0 or norm_cutoff >= 1:
            return audio

        b, a = signal.butter(5, norm_cutoff, btype=ftype, analog=False)

        if audio.ndim == 1:
            filtered = signal.filtfilt(b, a, audio)
        else:
            filtered = signal.filtfilt(b, a, audio, axis=-1)

        return filtered


def apply_normalize(audio_or_path, output_path=None):
    if _is_file_path(audio_or_path):
        input_path = audio_or_path
        require_output_target(output_path, "Output file path")
        check_audio_file_channels(input_path)
        y, sr = librosa.load(input_path, sr=None, mono=False)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y_norm = y / max_val
        else:
            y_norm = y
        sf.write(output_path, y_norm.T, sr)
    else:
        audio = audio_or_path
        if is_empty_audio(audio):
            return audio

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio


def apply_reverse_audio(audio_or_path, output_path=None):
    if _is_file_path(audio_or_path):
        input_path = audio_or_path
        require_output_target(output_path, "Output file path")
        check_audio_file_channels(input_path)
        y, sr = librosa.load(input_path, sr=None, mono=False)
        if y.ndim == 1:
            y_rev = np.flip(y)
        else:
            y_rev = np.flip(y, axis=1)
        sf.write(output_path, y_rev.T, sr)
    else:
        audio = audio_or_path
        if is_empty_audio(audio):
            return audio

        if audio.ndim == 1:
            return np.flip(audio)
        else:
            return np.flip(audio, axis=-1)


def apply_mono_mix(stereo_audio_or_path, output_path=None):
    if _is_file_path(stereo_audio_or_path):
        input_path = stereo_audio_or_path
        require_output_target(output_path, "Output file path")
        y, sr = librosa.load(input_path, sr=None, mono=True)
        sf.write(output_path, y, sr)
    else:
        stereo_audio = stereo_audio_or_path
        if stereo_audio is None or len(stereo_audio) == 0:
            return stereo_audio

        if stereo_audio.ndim == 1:
            return stereo_audio

        if stereo_audio.shape[0] == 2:
            return np.mean(stereo_audio, axis=0)
        elif stereo_audio.shape[1] == 2:
            return np.mean(stereo_audio, axis=1)
        else:
            return stereo_audio.flatten()


def apply_transient_shaper(
    audio_or_path,
    sample_rate_or_output=None,
    attack=1.0,
    sustain=1.0,
    amount=None,
):
    validate_effect_number(attack, "Attack", 0.0, 10.0)
    validate_effect_number(sustain, "Sustain", 0.0, 10.0)
    if amount is not None:
        validate_effect_number(amount, "Amount", -10.0, 10.0)
    if _is_file_path(audio_or_path):
        input_path = audio_or_path
        output_path = require_output_target(
            sample_rate_or_output, "Output file path"
        )

        shaping_amount = (
            amount
            if amount is not None
            else (attack - 1.0 if attack != 1.0 else 0.5)
        )

        y, sr = librosa.load(input_path, sr=None)
        hop_length = 512
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length
        )
        onset_env = librosa.util.normalize(onset_env)
        repeats = len(y) // len(onset_env) + 1
        envelope_stretched = np.repeat(onset_env, repeats)[: len(y)]
        y_shaped = (
            y
            * (1 + (envelope_stretched * shaping_amount))
            * (1 + ((1 - envelope_stretched) * (sustain - 1)))
        )
        y_shaped = librosa.util.normalize(y_shaped)
        sf.write(output_path, y_shaped, sr)
    else:
        audio = audio_or_path
        sample_rate = sample_rate_or_output

        if is_empty_audio(audio):
            return audio

        validate_sample_rate(sample_rate)

        hop_length = 512
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=sample_rate, hop_length=hop_length
        )
        onset_env = librosa.util.normalize(onset_env)

        repeats = len(audio) // len(onset_env) + 1
        envelope_stretched = np.repeat(onset_env, repeats)[: len(audio)]

        shaped = (
            audio
            * (1 + (envelope_stretched * (attack - 1)))
            * (1 + ((1 - envelope_stretched) * (sustain - 1)))
        )
        shaped = librosa.util.normalize(shaped)

        return shaped


def apply_noise_gate(
    audio_or_path, sample_rate_or_output=None, threshold_db=-40.0
):
    validate_effect_number(threshold_db, "Threshold", -120.0, 0.0)
    if _is_file_path(audio_or_path):
        input_path = audio_or_path
        output_path = require_output_target(
            sample_rate_or_output, "Output file path"
        )

        check_audio_file_channels(input_path)
        y, sr = librosa.load(input_path, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        threshold_linear = 10 ** (threshold_db / 20)
        mask = (np.abs(y) > threshold_linear).astype(float)
        smooth_len = max(1, int(sr * 0.02))
        kernel = np.hanning(smooth_len)
        kernel /= kernel.sum()
        for i in range(y.shape[0]):
            mask[i] = signal.convolve(mask[i], kernel, mode="same")
        y_gated = y * mask
        sf.write(output_path, y_gated.T, sr)
    else:
        audio = audio_or_path
        sample_rate = sample_rate_or_output

        if is_empty_audio(audio):
            return audio

        validate_sample_rate(sample_rate)

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            was_mono = True
        else:
            was_mono = False

        threshold_linear = 10 ** (threshold_db / 20)
        mask = (np.abs(audio) > threshold_linear).astype(float)

        smooth_len = max(1, int(sample_rate * 0.02))
        kernel = np.hanning(smooth_len)
        kernel /= kernel.sum()

        for i in range(audio.shape[0]):
            mask[i] = signal.convolve(mask[i], kernel, mode="same")

        gated = audio * mask

        if was_mono:
            return gated[0]
        return gated


def trim_audio_silence(audio_or_path, sample_rate_or_dir=None, top_db=24):
    validate_effect_number(top_db, "Silence threshold", 1.0, 120.0)
    if _is_file_path(audio_or_path):
        audio_path = audio_or_path
        dir_name = require_output_target(sample_rate_or_dir, "Output folder")

        try:
            y, sr = librosa.load(audio_path, sr=None)
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
    else:
        audio = audio_or_path

        if is_empty_audio(audio):
            return audio

        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed


def slice_audio_onsets(audio_or_path, sample_rate_or_dir=None, db_threshold=20):
    validate_effect_number(db_threshold, "Slice threshold", 1.0, 120.0)
    if _is_file_path(audio_or_path):
        input_path = audio_or_path
        output_dir = require_output_target(sample_rate_or_dir, "Output folder")

        y, sr = librosa.load(input_path, sr=None)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, backtrack=True, units="samples"
        )
        onset_frames = np.concatenate(([0], onset_frames, [len(y)]))
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        saved_files = []
        for i in range(len(onset_frames) - 1):
            start = onset_frames[i]
            end = onset_frames[i + 1]
            chunk = y[start:end]
            if len(chunk) < 1000:
                continue
            rms = np.sqrt(np.mean(chunk**2))
            db = 20 * np.log10(rms + 1e-6)
            if db > -db_threshold:
                out_name = f"{base_name}_Slice_{i + 1:02d}.wav"
                out_path = os.path.join(output_dir, out_name)
                sf.write(out_path, chunk, sr)
                saved_files.append(out_path)
        return saved_files
    else:
        audio = audio_or_path
        sample_rate = sample_rate_or_dir

        if is_empty_audio(audio):
            return []

        validate_sample_rate(sample_rate)

        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sample_rate, backtrack=True, units="samples"
        )

        if len(onset_frames) == 0:
            return [audio]

        onset_frames = np.concatenate(([0], onset_frames, [len(audio)]))

        slices = []
        for i in range(len(onset_frames) - 1):
            start = onset_frames[i]
            end = onset_frames[i + 1]
            chunk = audio[start:end]

            if len(chunk) >= 1000:
                slices.append(chunk)

        return slices
