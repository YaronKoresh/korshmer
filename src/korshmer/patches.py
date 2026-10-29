import logging
import os

import numpy as np
import soundfile as sf

from korshmer.lazy_imports import _get_demucs_modules, _get_torch_modules

ORT_LARGE_INPUT_DIM = 175376
ORT_CHUNK_SIZE = 43844
ORT_NUM_CHUNKS = 4

try:
    import onnxruntime as ort

    _original_run = ort.InferenceSession.run

    def patched_ort_run(self, output_names, input_feed, run_options=None):
        culprit_key = None
        input_data = None
        for k, v in input_feed.items():
            if (
                hasattr(v, "shape")
                and len(v.shape) >= 2
                and v.shape[1] == ORT_LARGE_INPUT_DIM
            ):
                culprit_key = k
                input_data = v
                break
        if culprit_key is not None:
            results_list = []
            for i in range(ORT_NUM_CHUNKS):
                start = i * ORT_CHUNK_SIZE
                end = start + ORT_CHUNK_SIZE
                segment = input_data[:, start:end, :]
                current_feed = input_feed.copy()
                current_feed[culprit_key] = segment
                res = _original_run(
                    self, output_names, current_feed, run_options
                )
                results_list.append(res)
            final_outputs = []
            num_output_tensors = len(results_list[0])
            for out_idx in range(num_output_tensors):
                arrays_to_cat = [r[out_idx] for r in results_list]
                concatenated = np.concatenate(arrays_to_cat, axis=1)
                final_outputs.append(concatenated)
            return final_outputs
        return _original_run(self, output_names, input_feed, run_options)

    ort.InferenceSession.run = patched_ort_run
    logging.info("ONNX Runtime patching applied successfully")
except ImportError as e:
    logging.debug(f"ONNX Runtime not available, skipping patching: {e}")
except Exception as e:
    logging.error(f"Failed to patch ONNX Runtime: {e}", exc_info=True)


def direct_soundfile_save(
    uri, src, sample_rate, channels_first=True, **kwargs
) -> None:
    torch, _ = _get_torch_modules()
    if src.device.type != "cpu":
        src = src.cpu()
    src = src.detach().float()
    if src.ndim == 1:
        pass
    elif src.ndim == 2:
        if channels_first:
            src = src.t()

    src_np = src.numpy()
    subtype = kwargs.get("subtype", None)
    sf.write(str(uri), src_np, sample_rate, subtype=subtype)


def _apply_torchaudio_patches():
    _, torchaudio = _get_torch_modules()
    torchaudio.save = direct_soundfile_save
    torchaudio.save_with_torchcodec = direct_soundfile_save
    try:
        import torchaudio._torchcodec

        torchaudio._torchcodec.save_with_torchcodec = direct_soundfile_save
    except ImportError:
        pass

    os.environ["TORCHAUDIO_BACKEND"] = "soundfile"
    os.environ["DEMUCS_NO_TORCHCODEC"] = "1"


def _prevent_clip(torch, wav, mode):
    if mode == "rescale":
        return wav / max(1.01 * wav.abs().max(), 1)
    if mode == "clamp":
        return wav.clamp(-1, 1)
    if mode == "tanh":
        return torch.tanh(wav)
    return wav


def _subtype_for(suffix, bits_per_sample, as_float):
    if suffix == "mp3":
        return None
    if suffix == "flac":
        return "PCM_24" if bits_per_sample > 16 else "PCM_16"
    if as_float:
        return "FLOAT"
    return {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}.get(
        bits_per_sample, "PCM_16"
    )


def patched_save_audio(
    wav,
    path,
    samplerate,
    bitrate=320,
    clip="rescale",
    bits_per_sample=16,
    as_float=False,
    preset=2,
) -> None:
    torch, _ = _get_torch_modules()
    if wav.dim() == 1:
        wav = wav[None]
    wav = _prevent_clip(torch, wav, clip)
    suffix = str(path).rsplit(".", 1)[-1].lower()
    if suffix not in ("wav", "flac", "mp3"):
        raise ValueError(f"Invalid suffix for path: .{suffix}")
    direct_soundfile_save(
        path,
        wav,
        samplerate,
        subtype=_subtype_for(suffix, bits_per_sample, as_float),
    )


def _apply_demucs_patches():
    demucs_apply, demucs_audio, demucs_separate = _get_demucs_modules()
    demucs_audio.save_audio = patched_save_audio
    demucs_separate.save_audio = patched_save_audio
    if hasattr(demucs_apply, "save_audio"):
        demucs_apply.save_audio = patched_save_audio
