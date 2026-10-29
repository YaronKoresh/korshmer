import hashlib
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from korshmer.audio_processing.validation import (
    validate_effect_integer,
    validate_effect_number,
)
from korshmer.constants import MODELS_DIR

logger = logging.getLogger(__name__)

MDX_MODEL_FILE = "UVR-MDX-NET-Voc_FT.onnx"
MDX_MODEL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
)
MDX_MODEL_SHA256 = (
    "534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111"
)
MDX_MODEL_BYTES = 66762490
MDX_MODELS_SUBDIR = "mdx"
VERIFIED_SUFFIX = ".verified"
PART_SUFFIX = ".part"
DOWNLOAD_TRIES = 3
DOWNLOAD_TIMEOUT_S = 30.0
DOWNLOAD_RETRY_DELAY_S = 2.0
DOWNLOAD_BLOCK_BYTES = 1 << 20
HASH_BLOCK_BYTES = 1 << 22
INPUT_PEAK = 0.9
SILENT_BIN_COUNT = 3
MDX_CHANNELS = 2
GPU_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
MAX_OVERLAP = 0.9


@dataclass(frozen=True)
class MdxParams:
    n_fft: int = 7680
    hop_length: int = 1024
    dim_f: int = 3072
    dim_t: int = 256
    overlap: float = 0.25
    sample_rate: int = 44100

    def __post_init__(self):
        validate_effect_integer(self.n_fft, "FFT size", 64, 65536)
        validate_effect_integer(self.hop_length, "Hop length", 1, 16384)
        validate_effect_integer(self.dim_f, "Frequency bins", 8, 32769)
        validate_effect_integer(self.dim_t, "Segment frames", 8, 4096)
        validate_effect_number(self.overlap, "Overlap", 0.0, MAX_OVERLAP)
        validate_effect_integer(self.sample_rate, "Sample rate", 8000, 192000)
        if self.dim_f > self.n_fft // 2 + 1:
            raise ValueError(
                f"Frequency bins must fit the FFT size, got {self.dim_f!r}"
            )
        if self.chunk_size <= self.n_fft:
            raise ValueError(
                f"Segment must be longer than the FFT, got {self.dim_t!r}"
            )

    @property
    def chunk_size(self):
        return self.hop_length * (self.dim_t - 1)

    @property
    def trim(self):
        return self.n_fft // 2

    @property
    def gen_size(self):
        return self.chunk_size - 2 * self.trim


def mdx_model_path(models_dir=None):
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    return base / MDX_MODELS_SUBDIR / MDX_MODEL_FILE


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_stamp(path):
    info = os.stat(path)
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _verified_sidecar(path):
    return Path(f"{path}{VERIFIED_SUFFIX}")


def _mark_verified(path):
    try:
        _verified_sidecar(path).write_text(
            json.dumps(_file_stamp(path)), encoding="utf-8"
        )
    except OSError as exc:
        logger.debug(f"Could not store the model verdict: {exc}")


def _is_verified(path):
    if not os.path.isfile(path):
        return False
    try:
        stored = json.loads(_verified_sidecar(path).read_text("utf-8"))
        if stored == _file_stamp(path):
            return True
    except (OSError, ValueError):
        pass
    if _sha256_of(path) != MDX_MODEL_SHA256:
        return False
    _mark_verified(path)
    return True


def _fetch(url, destination, progress, should_stop=None):
    request = urllib.request.Request(url, headers={"User-Agent": "korshmer"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as reply:
        total = int(reply.headers.get("Content-Length") or MDX_MODEL_BYTES)
        done = 0
        with open(destination, "wb") as handle:
            for block in iter(lambda: reply.read(DOWNLOAD_BLOCK_BYTES), b""):
                if should_stop is not None and should_stop():
                    raise InterruptedError("Stopped by user")
                handle.write(block)
                done += len(block)
                if progress is not None:
                    progress(
                        min(1.0, done / max(1, total)),
                        "Downloading the vocal model...",
                    )


def _download(path, progress, should_stop=None):
    os.makedirs(path.parent, exist_ok=True)
    partial = Path(f"{path}{PART_SUFFIX}")
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        try:
            _fetch(MDX_MODEL_URL, partial, progress, should_stop)
            if _sha256_of(partial) != MDX_MODEL_SHA256:
                raise ValueError("checksum mismatch")
            os.replace(partial, path)
            _mark_verified(path)
            return path
        except InterruptedError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:
            logger.warning(
                f"Vocal model download failed ({attempt}/{DOWNLOAD_TRIES}): "
                f"{exc}"
            )
            partial.unlink(missing_ok=True)
            if attempt < DOWNLOAD_TRIES:
                time.sleep(DOWNLOAD_RETRY_DELAY_S)
    return None


def ensure_mdx_model(
    models_dir=None, download=True, progress=None, should_stop=None
):
    try:
        path = mdx_model_path(models_dir)
        if _is_verified(path):
            return path
        if not download:
            return None
        return _download(path, progress, should_stop)
    except InterruptedError:
        raise
    except Exception as exc:
        logger.warning(f"Vocal model unavailable: {exc}")
        return None


def _default_providers():
    import onnxruntime as ort

    if GPU_PROVIDER in ort.get_available_providers():
        return [GPU_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


def _open_session(model_path, providers):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_mem_pattern = False
    wanted = list(providers) if providers else _default_providers()
    try:
        return ort.InferenceSession(
            str(model_path), sess_options=options, providers=wanted
        )
    except Exception as exc:
        if wanted == [CPU_PROVIDER]:
            raise
        logger.warning(f"Vocal model fell back to the CPU: {exc}")
        return ort.InferenceSession(
            str(model_path), sess_options=options, providers=[CPU_PROVIDER]
        )


def _spectrogram(chunk, window, params, torch):
    spec = torch.stft(
        chunk,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    frames = spec.shape[-1]
    real = torch.view_as_real(spec).permute(0, 3, 1, 2)
    real = real.reshape(2 * MDX_CHANNELS, -1, frames)
    return real[:, : params.dim_f, :]


def _waveform(prediction, window, params, torch):
    bins = params.n_fft // 2 + 1
    frames = prediction.shape[-1]
    padding = torch.zeros(
        2 * MDX_CHANNELS, bins - params.dim_f, frames, dtype=prediction.dtype
    )
    full = torch.cat([prediction, padding], dim=1)
    full = full.reshape(MDX_CHANNELS, 2, bins, frames)
    spec = torch.complex(full[:, 0], full[:, 1])
    return torch.istft(
        spec,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window=window,
        center=True,
        length=params.chunk_size,
    )


def _demix(mix, session, params, progress, should_stop):
    import torch

    chunk_size = params.chunk_size
    trim = params.trim
    length = mix.shape[1]
    pad = params.gen_size + trim - (length % params.gen_size)
    padded = np.concatenate(
        [
            np.zeros((MDX_CHANNELS, trim), dtype=np.float32),
            mix,
            np.zeros((MDX_CHANNELS, pad), dtype=np.float32),
        ],
        axis=1,
    )
    total = padded.shape[1]
    step = max(1, int((1.0 - params.overlap) * chunk_size))
    starts = range(0, total, step)
    result = np.zeros((MDX_CHANNELS, total), dtype=np.float32)
    divider = np.zeros(total, dtype=np.float32)
    window = torch.hann_window(params.n_fft, periodic=True)
    input_name = session.get_inputs()[0].name
    for index, start in enumerate(starts):
        if should_stop is not None and should_stop():
            raise InterruptedError("Stopped by user")
        end = min(start + chunk_size, total)
        part = np.zeros((MDX_CHANNELS, chunk_size), dtype=np.float32)
        part[:, : end - start] = padded[:, start:end]
        with torch.no_grad():
            spec = _spectrogram(torch.from_numpy(part), window, params, torch)
            spec[:, :SILENT_BIN_COUNT, :] = 0.0
            batch = spec.unsqueeze(0).numpy()
            prediction = session.run(None, {input_name: batch})[0]
            wave = _waveform(
                torch.from_numpy(np.asarray(prediction[0])),
                window,
                params,
                torch,
            ).numpy()
        span = end - start
        if params.overlap > 0:
            taper = np.hanning(span).astype(np.float32)
            result[:, start:end] += wave[:, :span] * taper
            divider[start:end] += taper
        else:
            result[:, start:end] += wave[:, :span]
            divider[start:end] += 1.0
        if progress is not None:
            progress((index + 1) / len(starts), "Separating the vocals...")
    vocals = result / np.maximum(divider, np.finfo(np.float32).tiny)
    return vocals[:, trim : trim + length]


def separate_vocals(
    mix,
    sample_rate,
    model_path,
    params=None,
    progress=None,
    should_stop=None,
    providers=None,
):
    params = params if params is not None else MdxParams()
    audio = np.asarray(mix, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != MDX_CHANNELS:
        raise ValueError(
            f"Mix must have shape (2, samples), got {tuple(audio.shape)!r}"
        )
    if sample_rate != params.sample_rate:
        raise ValueError(
            f"Mix must be at {params.sample_rate} Hz, got {sample_rate!r}"
        )
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(audio)
    session = _open_session(model_path, providers)
    vocals = _demix(
        audio * np.float32(INPUT_PEAK / peak),
        session,
        params,
        progress,
        should_stop,
    )
    return (vocals * np.float32(peak / INPUT_PEAK)).astype(np.float32)
