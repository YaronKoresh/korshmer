import inspect
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from korshmer.audio_processing import mdx_vocals
from korshmer.audio_processing.stem_session import (
    _load_stereo_at_rate,
    _locate_demucs_output,
)

logger = logging.getLogger(__name__)

QUALITY_PRESETS = {
    "fast": (1, 0.1, 0.25),
    "balanced": (1, 0.25, 0.25),
    "best": (2, 0.25, 0.5),
}
VOCAL_BACKENDS = ("auto", "mdx", "demucs_ft", "demucs_6s")
DEMUCS_MODEL = "htdemucs_6s"
VOCAL_SPECIALIST_MODEL = "htdemucs_ft"
VOCAL_SPECIALIST_INDEX = 3
SAMPLE_RATE = 44100
MIX_DIR = "_mix"
ACCOMPANIMENT_DIR = "_accompaniment"
VOCALS_FILE = "vocals.wav"
SCRATCH_SUBTYPE = "FLOAT"
STEM_SUBTYPE = "PCM_24"
DECODE_END = 0.05
DOWNLOAD_END = 0.10
VOCALS_END = 0.45
CLIP_MARGIN = 1.01
MIN_STD = 1e-8


@dataclass(frozen=True)
class SeparationOptions:
    quality: str = "best"
    vocal_backend: str = "auto"

    def __post_init__(self):
        if self.quality not in QUALITY_PRESETS:
            raise ValueError(
                f"Quality must be one of {tuple(QUALITY_PRESETS)}, "
                f"got {self.quality!r}"
            )
        if self.vocal_backend not in VOCAL_BACKENDS:
            raise ValueError(
                f"Vocal backend must be one of {VOCAL_BACKENDS}, "
                f"got {self.vocal_backend!r}"
            )


def pick_vocal_backend(options, mdx_available):
    if options.quality == "fast" or options.vocal_backend == "demucs_6s":
        return "demucs_6s"
    if options.vocal_backend == "demucs_ft":
        return "demucs_ft"
    return "mdx" if mdx_available else "demucs_ft"


def _reporter(callback):
    state = {"last": 0.0}

    def report(fraction, text):
        if fraction is not None:
            state["last"] = min(1.0, max(state["last"], float(fraction)))
        if callback is not None:
            callback(state["last"], text)

    return report


def _window(report, start, end):
    return lambda fraction, text: report(start + (end - start) * fraction, text)


def _check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise InterruptedError("Stopped by user")


def _fall_back(report, reason, exc, next_step):
    logger.warning(f"{reason}: {exc}; {next_step}")
    report(None, f"{reason}; {next_step}")


def _mdx_vocals(mix, mdx_overlap, report, should_stop):
    model = mdx_vocals.ensure_mdx_model(
        progress=_window(report, DECODE_END, DOWNLOAD_END),
        should_stop=should_stop,
    )
    if model is None:
        raise FileNotFoundError("the vocal model could not be downloaded")
    return mdx_vocals.separate_vocals(
        mix,
        SAMPLE_RATE,
        model,
        mdx_vocals.MdxParams(overlap=mdx_overlap),
        progress=_window(report, DOWNLOAD_END, VOCALS_END),
        should_stop=should_stop,
    )


def _specialist_vocals(mix, shifts, overlap, should_stop):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from korshmer.processing.device import get_device_manager

    _check_stop(should_stop)
    model = get_model(VOCAL_SPECIALIST_MODEL).models[VOCAL_SPECIALIST_INDEX]
    _check_stop(should_stop)
    parameters = inspect.signature(apply_model).parameters
    extra = {}
    if "callback" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        extra["callback"] = lambda _info: _check_stop(should_stop)
    wav = torch.from_numpy(np.ascontiguousarray(mix))
    reference = wav.mean(0)
    scale = max(float(reference.std()), MIN_STD)
    wav = (wav - reference.mean()) / scale
    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            shifts=shifts,
            overlap=overlap,
            split=True,
            device=get_device_manager().select_device(),
            **extra,
        )[0]
    sources = sources * scale + reference.mean()
    return sources[model.sources.index("vocals")].cpu().numpy()


def _extract_vocals(mix, options, report, should_stop):
    shifts, overlap, mdx_overlap = QUALITY_PRESETS[options.quality]
    backend = pick_vocal_backend(options, True)
    if backend == "mdx":
        report(DECODE_END, "Separating the vocals...")
        try:
            return _mdx_vocals(mix, mdx_overlap, report, should_stop), "mdx"
        except InterruptedError:
            raise
        except Exception as exc:
            backend = pick_vocal_backend(options, False)
            _fall_back(
                report,
                "Vocal model unavailable",
                exc,
                "using the Demucs vocal model",
            )
    _check_stop(should_stop)
    if backend == "demucs_ft":
        try:
            vocals = _specialist_vocals(mix, shifts, overlap, should_stop)
            return vocals, VOCAL_SPECIALIST_MODEL
        except InterruptedError:
            raise
        except Exception as exc:
            _fall_back(
                report,
                "Demucs vocal model unavailable",
                exc,
                "using one pass on the full mix",
            )
    return None, ""


def _write_scratch(data, job_dir, folder, track):
    directory = os.path.join(job_dir, folder)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{track}.wav")
    sf.write(
        path,
        np.ascontiguousarray(data.T),
        SAMPLE_RATE,
        subtype=SCRATCH_SUBTYPE,
    )
    return path


def _write_vocals(stems_dir, vocals):
    target = os.path.join(stems_dir, VOCALS_FILE)
    subtype = STEM_SUBTYPE
    if os.path.isfile(target):
        subtype = sf.info(target).subtype
    peak = float(np.max(np.abs(vocals))) if vocals.size else 0.0
    vocals = vocals / np.float32(max(CLIP_MARGIN * peak, 1.0))
    sf.write(
        target, np.ascontiguousarray(vocals.T), SAMPLE_RATE, subtype=subtype
    )


def separate_with_label(
    audio_path, job_dir, options=None, progress=None, should_stop=None
):
    options = options if options is not None else SeparationOptions()
    shifts, overlap = QUALITY_PRESETS[options.quality][:2]
    report = _reporter(progress)
    track = Path(str(audio_path)).stem
    report(0.0, "Reading the audio...")
    _check_stop(should_stop)
    mix = np.ascontiguousarray(
        _load_stereo_at_rate(str(audio_path), SAMPLE_RATE).T
    )
    report(DECODE_END, "Audio ready")
    try:
        vocals, vocal_model = _extract_vocals(mix, options, report, should_stop)
        if vocals is None:
            source = _write_scratch(mix, job_dir, MIX_DIR, track)
        else:
            source = _write_scratch(
                mix - vocals, job_dir, ACCOMPANIMENT_DIR, track
            )
        _check_stop(should_stop)
        report(VOCALS_END, "Separating the instruments...")
        from korshmer.audio_processing import core_analysis

        core_analysis.run_demucs(
            source, job_dir, DEMUCS_MODEL, shifts=shifts, overlap=overlap
        )
        _check_stop(should_stop)
        if not os.path.isdir(os.path.join(job_dir, DEMUCS_MODEL)):
            raise RuntimeError("Demucs failed: it wrote no stems")
        stems_dir = _locate_demucs_output(job_dir, DEMUCS_MODEL, track)
        if vocals is not None:
            _write_vocals(stems_dir, vocals)
        report(1.0, "Stems ready")
        return stems_dir, "+".join(filter(None, [vocal_model, DEMUCS_MODEL]))
    finally:
        for folder in (MIX_DIR, ACCOMPANIMENT_DIR):
            shutil.rmtree(os.path.join(job_dir, folder), ignore_errors=True)
