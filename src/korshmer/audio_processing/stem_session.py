import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from korshmer.audio.loader import check_audio_file_channels
from korshmer.constants import DATA_DIR, NOTE_NAMES, SCALES, STEM_TYPE_PATTERNS
from korshmer.utils.scale_names import sanitize_scale_name

logger = logging.getLogger(__name__)

CLEAN_VOCALS_DIR = "Editor"
CLEAN_VOCALS_NAME = "vocals_clean.wav"

STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_COLORS = {
    "vocals": "#EA4335",
    "drums": "#F29900",
    "bass": "#1E8E3E",
    "guitar": "#9334E6",
    "piano": "#1A73E8",
    "other": "#5F6368",
}
MANIFEST_NAME = "stems_session.json"
STEMS_FOLDER_NAME = "Stems"
STEM_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")
DEFAULT_TEMPO = 120.0
DEFAULT_ROOT = "C"
DEFAULT_SCALE = "Major"
TEMPO_HALVING_THRESHOLD = 185
TEMPO_DOUBLING_THRESHOLD = 75
MANIFEST_SEARCH_LEVELS = 3


KNOWN_STEM_MODELS = (
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_q",
    "mdx_extra",
    "mdx_extra_q",
)
MIX_CHANNELS = 2


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _timestamp_folder_name():
    now = datetime.now()
    return (
        now.strftime("%Y_%m_%d__%H_%M_%S")
        + f"__{str(now.timestamp()).replace('.', '')}"
    )


def _report(progress_callback, percent, message):
    if progress_callback is None:
        return
    try:
        progress_callback(int(percent), str(message))
    except Exception as exc:
        logger.debug(f"Progress callback failed: {exc}")


def _raise_if_stopped(check_stop_callback):
    if check_stop_callback is not None and check_stop_callback():
        raise InterruptedError("Stopped by user")


def normalize_detected_tempo(tempo):
    try:
        value = float(tempo) if tempo is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    if not np.isfinite(value) or value <= 0:
        return DEFAULT_TEMPO
    if value > TEMPO_HALVING_THRESHOLD:
        value = value / 2.0
    elif value < TEMPO_DOUBLING_THRESHOLD:
        value = value * 2.0
    return float(value)


def normalize_scale_name(scale):
    sanitized = sanitize_scale_name(scale)
    if not sanitized:
        return DEFAULT_SCALE
    name = sanitized
    if name in SCALES:
        return name
    lowered = name.lower()
    if lowered.startswith("minor"):
        return "Minor"
    if lowered.startswith("major"):
        return "Major"
    return name or DEFAULT_SCALE


def normalize_root_name(root):
    if isinstance(root, (int, np.integer)):
        return NOTE_NAMES[int(root) % len(NOTE_NAMES)]
    name = str(root or "").strip()
    if name in NOTE_NAMES:
        return name
    return DEFAULT_ROOT


def detect_session_key(audio_path):
    try:
        from korshmer.analysis import detect_key_and_mode

        root_idx, mode = detect_key_and_mode(str(audio_path))
        return normalize_root_name(root_idx), normalize_scale_name(mode)
    except Exception as exc:
        logger.warning(f"Key detection failed for {audio_path}: {exc}")
        return DEFAULT_ROOT, DEFAULT_SCALE


def detect_session_tempo(audio_path):
    try:
        from korshmer.audio_processing import core_analysis

        return normalize_detected_tempo(
            core_analysis.analyse_tempo(str(audio_path))
        )
    except Exception as exc:
        logger.warning(f"Tempo detection failed for {audio_path}: {exc}")
        return DEFAULT_TEMPO


def _clean_grid_offset(value):
    try:
        offset = float(value)
    except (TypeError, ValueError):
        return None
    return offset if np.isfinite(offset) else None


def _clean_music_regions(regions):
    cleaned = []
    for region in regions if isinstance(regions, (list, tuple)) else []:
        try:
            start, end = (float(edge) for edge in region)
        except (TypeError, ValueError):
            continue
        if np.isfinite(start) and np.isfinite(end) and end > start >= 0.0:
            cleaned.append([start, end])
    return cleaned


def _clean_music_cut(value):
    if not isinstance(value, dict):
        return None
    timeline = []
    for entry in value.get("timeline") or []:
        try:
            begin, end, position = (float(edge) for edge in entry)
        except (TypeError, ValueError):
            continue
        if (
            np.isfinite([begin, end, position]).all()
            and end > begin >= 0.0
            and position >= 0.0
        ):
            timeline.append([begin, end, position])
    source = value.get("source")
    if not timeline or not isinstance(source, str) or not source:
        return None
    if os.path.basename(source) != source:
        return None
    return {"timeline": timeline, "source": source}


@dataclass
class StemSession:
    source_path: str
    output_dir: str
    tempo: float = DEFAULT_TEMPO
    root: str = DEFAULT_ROOT
    scale: str = DEFAULT_SCALE
    stems: dict = field(default_factory=dict)
    midi_files: list = field(default_factory=list)
    created: str = field(default_factory=_now_iso)
    grid_offset: float | None = None
    music_regions: list = field(default_factory=list)
    separation: str = ""
    music_cut: dict | None = None

    def manifest_path(self):
        return os.path.join(self.output_dir, MANIFEST_NAME)

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        payload = asdict(self)
        payload["tempo"] = float(self.tempo)
        payload["stems"] = {str(k): str(v) for k, v in self.stems.items()}
        payload["midi_files"] = [str(p) for p in self.midi_files]
        payload["grid_offset"] = _clean_grid_offset(self.grid_offset)
        payload["music_regions"] = _clean_music_regions(self.music_regions)
        payload["separation"] = str(self.separation or "")
        payload["music_cut"] = _clean_music_cut(self.music_cut)
        path = self.manifest_path()
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)
        return path

    @classmethod
    def load(cls, manifest_path):
        manifest_path = str(manifest_path)
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid stems manifest: {manifest_path}")
        if not isinstance(payload.get("stems"), dict):
            payload["stems"] = {}
        if not isinstance(payload.get("midi_files"), list):
            payload["midi_files"] = []
        output_dir = os.path.dirname(os.path.abspath(manifest_path))
        session = cls(
            source_path=str(payload.get("source_path", "")),
            output_dir=output_dir,
            tempo=normalize_detected_tempo(payload.get("tempo")),
            root=normalize_root_name(payload.get("root")),
            scale=normalize_scale_name(payload.get("scale")),
            stems=_relocate_paths(
                payload.get("stems") or {},
                os.path.join(output_dir, STEMS_FOLDER_NAME),
                output_dir,
            ),
            midi_files=_relocate_path_list(
                payload.get("midi_files") or [],
                os.path.join(output_dir, "MIDI"),
                output_dir,
            ),
            created=str(payload.get("created") or _now_iso()),
            grid_offset=_clean_grid_offset(payload.get("grid_offset")),
            music_regions=_clean_music_regions(payload.get("music_regions")),
            separation=str(payload.get("separation") or ""),
            music_cut=_clean_music_cut(payload.get("music_cut")),
        )
        return session

    def cut_source_path(self):
        if not self.music_cut:
            return None
        path = os.path.join(self.output_dir, self.music_cut["source"])
        return path if os.path.isfile(path) else None

    @classmethod
    def find(cls, folder):
        if not folder:
            return None
        current = os.path.abspath(str(folder))
        if os.path.isfile(current):
            current = os.path.dirname(current)
        for _level in range(MANIFEST_SEARCH_LEVELS + 1):
            candidate = os.path.join(current, MANIFEST_NAME)
            if os.path.isfile(candidate):
                try:
                    return cls.load(candidate)
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ) as exc:
                    logger.warning(f"Unreadable manifest {candidate}: {exc}")
                    return None
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None

    def vocals_path(self):
        path = self.stems.get("vocals")
        if path and os.path.isfile(path):
            return path
        return None

    def ordered_stems(self):
        ordered = [
            (name, self.stems[name])
            for name in STEM_ORDER
            if name in self.stems
        ]
        extras = sorted(
            (name, path)
            for name, path in self.stems.items()
            if name not in STEM_ORDER
        )
        return ordered + extras


def _relocate_paths(mapping, *fallback_dirs):
    relocated = {}
    for name, path in mapping.items():
        relocated[str(name)] = _relocate_path(str(path), fallback_dirs)
    return relocated


def _relocate_path_list(paths, *fallback_dirs):
    return [_relocate_path(str(path), fallback_dirs) for path in paths]


def _relocate_path(path, fallback_dirs):
    if os.path.isfile(path):
        return path
    base_name = os.path.basename(path)
    for folder in fallback_dirs:
        candidate = os.path.join(folder, base_name)
        if os.path.isfile(candidate):
            return candidate
    return path


def classify_stem_name(filename):
    name = os.path.splitext(os.path.basename(str(filename)))[0].lower()
    if "no_vocal" in name or "no vocal" in name or "novocal" in name:
        return "other"
    for stem_type in STEM_ORDER:
        patterns = STEM_TYPE_PATTERNS.get(stem_type, [stem_type])
        if any(pattern in name for pattern in patterns):
            return stem_type
    for stem_type, patterns in STEM_TYPE_PATTERNS.items():
        if any(pattern in name for pattern in patterns):
            return stem_type
    return None


def find_stem_files(search_dir):
    found = {}
    if not search_dir or not os.path.isdir(str(search_dir)):
        return found
    for root, dirs, files in os.walk(str(search_dir)):
        dirs.sort()
        for file_name in sorted(files):
            lowered = file_name.lower()
            if not lowered.endswith(STEM_AUDIO_EXTENSIONS):
                continue
            if "trimmed" in lowered or "_clean" in lowered:
                continue
            stem_type = classify_stem_name(file_name)
            if stem_type is None:
                continue
            full_path = os.path.join(root, file_name)
            if stem_type not in found or (
                lowered.endswith(".wav")
                and not found[stem_type].lower().endswith(".wav")
            ):
                found[stem_type] = full_path
    return {
        name: found[name]
        for name in STEM_ORDER + sorted(set(found) - set(STEM_ORDER))
        if name in found
    }


def _read_audio_any(path):
    check_audio_file_channels(path)
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sample_rate)
    except Exception as exc:
        logger.debug(f"soundfile could not read {path}: {exc}")
    import librosa

    data, sample_rate = librosa.load(str(path), sr=None, mono=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        data = data.T
    return data, int(sample_rate)


def collect_stems(stems_path, out_folder, float32=False):
    collected = {}
    stems_path = str(stems_path)
    if os.path.isfile(stems_path):
        stem_type = classify_stem_name(stems_path)
        sources = {stem_type or Path(stems_path).stem: stems_path}
    else:
        sources = find_stem_files(stems_path)
    if not sources:
        return collected
    os.makedirs(out_folder, exist_ok=True)
    subtype = "FLOAT" if float32 else "PCM_24"
    for stem_name, source_path in sources.items():
        destination = os.path.join(out_folder, f"{stem_name}.wav")
        try:
            data, sample_rate = _read_audio_any(source_path)
            if data.size == 0 or sample_rate <= 0:
                logger.warning(f"Skipping empty stem {source_path}")
                continue
            if os.path.abspath(source_path) == os.path.abspath(destination):
                collected[stem_name] = destination
                continue
            sf.write(destination, data, sample_rate, subtype=subtype)
            collected[stem_name] = destination
        except Exception as exc:
            logger.error(f"Could not export stem {source_path}: {exc}")
    return collected


def validate_stem_model_name(model_name):
    if model_name not in KNOWN_STEM_MODELS:
        allowed = ", ".join(KNOWN_STEM_MODELS)
        raise ValueError(
            f"Unknown stem separation model {model_name!r}. "
            f"Allowed models: {allowed}."
        )
    return model_name


def _locate_demucs_output(job_dir, model_name, song_name):
    model_root = os.path.join(job_dir, model_name)
    preferred = os.path.join(model_root, song_name)
    if os.path.isdir(preferred):
        return preferred
    if os.path.isdir(model_root):
        subdirs = sorted(
            entry
            for entry in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, entry))
        )
        if subdirs:
            return os.path.join(model_root, subdirs[0])
        return model_root
    for root, _dirs, files in os.walk(job_dir):
        if any(f.lower().endswith(STEM_AUDIO_EXTENSIONS) for f in files):
            return root
    return job_dir


def _run_demucs_job(
    audio_path,
    job_dir,
    model_name,
    options=None,
    progress=None,
    should_stop=None,
):
    try:
        from korshmer import patches as demucs_patches
        from korshmer.audio_processing import core_analysis

        demucs_patches._apply_demucs_patches()
        demucs_patches._apply_torchaudio_patches()
        if options is None or model_name != "htdemucs_6s":
            core_analysis.run_demucs(audio_path, job_dir, model_name)
            return None
        from korshmer.audio_processing.separation_engine import (
            separate_with_label,
        )

        return separate_with_label(
            audio_path, job_dir, options, progress, should_stop
        )[1]
    except ImportError as exc:
        raise RuntimeError(
            "Stem separation requires the 'torch' and 'demucs' packages. "
            f"Install them to enable separation ({exc})."
        ) from exc
    except InterruptedError:
        raise
    except Exception as exc:
        message = str(exc)
        if not message.startswith("Demucs failed"):
            message = f"Demucs failed: {message}"
        raise RuntimeError(message) from exc


def default_session_output_root(audio_path):
    song_name = Path(str(audio_path)).stem
    return os.path.join(str(DATA_DIR), song_name, _timestamp_folder_name())


def separate_stems(
    audio_path,
    output_root=None,
    model_name="htdemucs_6s",
    progress_callback=None,
    check_stop_callback=None,
    float32=False,
    options=None,
):
    validate_stem_model_name(model_name)
    audio_path = str(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    song_name = Path(audio_path).stem
    if output_root is None:
        output_root = default_session_output_root(audio_path)
    output_root = str(output_root)
    _report(progress_callback, 5, "Preparing stem separation...")
    temp_work_dir = os.path.join(str(DATA_DIR), "_temp_work")
    os.makedirs(temp_work_dir, exist_ok=True)
    job_dir = tempfile.mkdtemp(prefix=f"{song_name}_sep_", dir=temp_work_dir)
    try:
        _raise_if_stopped(check_stop_callback)
        _report(progress_callback, 15, f"Separating stems ({model_name})...")
        separation = ""
        if options is None:
            _run_demucs_job(audio_path, job_dir, model_name)
        else:
            separation = _run_demucs_job(
                audio_path,
                job_dir,
                model_name,
                options=options,
                progress=lambda fraction, text: _report(
                    progress_callback, 15 + 70 * fraction, text
                ),
                should_stop=check_stop_callback,
            )
        _raise_if_stopped(check_stop_callback)
        _report(progress_callback, 85, "Collecting stems...")
        stems_dir = _locate_demucs_output(job_dir, model_name, song_name)
        stems = collect_stems(
            stems_dir,
            os.path.join(output_root, STEMS_FOLDER_NAME),
            float32=float32,
        )
        if not stems:
            raise RuntimeError(
                "Demucs produced no stems. Check the Demucs installation "
                "and the input audio file."
            )
        _report(progress_callback, 92, "Detecting tempo and key...")
        tempo = detect_session_tempo(audio_path)
        root, scale = detect_session_key(audio_path)
        session = StemSession(
            source_path=os.path.abspath(audio_path),
            output_dir=output_root,
            tempo=tempo,
            root=root,
            scale=scale,
            stems=stems,
            midi_files=[],
            separation=separation or "",
        )
        session.save()
        _report(progress_callback, 100, f"Stems ready: {len(stems)} stems")
        return session
    finally:
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)


def cleaned_vocals_path(session):
    source = session.vocals_path()
    if not source:
        return None
    target = os.path.join(
        str(session.output_dir), CLEAN_VOCALS_DIR, CLEAN_VOCALS_NAME
    )
    try:
        if os.path.isfile(target) and os.path.getmtime(
            target
        ) >= os.path.getmtime(source):
            return target
        from korshmer.audio_processing.vocal_cleanup import clean_vocal

        data, sample_rate = _read_audio_any(source)
        cleaned = clean_vocal(data, sample_rate)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        partial = f"{target}.part"
        sf.write(partial, cleaned, sample_rate, subtype="FLOAT", format="WAV")
        os.replace(partial, target)
        return target
    except Exception as exc:
        logger.warning(f"Could not clean vocals {source}: {exc}")
        return source


def vocal_gain_for_file(path):
    try:
        from korshmer.audio_processing.vocal_cleanup import vocal_clip_gain

        data, _rate = _read_audio_any(path)
        return vocal_clip_gain(data)
    except Exception as exc:
        logger.warning(f"Could not measure vocals {path}: {exc}")
        return 1.0


def _to_stereo(data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.shape[1] == 1:
        return np.repeat(data, MIX_CHANNELS, axis=1)
    if data.shape[1] > MIX_CHANNELS:
        return np.ascontiguousarray(data[:, :MIX_CHANNELS])
    return data


def _resample_stereo(data, source_rate, target_rate):
    if source_rate == target_rate or data.shape[0] == 0:
        return data
    import librosa

    resampled = librosa.resample(
        np.ascontiguousarray(data.T),
        orig_sr=int(source_rate),
        target_sr=int(target_rate),
    )
    return np.ascontiguousarray(resampled.T.astype(np.float32))


def _load_stereo_at_rate(path, sample_rate):
    data, source_rate = _read_audio_any(path)
    data = _to_stereo(data)
    return _resample_stereo(data, source_rate, sample_rate)


def _clip_gain(clip):
    try:
        gain = float(clip.get("gain", 1.0))
    except (TypeError, ValueError):
        gain = 1.0
    if not np.isfinite(gain) or gain < 0:
        gain = 0.0
    return gain


def _clip_start_samples(clip, sample_rate):
    try:
        start = float(clip.get("start", 0.0))
    except (TypeError, ValueError):
        start = 0.0
    if not np.isfinite(start) or start < 0:
        start = 0.0
    return int(round(start * sample_rate))


def mix_render_with_clips(
    render_path, clips, out_path, sample_rate=44100, headroom_db=-1.0
):
    sample_rate = int(sample_rate) if sample_rate and sample_rate > 0 else 44100
    layers = []
    if render_path and os.path.isfile(str(render_path)):
        try:
            layers.append((0, _load_stereo_at_rate(render_path, sample_rate)))
        except Exception as exc:
            logger.error(f"Could not load render {render_path}: {exc}")
    for clip in clips or []:
        if not isinstance(clip, dict) or clip.get("muted"):
            continue
        path = clip.get("path")
        if not path or not os.path.isfile(str(path)):
            logger.warning(f"Clip file missing, skipped: {path}")
            continue
        try:
            data = _load_stereo_at_rate(path, sample_rate)
        except Exception as exc:
            logger.warning(f"Clip could not be read, skipped: {path}: {exc}")
            continue
        gain = _clip_gain(clip)
        if gain != 1.0:
            data = data * np.float32(gain)
        layers.append((_clip_start_samples(clip, sample_rate), data))
    layers = [(offset, data) for offset, data in layers if data.shape[0] > 0]
    if not layers:
        return None
    total_length = max(offset + data.shape[0] for offset, data in layers)
    mix = np.zeros((total_length, MIX_CHANNELS), dtype=np.float32)
    for offset, data in layers:
        mix[offset : offset + data.shape[0]] += data
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    ceiling = float(10 ** (float(headroom_db) / 20.0))
    if peak > ceiling and peak > 0:
        mix *= np.float32(ceiling / peak)
    out_path = str(out_path)
    if not out_path.lower().endswith(".wav"):
        out_path = f"{os.path.splitext(out_path)[0]}.wav"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        sf.write(out_path, mix, sample_rate, subtype="FLOAT")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.error(f"Could not write mix {out_path}: {exc}")
        return None
    return out_path


def build_session_from_processing(
    source_path,
    output_root,
    tempo,
    root,
    scale,
    stems,
    midi_files,
    grid_offset=None,
    music_regions=None,
    separation="",
    music_cut=None,
):
    return StemSession(
        source_path=os.path.abspath(str(source_path)),
        output_dir=str(output_root),
        tempo=normalize_detected_tempo(tempo),
        root=normalize_root_name(root),
        scale=normalize_scale_name(scale),
        stems={str(k): str(v) for k, v in (stems or {}).items()},
        midi_files=[str(p) for p in (midi_files or [])],
        grid_offset=_clean_grid_offset(grid_offset),
        music_regions=_clean_music_regions(music_regions),
        separation=str(separation or ""),
        music_cut=_clean_music_cut(music_cut),
    )
