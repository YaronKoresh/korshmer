import json
import logging
import os
import shutil
import struct
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from korshmer.audio_processing.validation import (
    validate_bit_depth,
    validate_normalize_target_db,
)
from korshmer.constants import (
    EXPORT_FFMPEG_TIMEOUT_SECONDS,
    EXPORT_LUFS_TARGET_MIN,
    EXPORT_TRUE_PEAK_CEILING_DB,
    EXPORT_TRUE_PEAK_CEILING_MIN_DB,
    EXPORT_TRUE_PEAK_CHUNK_FRAMES,
    EXPORT_TRUE_PEAK_OVERSAMPLING,
    EXPORT_TRUE_PEAK_PAD_FRAMES,
)
from korshmer.project_management.file_io_log import logged_file_op, logger
from korshmer.utils.filesystem import is_safe_path

NORMALIZE_TARGET_DB_KEYS: dict[str, tuple[str, float]] = {
    "peak": ("target_peak", -0.1),
    "rms": ("target_rms", -18),
}


def validate_preset_normalize_target(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError(
            f"Export preset must be a mapping of settings, got {config!r}"
        )
    mode = config.get("normalize", "none")
    if not isinstance(mode, str) or mode not in NORMALIZE_TARGET_DB_KEYS:
        return
    key, default = NORMALIZE_TARGET_DB_KEYS[mode]
    try:
        validate_normalize_target_db(config.get(key, default))
    except ValueError as e:
        raise ValueError(
            f"Invalid '{key}' for normalize mode '{mode}': {e}"
        ) from e


def validate_preset_bit_depth(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError(
            f"Export preset must be a mapping of settings, got {config!r}"
        )
    if "bit_depth" not in config:
        return
    try:
        validate_bit_depth(config["bit_depth"])
    except ValueError as e:
        raise ValueError(f"Invalid 'bit_depth': {e}") from e


ENCODER_RANGES: dict[str, tuple[float, float]] = {
    "bitrate": (64, 320),
    "compression": (0, 8),
}
QUALITY_RANGES: dict[str, tuple[float, float]] = {
    "ogg": (-1, 10),
    "mp3": (0, 9),
}
AAC_FORMATS = ("m4a", "aac")
PCM_FORMATS = ("wav", "flac", "aiff")
PCM_SUBTYPE_BITS = {"PCM_16": 16, "PCM_24": 24}
SOURCE_PCM_BITS = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24}
EXPORT_THREAD_NAME = "korshmer-export"
RIFF_INFO_TAGS = {
    "title": b"INAM",
    "artist": b"IART",
    "album": b"IPRD",
    "genre": b"IGNR",
    "date": b"ICRD",
    "comment": b"ICMT",
}
MP4_TAGS = {
    "title": "\xa9nam",
    "artist": "\xa9ART",
    "album": "\xa9alb",
    "genre": "\xa9gen",
    "date": "\xa9day",
    "comment": "\xa9cmt",
}


def _require_range(name: str, value: Any, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= value <= high
    ):
        raise ValueError(
            f"Invalid '{name}': must be a number from {low} to {high}, "
            f"got {value!r}"
        )


def validate_preset_encoder(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError(
            f"Export preset must be a mapping of settings, got {config!r}"
        )
    for name, (low, high) in ENCODER_RANGES.items():
        if name in config:
            _require_range(name, config[name], low, high)
    quality_range = QUALITY_RANGES.get(config.get("format", "wav"))
    if "quality" in config and quality_range is not None:
        _require_range("quality", config["quality"], *quality_range)


def validate_true_peak_ceiling(value: Any) -> float:
    _require_range(
        "true_peak_ceiling", value, EXPORT_TRUE_PEAK_CEILING_MIN_DB, 0
    )
    return float(value)


def validate_target_lufs(value: Any) -> float:
    _require_range("target_lufs", value, EXPORT_LUFS_TARGET_MIN, 0)
    return float(value)


def validate_preset_finishing(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError(
            f"Export preset must be a mapping of settings, got {config!r}"
        )
    if config.get("normalize") == "lufs" and "target_lufs" in config:
        validate_target_lufs(config["target_lufs"])
    if "true_peak_ceiling" in config:
        validate_true_peak_ceiling(config["true_peak_ceiling"])
    if "dither" in config and not isinstance(config["dither"], bool):
        raise ValueError(
            f"Invalid 'dither': must be true or false, got {config['dither']!r}"
        )
    seed = config.get("dither_seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError(
            f"Invalid 'dither_seed': must be a whole number of 0 or more, "
            f"got {seed!r}"
        )


def true_peak(audio_data: np.ndarray) -> float:
    from scipy import signal

    frames = len(audio_data)
    channels = audio_data.reshape(frames, -1)
    up = EXPORT_TRUE_PEAK_OVERSAMPLING
    chunk = EXPORT_TRUE_PEAK_CHUNK_FRAMES
    pad = EXPORT_TRUE_PEAK_PAD_FRAMES
    peak = 0.0
    for channel in range(channels.shape[1]):
        for first in range(0, frames, chunk):
            low = max(0, first - pad)
            last = min(frames, first + chunk)
            piece = np.asarray(
                channels[low : min(frames, last + pad), channel],
                dtype=np.float64,
            )
            if len(piece) > 1:
                piece = signal.resample_poly(piece, up, 1)
                piece = piece[(first - low) * up : (last - low) * up]
            peak = max(peak, float(np.max(np.abs(piece))))
    return peak


def quantize_pcm(
    audio_data: np.ndarray,
    bit_depth: int,
    dither: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    full_scale = 2 ** (bit_depth - 1)
    scaled = np.asarray(audio_data, dtype=np.float64) * full_scale
    if dither:
        rng = np.random.default_rng(seed)
        scaled += rng.random(scaled.shape) - rng.random(scaled.shape)
    steps = np.clip(np.rint(scaled), -full_scale, full_scale - 1)
    samples: np.ndarray
    if bit_depth == 16:
        samples = steps.astype(np.int16)
    else:
        samples = steps.astype(np.int32) * 2 ** (32 - bit_depth)
    return samples


def write_riff_info(path: Path, metadata: dict[str, str]) -> bool:
    entries = b""
    for key, tag in RIFF_INFO_TAGS.items():
        value = metadata.get(key)
        if not value:
            continue
        text = str(value).encode("utf-8") + b"\0"
        entries += tag + struct.pack("<I", len(text)) + text
        entries += b"\0" * (len(text) % 2)
    if not entries:
        return False
    chunk = b"LIST" + struct.pack("<I", len(entries) + 4) + b"INFO" + entries
    with open(path, "r+b") as handle:
        header = handle.read(12)
        if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            return False
        size = handle.seek(0, os.SEEK_END)
        if size % 2:
            handle.write(b"\0")
            size += 1
        if size + len(chunk) - 8 > 0xFFFFFFFF:
            return False
        handle.write(chunk)
        handle.seek(4)
        handle.write(struct.pack("<I", size + len(chunk) - 8))
    return True


def validate_export_preset(config: dict[str, Any]) -> None:
    validate_preset_normalize_target(config)
    validate_preset_bit_depth(config)
    validate_preset_encoder(config)
    validate_preset_finishing(config)


class ExportManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.exports_dir = self.base_dir / "saved_data" / "exports"
        self.presets_file = self.base_dir / "saved_data" / "export_presets.json"

        self._ensure_directories()
        self.export_presets = self._load_presets()
        self.export_queue: deque[dict[str, Any]] = deque()
        self.is_exporting = False
        self.current_export: dict[str, Any] | None = None
        self._queue_lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._last_error: str | None = None

    def _ensure_directories(self):
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def _load_presets(self) -> dict[str, dict[str, Any]]:
        default_presets = {
            "Spotify": {
                "format": "ogg",
                "sample_rate": 44100,
                "bit_depth": 16,
                "normalize": "lufs",
                "target_lufs": -14,
                "true_peak_ceiling": EXPORT_TRUE_PEAK_CEILING_DB,
                "metadata": True,
            },
            "YouTube": {
                "format": "wav",
                "sample_rate": 48000,
                "bit_depth": 24,
                "normalize": "lufs",
                "target_lufs": -13,
                "true_peak_ceiling": EXPORT_TRUE_PEAK_CEILING_DB,
                "dither": False,
                "metadata": True,
            },
            "CD Quality": {
                "format": "wav",
                "sample_rate": 44100,
                "bit_depth": 16,
                "normalize": "peak",
                "target_peak": -0.1,
                "dither": True,
                "dither_seed": None,
                "metadata": True,
            },
            "Mastering": {
                "format": "wav",
                "sample_rate": 96000,
                "bit_depth": 32,
                "normalize": "none",
                "metadata": True,
            },
            "MP3 High": {
                "format": "mp3",
                "bitrate": 320,
                "sample_rate": 44100,
                "normalize": "rms",
                "target_rms": -18,
                "true_peak_ceiling": EXPORT_TRUE_PEAK_CEILING_DB,
                "metadata": True,
            },
            "MP3 Medium": {
                "format": "mp3",
                "bitrate": 192,
                "sample_rate": 44100,
                "normalize": "rms",
                "target_rms": -18,
                "true_peak_ceiling": EXPORT_TRUE_PEAK_CEILING_DB,
                "metadata": True,
            },
            "FLAC Lossless": {
                "format": "flac",
                "sample_rate": 44100,
                "bit_depth": 24,
                "normalize": "peak",
                "target_peak": -0.1,
                "metadata": True,
            },
            "FLAC Hi-Res": {
                "format": "flac",
                "sample_rate": 96000,
                "bit_depth": 24,
                "normalize": "none",
                "metadata": True,
            },
            "AIFF Studio": {
                "format": "aiff",
                "sample_rate": 48000,
                "bit_depth": 24,
                "normalize": "none",
                "metadata": True,
            },
        }

        if self.presets_file.exists():
            try:
                with logged_file_op("Load export presets", self.presets_file):
                    with open(self.presets_file, encoding="utf-8") as f:
                        loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"expected a JSON object, got {type(loaded).__name__}"
                    )
                for name, config in loaded.items():
                    try:
                        validate_export_preset(config)
                    except ValueError as e:
                        logging.warning(
                            f"Skipping export preset {name!r} from "
                            f"{self.presets_file.name}: {e}"
                        )
                        continue
                    default_presets[name] = config
            except Exception as e:
                logging.error(f"Failed to load export presets: {e}")

        return default_presets

    def _save_presets(self):
        try:
            with logged_file_op("Save export presets", self.presets_file):
                tmp = (
                    self.presets_file.parent / f".{self.presets_file.name}.tmp"
                )
                try:
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(self.export_presets, f, indent=2)
                    os.replace(tmp, self.presets_file)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
        except Exception as e:
            logging.error(f"Failed to save export presets: {e}")

    def add_to_queue(
        self,
        source_path: Path,
        output_path: Path | None = None,
        preset: str = "CD Quality",
        metadata: dict[str, str] | None = None,
        callback: Callable | None = None,
    ) -> bool:
        try:
            if not is_safe_path(source_path) or (
                output_path is not None and not is_safe_path(output_path)
            ):
                logging.error(
                    f"Unsafe export path: {source_path!r} -> {output_path!r}"
                )
                return False

            source_path = Path(source_path)

            if not source_path.exists():
                logging.error(f"Source file does not exist: {source_path}")
                return False

            if output_path is None:
                preset_config = self.export_presets.get(
                    preset, self.export_presets["CD Quality"]
                )
                ext = preset_config.get("format", "wav")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = (
                    self.exports_dir / f"{source_path.stem}_{timestamp}.{ext}"
                )
            else:
                output_path = Path(output_path)

            export_job = {
                "source": str(source_path),
                "output": str(output_path),
                "preset": preset,
                "metadata": metadata or {},
                "callback": callback,
                "status": "queued",
                "added": datetime.now().isoformat(),
            }

            with self._queue_lock:
                self.export_queue.append(export_job)
            return True

        except Exception as e:
            logging.error(f"Failed to add to export queue: {e}")
            return False

    def process_queue(self):
        with self._export_lock:
            if self.is_exporting:
                return

            with self._queue_lock:
                if not self.export_queue:
                    return

            self.is_exporting = True

        thread = threading.Thread(
            target=self._process_queue_thread,
            name=EXPORT_THREAD_NAME,
            daemon=True,
        )
        thread.start()

    def _process_queue_thread(self):
        try:
            while True:
                with self._queue_lock:
                    if not self.export_queue:
                        break
                    job = self.export_queue.popleft()
                self._run_job(job)
        finally:
            with self._export_lock:
                self.is_exporting = False
                self.current_export = None

    def _run_job(self, job: dict[str, Any]) -> None:
        self.current_export = job
        job["status"] = "processing"
        try:
            success = self._export_file(
                Path(job["source"]),
                Path(job["output"]),
                job["preset"],
                job["metadata"],
            )
            if not success:
                job["error"] = self._last_error or "Export failed"
        except Exception as e:
            logging.error(f"Export failed: {e}")
            success = False
            job["error"] = str(e)
        job["status"] = "completed" if success else "failed"
        if job["callback"]:
            try:
                job["callback"](job, success)
            except Exception as e:
                logging.error(f"Export listener failed: {e}")

    def _export_file(
        self, source: Path, output: Path, preset: str, metadata: dict[str, str]
    ) -> bool:
        self._last_error = None
        try:
            import soundfile as sf

            preset_config = self.export_presets.get(
                preset, self.export_presets["CD Quality"]
            )

            with logged_file_op("Read export source", source):
                audio_data, source_sr = sf.read(str(source))
                source_bits = SOURCE_PCM_BITS.get(sf.info(str(source)).subtype)

            target_sr = preset_config.get("sample_rate", source_sr)
            if target_sr != source_sr:
                audio_data = self._resample(audio_data, source_sr, target_sr)

            normalize_mode = preset_config.get("normalize", "none")
            if target_sr != source_sr or normalize_mode != "none":
                source_bits = None
            if normalize_mode != "none":
                try:
                    audio_data = self._normalize_audio(
                        audio_data, normalize_mode, preset_config, target_sr
                    )
                except ValueError as e:
                    return self._refuse(
                        f"invalid normalize target in export preset "
                        f"{preset!r}: {e}"
                    )

            output_format = preset_config.get("format", "wav")

            bit_depth = preset_config.get("bit_depth", 16)
            if output_format in PCM_FORMATS:
                try:
                    bit_depth = validate_bit_depth(bit_depth)
                except ValueError as e:
                    return self._refuse(
                        f"invalid bit depth in export preset {preset!r}: {e}"
                    )

            with logged_file_op("Export audio", output):
                self._write_output(
                    output,
                    audio_data,
                    target_sr,
                    output_format,
                    bit_depth,
                    preset_config,
                    metadata,
                    source_bits,
                )
                if preset_config.get("metadata", True) and metadata:
                    self._embed_metadata(output, metadata)

            if not output.exists():
                logger.warning("Export audio left no file: %s", output)
                self._last_error = f"The export wrote no file at {output}"
                return False

            return True

        except Exception as e:
            return self._refuse(str(e))

    def _refuse(self, reason: str) -> bool:
        self._last_error = reason
        logging.error(f"Export file failed: {reason}")
        return False

    def _write_output(
        self,
        output: Path,
        audio_data: np.ndarray,
        target_sr: int,
        output_format: str,
        bit_depth: int,
        preset_config: dict[str, Any],
        metadata: dict[str, str],
        source_bits: int | None = None,
    ) -> None:
        import soundfile as sf

        if output_format in PCM_FORMATS:
            subtype = self._get_subtype(bit_depth, output_format)
            if subtype in PCM_SUBTYPE_BITS:
                pcm_bits = PCM_SUBTYPE_BITS[subtype]
                reduces_depth = source_bits is None or source_bits > pcm_bits
                audio_data = quantize_pcm(
                    audio_data,
                    pcm_bits,
                    preset_config.get(
                        "dither", pcm_bits == 16 and reduces_depth
                    ),
                    preset_config.get("dither_seed"),
                )
            fmt_map = {"flac": "FLAC", "aiff": "AIFF"}
            write_kwargs = {"subtype": subtype}
            if output_format in fmt_map:
                write_kwargs["format"] = fmt_map[output_format]
            if output_format == "flac" and "compression" in preset_config:
                write_kwargs["compression_level"] = (
                    preset_config["compression"] / 8
                )
            sf.write(str(output), audio_data, target_sr, **write_kwargs)

        elif output_format == "ogg":
            ogg_kwargs = {}
            if "quality" in preset_config:
                ogg_kwargs["compression_level"] = (
                    10 - preset_config["quality"]
                ) / 11
            sf.write(
                str(output),
                audio_data,
                target_sr,
                format="OGG",
                subtype="VORBIS",
                **ogg_kwargs,
            )

        elif output_format == "mp3":
            temp_wav = output.with_suffix(".temp.wav")
            try:
                sf.write(
                    str(temp_wav),
                    quantize_pcm(audio_data, 16),
                    target_sr,
                    subtype="PCM_16",
                )

                bitrate = preset_config.get("bitrate", 320)
                if preset_config.get("vbr", False):
                    self._convert_to_mp3(
                        temp_wav,
                        output,
                        bitrate,
                        metadata,
                        vbr_quality=preset_config.get("quality", 2),
                    )
                else:
                    self._convert_to_mp3(temp_wav, output, bitrate, metadata)
            finally:
                if temp_wav.exists():
                    temp_wav.unlink()

        elif output_format in AAC_FORMATS:
            temp_wav = output.with_suffix(".temp.wav")
            try:
                sf.write(
                    str(temp_wav),
                    quantize_pcm(audio_data, 16),
                    target_sr,
                    subtype="PCM_16",
                )
                self._convert_to_aac(
                    temp_wav, output, preset_config.get("bitrate", 256)
                )
            finally:
                if temp_wav.exists():
                    temp_wav.unlink()

    def _resample(
        self, audio_data: np.ndarray, source_sr: int, target_sr: int
    ) -> np.ndarray:
        try:
            import librosa

            result: np.ndarray = librosa.resample(
                audio_data.T, orig_sr=source_sr, target_sr=target_sr
            ).T
        except (ImportError, ValueError, TypeError) as e:
            raise RuntimeError(
                f"Could not convert the sample rate from {source_sr} Hz to "
                f"{target_sr} Hz: {e}"
            ) from e
        return result

    def _normalize_audio(
        self,
        audio_data: np.ndarray,
        mode: str,
        config: dict[str, Any],
        sample_rate: int = 44100,
    ) -> np.ndarray:
        if mode not in ("peak", "rms", "lufs"):
            return audio_data
        ceiling_db: float | None = None
        if "true_peak_ceiling" in config:
            ceiling_db = validate_true_peak_ceiling(config["true_peak_ceiling"])
        elif mode != "peak":
            ceiling_db = EXPORT_TRUE_PEAK_CEILING_DB

        if mode == "peak":
            target_peak = validate_normalize_target_db(
                config.get("target_peak", -0.1)
            )
            current_peak = np.max(np.abs(audio_data))
            if current_peak > 0:
                target_linear = 10 ** (target_peak / 20)
                audio_data = audio_data * (target_linear / current_peak)

        elif mode == "rms":
            target_rms_db = validate_normalize_target_db(
                config.get("target_rms", -18)
            )
            current_rms = np.sqrt(np.mean(audio_data**2))
            if current_rms > 0:
                target_rms = 10 ** (target_rms_db / 20)
                audio_data = audio_data * (target_rms / current_rms)

        else:
            target_lufs = validate_target_lufs(config.get("target_lufs", -14))
            current_lufs = self._measure_lufs(audio_data, sample_rate)
            if np.isfinite(current_lufs):
                audio_data = audio_data * (
                    10 ** ((target_lufs - current_lufs) / 20)
                )

        if ceiling_db is not None:
            audio_data = self._hold_true_peak_under(audio_data, ceiling_db)
        return audio_data

    def _hold_true_peak_under(
        self, audio_data: np.ndarray, ceiling_db: float
    ) -> np.ndarray:
        ceiling = 10 ** (ceiling_db / 20)
        peak = true_peak(audio_data)
        if peak <= ceiling:
            return audio_data
        logging.info(
            f"Export gain lowered by {20 * np.log10(peak / ceiling):.2f} dB "
            f"to keep the true peak under {ceiling_db} dBTP"
        )
        return audio_data * (ceiling / peak)

    def _measure_lufs(
        self, audio_data: np.ndarray, sample_rate: int = 44100
    ) -> float:
        try:
            import pyloudnorm

            meter = pyloudnorm.Meter(sample_rate)
            return float(meter.integrated_loudness(audio_data))
        except (ImportError, ValueError):
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                return float(20 * np.log10(rms) - 0.691)
            return float("-inf")

    def _get_subtype(self, bit_depth: int, fmt: str = "wav") -> str:
        subtype_map = {
            "wav": {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"},
            "flac": {16: "PCM_16", 24: "PCM_24"},
            "aiff": {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"},
        }
        fmt_subtypes = subtype_map.get(fmt, subtype_map["wav"])
        if bit_depth in fmt_subtypes:
            return fmt_subtypes[bit_depth]
        if fmt == "flac" and bit_depth == 32:
            logging.warning(
                "FLAC does not support 32-bit; exporting at 24-bit instead."
            )
            return "PCM_24"
        raise ValueError(
            f"No export subtype for bit depth {bit_depth!r} in format {fmt!r}"
        )

    def _run_ffmpeg(
        self, kind: str, source: Path, output: Path, options: list[str]
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"{kind} export needs ffmpeg, which is not installed or not "
                f"on the PATH"
            )
        cmd = [ffmpeg, "-nostdin", "-v", "error"]
        cmd += ["-i", f"file:{os.path.abspath(source)}"]
        cmd += [*options, "-y", f"file:{os.path.abspath(output)}"]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=EXPORT_FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"ffmpeg did not finish the {kind} file within "
                f"{EXPORT_FFMPEG_TIMEOUT_SECONDS} seconds"
            ) from e
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"ffmpeg could not write the {kind} file: {detail[-300:] or e}"
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"{kind} export could not run ffmpeg: {e}"
            ) from e

    def _convert_to_mp3(
        self,
        input_wav: Path,
        output_mp3: Path,
        bitrate: int,
        metadata: dict[str, str],
        vbr_quality: int | None = None,
    ):
        if vbr_quality is None:
            options = ["-b:a", f"{bitrate}k"]
        else:
            options = ["-q:a", str(int(vbr_quality))]
        for key, value in (metadata or {}).items():
            options.extend(["-metadata", f"{key}={value}"])
        self._run_ffmpeg("MP3", input_wav, output_mp3, options)

    def _convert_to_aac(self, input_wav: Path, output: Path, bitrate: int):
        options = ["-c:a", "aac", "-b:a", f"{bitrate}k"]
        self._run_ffmpeg("AAC", input_wav, output, options)

    def _embed_metadata(self, file_path: Path, metadata: dict[str, str]):
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".wav":
                write_riff_info(file_path, metadata)

            elif suffix == ".ogg":
                from mutagen.oggvorbis import OggVorbis

                comments = OggVorbis(str(file_path))
                for key, value in metadata.items():
                    comments[key.upper()] = value
                comments.save()

            elif suffix in (".m4a", ".mp4"):
                from mutagen.mp4 import MP4

                atoms = MP4(str(file_path))
                for key, value in metadata.items():
                    if key in MP4_TAGS:
                        atoms[MP4_TAGS[key]] = [str(value)]
                atoms.save()

            elif suffix == ".flac":
                from mutagen.flac import FLAC

                flac_file = FLAC(str(file_path))
                for key, value in metadata.items():
                    flac_file[key.upper()] = value
                flac_file.save()

            elif suffix == ".mp3":
                from mutagen.id3 import ID3, TALB, TCON, TDRC, TIT2, TPE1
                from mutagen.mp3 import MP3

                mp3_file = MP3(str(file_path), ID3=ID3)
                if mp3_file.tags is None:
                    mp3_file.add_tags()
                mp3_tags: Any = mp3_file.tags

                if "title" in metadata:
                    mp3_tags.add(TIT2(encoding=3, text=metadata["title"]))
                if "artist" in metadata:
                    mp3_tags.add(TPE1(encoding=3, text=metadata["artist"]))
                if "album" in metadata:
                    mp3_tags.add(TALB(encoding=3, text=metadata["album"]))
                if "date" in metadata:
                    mp3_tags.add(TDRC(encoding=3, text=metadata["date"]))
                if "genre" in metadata:
                    mp3_tags.add(TCON(encoding=3, text=metadata["genre"]))

                mp3_file.save()

            elif suffix in (".aiff", ".aif"):
                from mutagen.aiff import AIFF
                from mutagen.id3 import TALB, TCON, TDRC, TIT2, TPE1

                aiff_file = AIFF(str(file_path))
                if aiff_file.tags is None:
                    aiff_file.add_tags()
                aiff_tags: Any = aiff_file.tags

                if "title" in metadata:
                    aiff_tags.add(TIT2(encoding=3, text=metadata["title"]))
                if "artist" in metadata:
                    aiff_tags.add(TPE1(encoding=3, text=metadata["artist"]))
                if "album" in metadata:
                    aiff_tags.add(TALB(encoding=3, text=metadata["album"]))
                if "date" in metadata:
                    aiff_tags.add(TDRC(encoding=3, text=metadata["date"]))
                if "genre" in metadata:
                    aiff_tags.add(TCON(encoding=3, text=metadata["genre"]))

                aiff_file.save()

        except Exception as e:
            logging.debug(f"Metadata embedding not available: {e}")

    def batch_export(
        self,
        source_files: list[Path],
        preset: str = "CD Quality",
        output_dir: Path | None = None,
        naming_pattern: str = "{name}_{timestamp}",
        metadata_template: dict[str, str] | None = None,
    ) -> int:
        count = 0

        if output_dir is not None and not is_safe_path(output_dir):
            logging.error(f"Unsafe export directory: {output_dir!r}")
            return 0

        if output_dir is None:
            output_dir = self.exports_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for source in source_files:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = source.stem

                output_name = naming_pattern.format(
                    name=name, timestamp=timestamp
                )

                preset_config = self.export_presets.get(
                    preset, self.export_presets["CD Quality"]
                )
                ext = preset_config.get("format", "wav")
                output_path = output_dir / f"{output_name}.{ext}"

                metadata = metadata_template.copy() if metadata_template else {}
                metadata["title"] = metadata.get("title", name)

                if self.add_to_queue(source, output_path, preset, metadata):
                    count += 1

                self._copy_chord_analysis(source, output_dir, name)

            except Exception as e:
                logging.error(f"Failed to queue {source}: {e}")

        return count

    def _copy_chord_analysis(self, source: Path, output_dir: Path, name: str):
        try:
            chord_file = source.parent / f"{name}_Chords.txt"
            if chord_file.exists():
                dest = output_dir / chord_file.name
                with logged_file_op("Copy chord analysis", dest):
                    shutil.copy2(chord_file, dest)
                logging.info(f"Exported chord analysis: {dest}")
                return

            data_dir = self.base_dir / "saved_data"
            if data_dir.exists():
                chord_file = data_dir / f"{name}_Chords.txt"
                if chord_file.exists():
                    dest = output_dir / chord_file.name
                    with logged_file_op("Copy chord analysis", dest):
                        shutil.copy2(chord_file, dest)
                    logging.info(f"Exported chord analysis: {dest}")
        except Exception as e:
            logging.debug(f"No chord analysis to export for {name}: {e}")

    def get_queue_status(self) -> dict[str, Any]:
        with self._queue_lock:
            queue_length = len(self.export_queue)
            queued_jobs = [
                {
                    "source": job["source"],
                    "output": job["output"],
                    "preset": job["preset"],
                    "status": job["status"],
                }
                for job in self.export_queue
            ]

        with self._export_lock:
            is_exporting = self.is_exporting
            current_export = self.current_export

        return {
            "queue_length": queue_length,
            "is_exporting": is_exporting,
            "current_export": current_export,
            "queued_jobs": queued_jobs,
        }

    def clear_queue(self):
        with self._queue_lock:
            self.export_queue.clear()

    def save_preset(self, name: str, config: dict[str, Any]):
        validate_export_preset(config)
        self.export_presets[name] = config
        self._save_presets()

    def delete_preset(self, name: str) -> bool:
        if name in self.export_presets:
            del self.export_presets[name]
            self._save_presets()
            return True
        return False

    def get_preset_names(self) -> list[str]:
        return list(self.export_presets.keys())
