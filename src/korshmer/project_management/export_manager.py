import json
import logging
import os
import shutil
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


def validate_export_preset(config: dict[str, Any]) -> None:
    validate_preset_normalize_target(config)
    validate_preset_bit_depth(config)


class ExportManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.exports_dir = self.base_dir / "saved_data" / "exports"
        self.presets_file = self.base_dir / "saved_data" / "export_presets.json"

        self._ensure_directories()
        self.export_presets = self._load_presets()
        self.export_queue = deque()
        self.is_exporting = False
        self.current_export: dict[str, Any] | None = None
        self._queue_lock = threading.Lock()
        self._export_lock = threading.Lock()

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
                "metadata": True,
            },
            "YouTube": {
                "format": "wav",
                "sample_rate": 48000,
                "bit_depth": 24,
                "normalize": "lufs",
                "target_lufs": -13,
                "metadata": True,
            },
            "CD Quality": {
                "format": "wav",
                "sample_rate": 44100,
                "bit_depth": 16,
                "normalize": "peak",
                "target_peak": -0.1,
                "dither": True,
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
                "metadata": True,
            },
            "MP3 Medium": {
                "format": "mp3",
                "bitrate": 192,
                "sample_rate": 44100,
                "normalize": "rms",
                "target_rms": -18,
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
            target=self._process_queue_thread, daemon=True
        )
        thread.start()

    def _process_queue_thread(self):
        while True:
            with self._queue_lock:
                if not self.export_queue:
                    break
                job = self.export_queue.popleft()

            self.current_export = job
            job["status"] = "processing"

            try:
                success = self._export_file(
                    Path(job["source"]),
                    Path(job["output"]),
                    job["preset"],
                    job["metadata"],
                )

                job["status"] = "completed" if success else "failed"

                if job["callback"]:
                    job["callback"](job, success)

            except Exception as e:
                logging.error(f"Export failed: {e}")
                job["status"] = "failed"
                if job["callback"]:
                    job["callback"](job, False)

        with self._export_lock:
            self.is_exporting = False
            self.current_export = None

    def _export_file(
        self, source: Path, output: Path, preset: str, metadata: dict[str, str]
    ) -> bool:
        try:
            import soundfile as sf

            preset_config = self.export_presets.get(
                preset, self.export_presets["CD Quality"]
            )

            with logged_file_op("Read export source", source):
                audio_data, source_sr = sf.read(str(source))

            target_sr = preset_config.get("sample_rate", source_sr)
            if target_sr != source_sr:
                audio_data = self._resample(audio_data, source_sr, target_sr)

            normalize_mode = preset_config.get("normalize", "none")
            if normalize_mode != "none":
                try:
                    audio_data = self._normalize_audio(
                        audio_data, normalize_mode, preset_config
                    )
                except ValueError as e:
                    logging.error(
                        f"Export file failed: invalid normalize target in "
                        f"export preset {preset!r}: {e}"
                    )
                    return False

            output_format = preset_config.get("format", "wav")

            bit_depth = preset_config.get("bit_depth", 16)
            if output_format in ("wav", "flac", "aiff"):
                try:
                    bit_depth = validate_bit_depth(bit_depth)
                except ValueError as e:
                    logging.error(
                        f"Export file failed: invalid bit depth in export "
                        f"preset {preset!r}: {e}"
                    )
                    return False
            if bit_depth == 16 and preset_config.get("dither", False):
                audio_data = self._apply_dither(audio_data)

            with logged_file_op("Export audio", output):
                self._write_output(
                    output,
                    audio_data,
                    target_sr,
                    output_format,
                    bit_depth,
                    preset_config,
                    metadata,
                )
                if preset_config.get("metadata", True) and metadata:
                    self._embed_metadata(output, metadata)

            if not output.exists():
                logger.warning("Export audio left no file: %s", output)

            return True

        except Exception as e:
            logging.error(f"Export file failed: {e}")
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
    ) -> None:
        import soundfile as sf

        if output_format in ["wav", "flac", "aiff"]:
            subtype = self._get_subtype(bit_depth, output_format)
            fmt_map = {"flac": "FLAC", "aiff": "AIFF"}
            write_kwargs = {"subtype": subtype}
            if output_format in fmt_map:
                write_kwargs["format"] = fmt_map[output_format]
            sf.write(str(output), audio_data, target_sr, **write_kwargs)

        elif output_format == "ogg":
            sf.write(
                str(output),
                audio_data,
                target_sr,
                format="OGG",
                subtype="VORBIS",
            )

        elif output_format == "mp3":
            temp_wav = output.with_suffix(".temp.wav")
            try:
                sf.write(str(temp_wav), audio_data, target_sr, subtype="PCM_16")

                bitrate = preset_config.get("bitrate", 320)
                self._convert_to_mp3(temp_wav, output, bitrate, metadata)
            finally:
                if temp_wav.exists():
                    temp_wav.unlink()

    def _resample(
        self, audio_data: np.ndarray, source_sr: int, target_sr: int
    ) -> np.ndarray:
        try:
            import librosa

            return librosa.resample(
                audio_data.T, orig_sr=source_sr, target_sr=target_sr
            ).T
        except (ImportError, ValueError, TypeError):
            return audio_data

    def _normalize_audio(
        self, audio_data: np.ndarray, mode: str, config: dict[str, Any]
    ) -> np.ndarray:
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

        elif mode == "lufs":
            target_lufs = config.get("target_lufs", -14)
            current_lufs = self._measure_lufs(audio_data)
            adjustment = target_lufs - current_lufs
            audio_data = audio_data * (10 ** (adjustment / 20))

        audio_data = np.clip(audio_data, -1.0, 1.0)
        return audio_data

    def _measure_lufs(self, audio_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            return 20 * np.log10(rms) - 0.691
        return -100

    def _apply_dither(self, audio_data: np.ndarray) -> np.ndarray:
        dither_amount = 1.0 / (2**15)
        dither = np.random.triangular(
            -dither_amount, 0, dither_amount, audio_data.shape
        )
        return audio_data + dither

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

    def _convert_to_mp3(
        self,
        input_wav: Path,
        output_mp3: Path,
        bitrate: int,
        metadata: dict[str, str],
    ):
        try:
            cmd = ["ffmpeg", "-i", str(input_wav), "-b:a", f"{bitrate}k"]

            if metadata:
                for key, value in metadata.items():
                    cmd.extend(["-metadata", f"{key}={value}"])

            cmd.extend(["-y", str(output_mp3)])

            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            logging.warning(f"MP3 conversion requires ffmpeg: {e}")

    def _embed_metadata(self, file_path: Path, metadata: dict[str, str]):
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".flac":
                from mutagen.flac import FLAC

                audio = FLAC(str(file_path))
                for key, value in metadata.items():
                    audio[key.upper()] = value
                audio.save()

            elif suffix == ".mp3":
                from mutagen.id3 import ID3, TALB, TCON, TDRC, TIT2, TPE1
                from mutagen.mp3 import MP3

                audio = MP3(str(file_path), ID3=ID3)
                if audio.tags is None:
                    audio.add_tags()

                if "title" in metadata:
                    audio.tags.add(TIT2(encoding=3, text=metadata["title"]))
                if "artist" in metadata:
                    audio.tags.add(TPE1(encoding=3, text=metadata["artist"]))
                if "album" in metadata:
                    audio.tags.add(TALB(encoding=3, text=metadata["album"]))
                if "date" in metadata:
                    audio.tags.add(TDRC(encoding=3, text=metadata["date"]))
                if "genre" in metadata:
                    audio.tags.add(TCON(encoding=3, text=metadata["genre"]))

                audio.save()

            elif suffix in (".aiff", ".aif"):
                from mutagen.aiff import AIFF
                from mutagen.id3 import TALB, TCON, TDRC, TIT2, TPE1

                audio = AIFF(str(file_path))
                if audio.tags is None:
                    audio.add_tags()

                if "title" in metadata:
                    audio.tags.add(TIT2(encoding=3, text=metadata["title"]))
                if "artist" in metadata:
                    audio.tags.add(TPE1(encoding=3, text=metadata["artist"]))
                if "album" in metadata:
                    audio.tags.add(TALB(encoding=3, text=metadata["album"]))
                if "date" in metadata:
                    audio.tags.add(TDRC(encoding=3, text=metadata["date"]))
                if "genre" in metadata:
                    audio.tags.add(TCON(encoding=3, text=metadata["genre"]))

                audio.save()

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
