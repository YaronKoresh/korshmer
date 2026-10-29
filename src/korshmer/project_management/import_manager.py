import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from korshmer.audio.loader import check_audio_file_type
from korshmer.exceptions import AudioFormatError
from korshmer.project_management.file_io_log import logged_file_op, logger
from korshmer.project_management.validators.project_validator import (
    INVALID_NAME_MESSAGE,
    is_valid_new_name,
)
from korshmer.utils.filesystem import is_safe_path

MIDI_SUFFIXES = frozenset({".mid", ".midi"})
MIDI_SIGNATURE = b"MThd"
DEFAULT_IMPORT_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aac",
    ".m4a",
    ".aiff",
    ".aif",
    ".mid",
    ".midi",
)


def _check_midi_file_type(path) -> bool:
    with open(path, "rb") as handle:
        header = handle.read(12)
    is_midi = header[0:4] == MIDI_SIGNATURE or (
        header[0:4] == b"RIFF" and header[8:12] == b"RMID"
    )
    if not is_midi:
        raise AudioFormatError(
            f"File content is not a recognized MIDI format: {path}"
        )
    return True


class ImportManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.imports_dir = self.base_dir / "saved_data" / "imports"

        self._ensure_directories()

    def _ensure_directories(self):
        self.imports_dir.mkdir(parents=True, exist_ok=True)

    def import_audio_files(
        self,
        file_paths: list[Path],
        auto_organize: bool = True,
        categorize: bool = True,
    ) -> list[Path]:
        imported_files = []

        for file_path in file_paths:
            try:
                if not is_safe_path(file_path):
                    logging.warning(
                        f"Skipping unsafe import path: {file_path!r}"
                    )
                    continue

                file_path = Path(file_path)

                if not file_path.exists():
                    logging.warning(f"File does not exist: {file_path}")
                    continue

                is_midi = file_path.suffix.lower() in MIDI_SUFFIXES
                try:
                    if is_midi:
                        _check_midi_file_type(str(file_path))
                    else:
                        check_audio_file_type(str(file_path))
                except AudioFormatError as e:
                    kind = "MIDI" if is_midi else "audio"
                    logging.warning(
                        f"Skipping file that is not valid {kind}: "
                        f"{file_path} ({e})"
                    )
                    continue

                if auto_organize:
                    target_dir = self._get_organized_path(file_path, categorize)
                else:
                    target_dir = self.imports_dir

                target_dir.mkdir(parents=True, exist_ok=True)

                target_path = target_dir / file_path.name
                counter = 1
                while target_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                import shutil

                with logged_file_op("Import file", target_path):
                    shutil.copy2(file_path, target_path)
                imported_files.append(target_path)

                logging.info(f"Imported: {file_path} -> {target_path}")

            except Exception as e:
                logging.error(f"Failed to import {file_path}: {e}")

        return imported_files

    def _get_organized_path(self, file_path: Path, categorize: bool) -> Path:
        if not categorize:
            return self.imports_dir

        suffix = file_path.suffix.lower()

        if suffix in {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".aac",
            ".m4a",
            ".aiff",
            ".aif",
        }:
            category = self._categorize_audio(file_path)
            return self.imports_dir / "audio" / category

        elif suffix in {".mid", ".midi"}:
            return self.imports_dir / "midi"

        else:
            return self.imports_dir / "other"

    def _categorize_audio(self, file_path: Path) -> str:
        name_lower = file_path.stem.lower()

        drum_keywords = [
            "drum",
            "kick",
            "snare",
            "hihat",
            "cymbal",
            "perc",
            "beat",
        ]
        bass_keywords = ["bass", "sub", "low"]
        vocal_keywords = ["vocal", "voice", "vox", "singing", "acapella"]
        fx_keywords = ["fx", "sfx", "effect", "riser", "impact", "sweep"]
        loop_keywords = ["loop", "cycle"]

        for keyword in drum_keywords:
            if keyword in name_lower:
                return "drums"

        for keyword in bass_keywords:
            if keyword in name_lower:
                return "bass"

        for keyword in vocal_keywords:
            if keyword in name_lower:
                return "vocals"

        for keyword in fx_keywords:
            if keyword in name_lower:
                return "fx"

        for keyword in loop_keywords:
            if keyword in name_lower:
                return "loops"

        return "samples"

    def extract_audio_from_video(
        self, video_path: Path, output_format: str = "wav"
    ) -> Path | None:
        try:
            if not is_safe_path(video_path):
                logging.error(f"Unsafe video path: {video_path!r}")
                return None

            video_path = Path(video_path)

            if not video_path.exists():
                logging.error(f"Video file does not exist: {video_path}")
                return None

            output_dir = self.imports_dir / "extracted_audio"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{video_path.stem}_{timestamp}.{output_format}"
            output_path = output_dir / output_name

            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le" if output_format == "wav" else "libmp3lame",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-y",
                output_path,
            ]

            logger.debug("Extract audio started: %s", output_path)
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.exists():
                logger.info(
                    "Extract audio done: %s (%d bytes)",
                    output_path,
                    output_path.stat().st_size,
                )
                logging.info(
                    f"Extracted audio from {video_path} to {output_path}"
                )
                return output_path
            else:
                logger.error(
                    "Extract audio failed: %s (ffmpeg exit code %s)",
                    output_path,
                    result.returncode,
                )
                logging.error(f"Failed to extract audio: {result.stderr}")
                return None

        except FileNotFoundError:
            logging.error(
                "ffmpeg not found. Please install ffmpeg to extract audio from videos."
            )
            return None

        except Exception as e:
            logging.error(f"Failed to extract audio from video: {e}")
            return None

    def batch_import_from_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_extensions: list[str] | None = None,
        auto_organize: bool = True,
    ) -> list[Path]:
        if not is_safe_path(directory):
            logging.error(f"Unsafe import directory: {directory!r}")
            return []

        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            logging.error(f"Directory does not exist: {directory}")
            return []

        if file_extensions is None:
            file_extensions = DEFAULT_IMPORT_EXTENSIONS

        wanted = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in file_extensions
        }
        candidates = directory.rglob("*") if recursive else directory.glob("*")
        file_paths = sorted(
            path
            for path in candidates
            if path.suffix.lower() in wanted and path.is_file()
        )

        return self.import_audio_files(file_paths, auto_organize=auto_organize)

    def analyze_sample_pack(self, pack_path: Path) -> dict[str, Any]:
        pack_path = Path(pack_path)

        if not pack_path.exists():
            return {}

        analysis = {
            "name": pack_path.name,
            "path": str(pack_path),
            "total_files": 0,
            "audio_files": 0,
            "midi_files": 0,
            "total_size": 0,
            "categories": {},
            "formats": {},
        }

        for file_path in pack_path.rglob("*"):
            if file_path.is_file():
                analysis["total_files"] += 1
                analysis["total_size"] += file_path.stat().st_size

                suffix = file_path.suffix.lower()
                analysis["formats"][suffix] = (
                    analysis["formats"].get(suffix, 0) + 1
                )

                if suffix in {
                    ".wav",
                    ".mp3",
                    ".flac",
                    ".ogg",
                    ".aac",
                    ".aiff",
                    ".aif",
                }:
                    analysis["audio_files"] += 1
                    category = self._categorize_audio(file_path)
                    analysis["categories"][category] = (
                        analysis["categories"].get(category, 0) + 1
                    )

                elif suffix in {".mid", ".midi"}:
                    analysis["midi_files"] += 1

        return analysis

    def create_import_session(self, session_name: str) -> Path:
        if not is_valid_new_name(session_name):
            raise ValueError(INVALID_NAME_MESSAGE)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.imports_dir / f"{session_name}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def get_import_stats(self) -> dict[str, Any]:
        total_files = 0
        total_size = 0
        file_types = {}

        for file_path in self.imports_dir.rglob("*"):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size

                suffix = file_path.suffix.lower()
                file_types[suffix] = file_types.get(suffix, 0) + 1

        return {
            "total_files": total_files,
            "total_size": total_size,
            "file_types": file_types,
        }
