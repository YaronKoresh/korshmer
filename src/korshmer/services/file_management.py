import json
import logging
import os
from pathlib import Path

from korshmer.constants import INPUT_LIST_FILE
from korshmer.exceptions import InvalidFileError
from korshmer.utils.filesystem import MIDI_EXTENSIONS, sanitize_path


class FileManagementService:
    VALID_AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif")
    VALID_MIDI_EXTENSIONS = MIDI_EXTENSIONS
    VALID_EXTENSIONS = VALID_AUDIO_EXTENSIONS + VALID_MIDI_EXTENSIONS

    @staticmethod
    def format_size(size):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    @staticmethod
    def get_path_size(path):
        total_size = 0
        try:
            if os.path.isfile(path):
                return os.path.getsize(path)
            elif os.path.isdir(path):
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
            return total_size
        except OSError as e:
            logging.debug(f"Could not get path size: {e}")
            return 0

    @staticmethod
    def save_file_list(paths):
        try:
            tmp = Path(str(INPUT_LIST_FILE) + ".tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(paths, f)
                os.replace(tmp, INPUT_LIST_FILE)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        except OSError as e:
            logging.warning(f"Failed to save file list: {e}")

    @staticmethod
    def load_file_list():
        if os.path.exists(INPUT_LIST_FILE):
            try:
                with open(INPUT_LIST_FILE, encoding="utf-8") as f:
                    paths = json.load(f)
                if not isinstance(paths, list):
                    return []
                result = []
                for p in paths:
                    try:
                        sanitize_path(p)
                    except (InvalidFileError, TypeError) as e:
                        logging.warning(f"Skipping unsafe saved path: {e}")
                        continue
                    if os.path.exists(p):
                        result.append(p)
                return result
            except (OSError, ValueError) as e:
                logging.debug(f"Could not load file list: {e}")
        return []

    @staticmethod
    def resolve_path(path):
        sanitize_path(path)
        return str(Path(path).resolve())

    @staticmethod
    def is_valid_audio(path):
        return path.lower().endswith(
            FileManagementService.VALID_AUDIO_EXTENSIONS
        )

    @staticmethod
    def is_valid_midi(path):
        return path.lower().endswith(
            FileManagementService.VALID_MIDI_EXTENSIONS
        )

    @staticmethod
    def is_valid_file(path):
        return path.lower().endswith(FileManagementService.VALID_EXTENSIONS)

    @staticmethod
    def list_output_files(directory, skip_names=None):
        if skip_names is None:
            skip_names = {"logs", "temp_preview.wav"}
        try:
            items = os.listdir(directory)
            items.sort(
                key=lambda x: os.path.getmtime(os.path.join(directory, x)),
                reverse=True,
            )
            result = []
            for name in items:
                full_path = str(Path(os.path.join(directory, name)).resolve())
                if os.path.isdir(full_path) and name in skip_names:
                    continue
                if name.startswith(".") or name.startswith("_"):
                    continue
                if name in skip_names:
                    continue
                if os.path.isdir(full_path) or name.endswith(
                    (*MIDI_EXTENSIONS, ".wav", ".mp3", ".png", ".jpg", ".txt")
                ):
                    result.append((name, full_path))
            return result
        except (RuntimeError, AttributeError, OSError) as e:
            logging.debug(f"Could not list output files: {e}")
            return []

    @staticmethod
    def get_file_stats(path):
        try:
            stats = os.stat(path)
            from datetime import datetime

            return {
                "size": FileManagementService.get_path_size(path),
                "size_formatted": FileManagementService.format_size(
                    FileManagementService.get_path_size(path)
                ),
                "created": datetime.fromtimestamp(stats.st_ctime).strftime(
                    "%d/%m/%Y %H:%M"
                ),
                "modified": datetime.fromtimestamp(stats.st_mtime).strftime(
                    "%d/%m/%Y %H:%M"
                ),
                "is_dir": os.path.isdir(path),
                "basename": os.path.basename(path),
            }
        except OSError:
            return None

    @staticmethod
    def find_source_for_comparison(midi_path, source_paths):
        from korshmer.audio_processing.stem_session import StemSession

        midi_path = os.path.abspath(midi_path)
        session = StemSession.find(os.path.dirname(midi_path))
        cut_source = session.cut_source_path() if session else None
        if cut_source and midi_path in {
            os.path.abspath(path) for path in session.midi_files
        }:
            return cut_source
        base_name = os.path.basename(midi_path)
        for path in source_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in base_name:
                return path
        return None
