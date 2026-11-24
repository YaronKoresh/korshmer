import base64
import binascii
import filecmp
import logging
import os
import shutil
from pathlib import Path

from korshmer.event_bus import Event, publish
from korshmer.project_management.file_io_log import logged_file_op
from korshmer.ui.viewmodels.arranger_vm import CLIPS_FOLDER

MIDI_FOLDER = "MIDI"
CONVERSION_FOLDER = "Conversion"
SHARED_MIDI_FOLDER = "Processed_MIDI"
REFUSED_FOLDER = "_not_opened_outside_song"
_open_song = {"folder": None}
_logger = logging.getLogger(__name__)


def set_open_song_folder(folder):
    _open_song["folder"] = str(folder) if folder else None


def open_song_folder():
    return _open_song["folder"]


def midi_folder(shared_data_folder):
    home = open_song_folder()
    folder = (
        os.path.join(home, MIDI_FOLDER)
        if home
        else os.path.join(str(shared_data_folder), SHARED_MIDI_FOLDER)
    )
    os.makedirs(folder, exist_ok=True)
    return folder


def conversion_kept_in_song(session_folder, song_folder):
    session_folder = str(session_folder or "")
    if not session_folder or not os.path.isdir(session_folder):
        return None
    if _inside(session_folder, song_folder) or _inside(
        song_folder, session_folder
    ):
        return None
    target = Path(song_folder) / CONVERSION_FOLDER
    with logged_file_op("Copy conversion into song", target):
        shutil.copytree(
            session_folder,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.kproj", "Versions", "Backups"),
        )
    return str(target)


def _inside(path, folder):
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
    except (ValueError, RuntimeError, OSError):
        return False
    return True


def _refused(stored, song_folder):
    token = base64.urlsafe_b64encode(stored.encode("utf-8")).decode("ascii")
    return os.path.join(
        str(song_folder), REFUSED_FOLDER, token + Path(stored).suffix
    )


def _refused_original(path):
    parts = Path(path).parts
    if len(parts) < 2 or parts[-2] != REFUSED_FOLDER:
        return None
    try:
        token = Path(parts[-1]).stem.encode("ascii")
        return base64.urlsafe_b64decode(token).decode("utf-8")
    except (binascii.Error, ValueError):
        return None


def _free_copy_path(source, folder):
    wanted = Path(folder) / Path(source).name
    count = 1
    while wanted.exists() and not filecmp.cmp(source, wanted, shallow=False):
        count += 1
        wanted = wanted.with_name(
            f"{Path(source).stem}_{count}{Path(source).suffix}"
        )
    return wanted


def kept_in_song(path, song_folder):
    path = str(path)
    if not os.path.isfile(path) or _inside(path, song_folder):
        return path
    clips = Path(song_folder) / CLIPS_FOLDER
    clips.mkdir(parents=True, exist_ok=True)
    target = _free_copy_path(path, clips)
    if not target.exists():
        with logged_file_op("Copy clip into song", target):
            shutil.copy2(path, target)
    return str(target)


def _stored(path, song_folder):
    original = _refused_original(path) if path else None
    if original is not None:
        return original
    if path and _inside(path, song_folder):
        return os.path.relpath(path, song_folder)
    return path


def _found(path, song_folder):
    if path and not os.path.isabs(path):
        found = os.path.normpath(os.path.join(str(song_folder), path))
        if _inside(found, song_folder):
            return found
        _logger.warning("Clip outside the song folder not opened: %s", path)
        publish(
            Event.STATUS_UPDATE,
            text=f"Not opened, outside the song folder: {path}",
        )
        return _refused(path, song_folder)
    return path


def _with_clip_paths(editor, song_folder, convert):
    editor = dict(editor)
    clips = []
    for clip in editor.get("clips") or []:
        clip = dict(clip)
        clip["path"] = convert(str(clip.get("path", "")), song_folder)
        source = clip.get("source")
        if source:
            clip["source"] = [convert(str(source[0]), song_folder), *source[1:]]
        base = clip.get("tempo_base")
        if base:
            clip["tempo_base"] = [convert(str(base[0]), song_folder), *base[1:]]
        clips.append(clip)
    editor["clips"] = clips
    return editor


def stored_editor(editor, song_folder):
    return _with_clip_paths(editor, song_folder, _stored)


def opened_editor(editor, song_folder):
    return _with_clip_paths(editor, song_folder, _found)


def own_song_path(song_path, saved_data_dir, suffix):
    song_path = Path(song_path).with_suffix(suffix)
    folder = song_path.parent
    others = [
        other
        for other in folder.glob(f"*{suffix}")
        if other.name != song_path.name
    ]
    shared = folder.resolve() == Path(saved_data_dir).resolve() or bool(others)
    if shared:
        return folder / song_path.stem / song_path.name
    return song_path
