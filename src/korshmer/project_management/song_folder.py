import filecmp
import os
import shutil
from pathlib import Path

from korshmer.project_management.file_io_log import logged_file_op
from korshmer.ui.viewmodels.arranger_vm import CLIPS_FOLDER


def _inside(path, folder):
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
    except ValueError:
        return False
    return True


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
    if path and _inside(path, song_folder):
        return os.path.relpath(path, song_folder)
    return path


def _found(path, song_folder):
    if path and not os.path.isabs(path):
        return os.path.normpath(os.path.join(str(song_folder), path))
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
