from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from korshmer.constants import MAX_PROJECT_MANIFEST_BYTES
from korshmer.project_management.file_io_log import logged_file_op, logger

PARTIAL_DIR_NAME = ".partial"


def cleanup_old_backups(
    backups_dir: Path, project_name: str, max_backups: int
) -> None:
    backups = sorted(
        backups_dir.glob(f"{glob.escape(project_name)}_backup_*.kproj"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    for old_backup in backups[max_backups:]:
        try:
            old_backup.unlink()
            logger.info("Delete old backup done: %s", old_backup)
        except Exception as e:
            logging.warning(f"Failed to delete old backup {old_backup}: {e}")


def create_backup_file(
    current_project_path: Path,
    backups_dir: Path,
    max_backups: int,
    save_compressed: Callable[[Path], Any],
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = current_project_path.stem
    backup_name = f"{project_name}_backup_{timestamp}.kproj"
    backup_path = backups_dir / backup_name

    partial_dir = backups_dir / PARTIAL_DIR_NAME
    partial_path = partial_dir / backup_name
    try:
        with logged_file_op("Create backup", backup_path):
            partial_dir.mkdir(parents=True, exist_ok=True)
            if current_project_path.suffix == ".kproj":
                shutil.copy2(current_project_path, partial_path)
            else:
                save_compressed(partial_path)
            os.replace(partial_path, backup_path)
    finally:
        partial_path.unlink(missing_ok=True)

    cleanup_old_backups(backups_dir, project_name, max_backups)


def snapshot_path_for(current_project_path: Path, snapshots_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = current_project_path.stem
    return snapshots_dir / f"{project_name}_snapshot_{timestamp}.kproj"


def build_snapshot_data(
    current_project_data: dict[str, Any],
    current_project_path: Path,
    description: str,
) -> dict[str, Any]:
    snapshot_data = current_project_data.copy()
    snapshot_data["snapshot_info"] = {
        "timestamp": datetime.now().isoformat(),
        "description": description,
        "parent_project": str(current_project_path),
    }
    return snapshot_data


VERSION_TIME_FORMAT = "%d %b %Y, %H:%M"
UNNAMED_VERSION = "Version saved"


def _snapshot_info(snapshot_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(snapshot_path, "r") as archive:
            size = archive.getinfo("manifest.json").file_size
            if size > MAX_PROJECT_MANIFEST_BYTES:
                return {}
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return {}
    info = manifest.get("snapshot_info") if isinstance(manifest, dict) else None
    return info if isinstance(info, dict) else {}


def snapshot_label(snapshot_path: Path) -> str:
    snapshot_path = Path(snapshot_path)
    info = _snapshot_info(snapshot_path)
    try:
        saved = datetime.fromisoformat(str(info.get("timestamp")))
    except ValueError:
        saved = datetime.fromtimestamp(snapshot_path.stat().st_mtime)
    when = saved.strftime(VERSION_TIME_FORMAT)
    description = " ".join(str(info.get("description") or "").split())
    return f"{description or UNNAMED_VERSION} — {when}"


def snapshot_choices(snapshot_paths) -> dict[str, Path]:
    choices: dict[str, Path] = {}
    for path in snapshot_paths:
        label = snapshot_label(path)
        count = 2
        unique = label
        while unique in choices:
            unique = f"{label} ({count})"
            count += 1
        choices[unique] = Path(path)
    return choices
