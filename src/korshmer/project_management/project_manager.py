import glob
import hashlib
import json
import logging
import shutil
import threading
import time
import zipfile
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from korshmer import __version__
from korshmer.project_management.file_io_log import logged_file_op
from korshmer.project_management.project_backup import (
    build_snapshot_data,
    create_backup_file,
    snapshot_path_for,
)
from korshmer.project_management.project_loader import (
    load_compressed,
    load_uncompressed,
)
from korshmer.project_management.project_migration import (
    migrate_project_data,
    validate_project_effects_chain,
)
from korshmer.project_management.project_recovery import (
    clear_recovery_file,
    load_recovery_file,
    save_recovery_file,
)
from korshmer.project_management.project_saver import (
    save_compressed,
    save_uncompressed,
)
from korshmer.project_management.validators.project_validator import (
    is_safe_name,
    is_valid_new_name,
    resolve_inside,
)
from korshmer.utils.filesystem import is_safe_path

F = TypeVar("F", bound=Callable[..., Any])


class ReadWriteLock:
    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int | None = None
        self._write_depth = 0

    def acquire_read(self) -> None:
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                return
            while self._writer is not None:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._write_depth += 1
                return
            while self._writer is not None or self._readers > 0:
                self._cond.wait()
            self._writer = me
            self._write_depth = 1

    def release_write(self) -> None:
        with self._cond:
            self._write_depth -= 1
            if self._write_depth == 0:
                self._writer = None
                self._cond.notify_all()


def _read_locked(method: F) -> F:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._state_lock.acquire_read()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._state_lock.release_read()

    return wrapper


def _write_locked(method: F) -> F:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._state_lock.acquire_write()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._state_lock.release_write()

    return wrapper


class ProjectManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.saved_data_dir = self.base_dir / "saved_data"
        self.projects_dir = self.base_dir / "saved_data" / "projects"
        self.backups_dir = self.base_dir / "saved_data" / "backups"
        self.templates_dir = self.base_dir / "saved_data" / "templates"
        self.snapshots_dir = self.base_dir / "saved_data" / "snapshots"

        self._state_lock = ReadWriteLock()
        self._ensure_directories()
        self.current_project_path: Path | None = None
        self.current_project_data: dict[str, Any] = {}
        self.auto_backup_interval = 300
        self.last_backup_time = 0
        self.last_load_backup: Path | None = None
        self.backup_enabled = True
        self.max_backups = 10
        self._last_saved_hash: str | None = None
        self._incremental_save_enabled = True

    def _ensure_directories(self):
        for directory in [
            self.projects_dir,
            self.backups_dir,
            self.templates_dir,
            self.snapshots_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def is_path_within_saved_data(self, path) -> bool:
        return resolve_inside(self.saved_data_dir, path) is not None

    def _project_dir_for(self, name) -> Path | None:
        if not is_safe_name(name):
            logging.error(f"Rejected unsafe project name: {name!r}")
            return None
        project_dir = resolve_inside(
            self.projects_dir, self.projects_dir / name
        )
        if project_dir is None:
            logging.error(f"Rejected unsafe project name: {name!r}")
        return project_dir

    def _template_path_for(self, template_name) -> Path | None:
        if not is_safe_name(template_name):
            logging.error(f"Rejected unsafe template name: {template_name!r}")
            return None
        template_path = resolve_inside(
            self.templates_dir, self.templates_dir / f"{template_name}.kproj"
        )
        if template_path is None:
            logging.error(f"Rejected unsafe template name: {template_name!r}")
        return template_path

    @_write_locked
    def create_project(
        self, name: str, template: str | None = None
    ) -> dict[str, Any] | None:
        if not is_valid_new_name(name):
            logging.error(f"Rejected invalid project name: {name!r}")
            return None
        project_dir = self._project_dir_for(name)
        if project_dir is None:
            return None

        project_data = {
            "name": name,
            "version": __version__,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "metadata": {
                "author": "",
                "description": "",
                "tags": [],
                "color": "#00FFFF",
                "bpm": 120,
                "time_signature": "4/4",
                "key": "C",
            },
            "tracks": [],
            "markers": [],
            "assets": [],
            "settings": {},
        }

        if template and template != "blank":
            template_data = self.load_template(template)
            if template_data:
                project_data.update(template_data)
                project_data["name"] = name
                project_data["created"] = datetime.now().isoformat()

        self.current_project_data = project_data

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create project directory: {e}")
            return None
        return project_data

    @_write_locked
    def set_metadata(self, metadata: dict[str, Any]):
        if "metadata" not in self.current_project_data:
            self.current_project_data["metadata"] = {}
        self.current_project_data["metadata"].update(metadata)

    @_write_locked
    def save_project(
        self, project_path: Path | None = None, compress: bool = True
    ) -> bool:
        try:
            if project_path is None:
                project_path = self.current_project_path

            if project_path is None:
                logging.error("No project path specified")
                return False

            if not is_safe_path(project_path):
                logging.error(f"Unsafe project path: {project_path!r}")
                return False

            project_path = Path(project_path)
            final_path = project_path.with_suffix(
                ".kproj" if compress else ".json"
            )
            if not self.is_path_within_saved_data(final_path):
                logging.error(
                    f"Refusing to save project outside "
                    f"{self.saved_data_dir}: {final_path}"
                )
                return False

            self.current_project_data["modified"] = datetime.now().isoformat()

            if compress:
                return self._save_compressed(project_path)
            else:
                return self._save_uncompressed(project_path)

        except Exception as e:
            logging.error(f"Failed to save project: {e}")
            return False

    def _save_compressed(self, project_path: Path) -> bool:
        self.current_project_path = save_compressed(
            self.current_project_data,
            project_path,
            Path(self.base_dir) / "temp",
        )
        return True

    def _save_uncompressed(self, project_path: Path) -> bool:
        self.current_project_path = save_uncompressed(
            self.current_project_data, project_path
        )
        return True

    @_write_locked
    def load_project(self, project_path: Path) -> dict[str, Any] | None:
        try:
            if not is_safe_path(project_path):
                logging.error(f"Unsafe project path: {project_path!r}")
                return None

            project_path = Path(project_path)
            previous = (self.current_project_path, self.current_project_data)

            if project_path.suffix in (".kproj", ".json"):
                if project_path.suffix == ".kproj":
                    project_data = self._load_compressed(project_path)
                else:
                    project_data = self._load_uncompressed(project_path)
                try:
                    validate_project_effects_chain(project_data)
                except ValueError as e:
                    self.current_project_path, self.current_project_data = (
                        previous
                    )
                    logging.error(f"Rejected project {project_path}: {e}")
                    return None
                return project_data
            else:
                logging.error(
                    f"Unsupported project format: {project_path.suffix}"
                )
                return None

        except Exception as e:
            logging.error(f"Failed to load project: {e}")
            return None

    @_write_locked
    def load_project_with_fallback(
        self, project_path: Path
    ) -> dict[str, Any] | None:
        project_path = Path(project_path)
        self.last_load_backup = None
        data = self.load_project(project_path)
        if data is not None:
            return data

        source = (self.current_project_path, self.current_project_data)
        for backup in self.list_backups(project_path.stem):
            data = self.load_project(backup)
            if data is not None:
                logging.warning(
                    f"Loaded backup {backup.name} instead of {project_path}"
                )
                self.last_load_backup = backup
                self.current_project_path = project_path
                self.current_project_data = data
                return data
        self.current_project_path, self.current_project_data = source
        return None

    def _load_compressed(self, project_path: Path) -> dict[str, Any] | None:
        project_data = load_compressed(
            project_path,
            self.base_dir / "temp",
            self.base_dir / "saved_data" / "assets",
        )
        self.current_project_path = project_path
        self.current_project_data = project_data
        return project_data

    def _load_uncompressed(self, project_path: Path) -> dict[str, Any] | None:
        project_data = load_uncompressed(project_path)
        self.current_project_path = project_path
        self.current_project_data = project_data
        return project_data

    @_write_locked
    def create_backup(self) -> bool:
        if not self.backup_enabled or not self.current_project_path:
            return False

        current_time = time.time()
        if current_time - self.last_backup_time < self.auto_backup_interval:
            return False

        project_path = self.current_project_path

        def save_backup(backup_path: Path) -> bool:
            try:
                return self._save_compressed(backup_path)
            finally:
                self.current_project_path = project_path

        try:
            create_backup_file(
                project_path,
                self.backups_dir,
                self.max_backups,
                save_backup,
            )
            self.last_backup_time = current_time
            return True

        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return False

    @_write_locked
    def create_snapshot(self, description: str = "") -> bool:
        if not self.current_project_path:
            return False

        try:
            snapshot_path = snapshot_path_for(
                self.current_project_path, self.snapshots_dir
            )
            snapshot_data = build_snapshot_data(
                self.current_project_data,
                self.current_project_path,
                description,
            )

            saved_path = self.current_project_path
            temp_data = self.current_project_data
            self.current_project_data = snapshot_data
            try:
                self._save_compressed(snapshot_path)
            finally:
                self.current_project_data = temp_data
                self.current_project_path = saved_path

            return True

        except Exception as e:
            logging.error(f"Failed to create snapshot: {e}")
            return False

    @_write_locked
    def restore_snapshot(self, snapshot_path: Path) -> bool:
        try:
            if resolve_inside(self.snapshots_dir, snapshot_path) is None:
                logging.error(
                    f"Refusing to restore snapshot outside "
                    f"{self.snapshots_dir}: {snapshot_path}"
                )
                return False
            snapshot_path = Path(snapshot_path)
            snapshot_data = self._load_compressed(snapshot_path)
            if snapshot_data:
                snapshot_data.pop("snapshot_info", None)
                self.current_project_data = snapshot_data
                return True
            return False

        except Exception as e:
            logging.error(f"Failed to restore snapshot: {e}")
            return False

    @_write_locked
    def save_as_template(self, template_name: str) -> bool:
        try:
            if not is_valid_new_name(template_name):
                logging.error(
                    f"Rejected invalid template name: {template_name!r}"
                )
                return False
            template_path = self._template_path_for(template_name)
            if template_path is None:
                return False

            template_data = self.current_project_data.copy()
            template_data["name"] = template_name
            template_data["is_template"] = True
            template_data.pop("created", None)
            template_data.pop("modified", None)

            saved_path = self.current_project_path
            temp_data = self.current_project_data
            self.current_project_data = template_data
            try:
                self._save_compressed(template_path)
            finally:
                self.current_project_data = temp_data
                self.current_project_path = saved_path

            return True

        except Exception as e:
            logging.error(f"Failed to save template: {e}")
            return False

    @_write_locked
    def load_template(self, template_name: str) -> dict[str, Any] | None:
        try:
            template_path = self._template_path_for(template_name)
            if template_path is None or not template_path.exists():
                return None

            saved_path = self.current_project_path
            saved_data = self.current_project_data
            try:
                result = self._load_compressed(template_path)
            finally:
                self.current_project_path = saved_path
                self.current_project_data = saved_data
            return result

        except Exception as e:
            logging.error(f"Failed to load template: {e}")
            return None

    @_read_locked
    def list_templates(self) -> list[str]:
        return [f.stem for f in self.templates_dir.glob("*.kproj")]

    @_read_locked
    def list_backups(self, project_name: str | None = None) -> list[Path]:
        if project_name:
            if not is_safe_name(project_name):
                logging.error(f"Rejected unsafe project name: {project_name!r}")
                return []
            pattern = f"{glob.escape(project_name)}_backup_*.kproj"
        else:
            pattern = "*_backup_*.kproj"

        return sorted(
            self.backups_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    @_read_locked
    def list_snapshots(self, project_name: str | None = None) -> list[Path]:
        if project_name:
            if not is_safe_name(project_name):
                logging.error(f"Rejected unsafe project name: {project_name!r}")
                return []
            pattern = f"{glob.escape(project_name)}_snapshot_*.kproj"
        else:
            pattern = "*_snapshot_*.kproj"

        return sorted(
            self.snapshots_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    def check_integrity(self, project_path: Path) -> tuple[bool, list[str]]:
        issues = []
        try:
            project_path = Path(project_path)

            if not project_path.exists():
                issues.append(f"Project file does not exist: {project_path}")
                return False, issues

            if project_path.suffix == ".kproj":
                try:
                    with zipfile.ZipFile(project_path, "r") as zf:
                        if "manifest.json" not in zf.namelist():
                            issues.append(
                                "Missing manifest.json in project archive"
                            )

                        bad_file = zf.testzip()
                        if bad_file:
                            issues.append(
                                f"Corrupted file in archive: {bad_file}"
                            )

                except zipfile.BadZipFile:
                    issues.append("Project file is not a valid ZIP archive")

            elif project_path.suffix == ".json":
                try:
                    with open(project_path, encoding="utf-8") as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid JSON: {e}")

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Integrity check failed: {e}")
            return False, issues

    @_write_locked
    def repair_project(self, project_path: Path) -> bool:
        try:
            project_path = Path(project_path)
            if not self.is_path_within_saved_data(project_path):
                logging.error(
                    f"Refusing to repair project outside "
                    f"{self.saved_data_dir}: {project_path}"
                )
                return False

            is_valid, issues = self.check_integrity(project_path)
            if is_valid:
                return True

            backup_path = project_path.with_suffix(
                project_path.suffix + ".backup"
            )
            with logged_file_op("Copy project", backup_path):
                shutil.copy2(project_path, backup_path)

            if project_path.suffix == ".kproj":
                try:
                    project_data = self._load_compressed(project_path)
                    if project_data:
                        self.current_project_data = project_data
                        self._save_compressed(project_path)
                        return True
                except Exception as e:
                    logging.error(
                        "Error repairing compressed project %s: %s",
                        project_path,
                        e,
                    )

            return False

        except Exception as e:
            logging.error(f"Failed to repair project: {e}")
            return False

    @_read_locked
    def get_project_stats(self) -> dict[str, Any]:
        if not self.current_project_data:
            return {}

        return {
            "name": self.current_project_data.get("name", "Untitled"),
            "tracks": len(self.current_project_data.get("tracks", [])),
            "markers": len(self.current_project_data.get("markers", [])),
            "assets": len(self.current_project_data.get("assets", [])),
            "created": self.current_project_data.get("created", ""),
            "modified": self.current_project_data.get("modified", ""),
            "version": self.current_project_data.get("version", ""),
            "size": self._calculate_project_size(),
        }

    def _calculate_project_size(self) -> int:
        if (
            not self.current_project_path
            or not self.current_project_path.exists()
        ):
            return 0
        return self.current_project_path.stat().st_size

    def _calculate_project_hash(self) -> str:
        def serialize_dict(obj):
            if isinstance(obj, dict):
                return tuple(
                    sorted((k, serialize_dict(v)) for k, v in obj.items())
                )
            elif isinstance(obj, list):
                return tuple(serialize_dict(item) for item in obj)
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        try:
            serialized = serialize_dict(self.current_project_data)
            hash_str = str(serialized)
            return hashlib.sha256(hash_str.encode()).hexdigest()
        except Exception as e:
            logging.warning(
                f"Failed to calculate structured hash, falling back to JSON: {e}"
            )
            project_json = json.dumps(self.current_project_data, sort_keys=True)
            return hashlib.sha256(project_json.encode()).hexdigest()

    @_write_locked
    def save_project_incremental(
        self, project_path: Path | None = None
    ) -> bool:
        if not self._incremental_save_enabled:
            return self.save_project(project_path)

        try:
            current_hash = self._calculate_project_hash()

            if self._last_saved_hash == current_hash:
                logging.debug("No changes detected, skipping save")
                return True

            success = self.save_project(project_path, compress=True)
            if success:
                self._last_saved_hash = current_hash

            return success

        except Exception as e:
            logging.error(f"Incremental save failed: {e}")
            return False

    @_write_locked
    def migrate_project(
        self, project_path: Path, target_version: str = __version__
    ) -> bool:
        try:
            project_path = Path(project_path)
            if not self.is_path_within_saved_data(project_path):
                logging.error(
                    f"Refusing to migrate project outside "
                    f"{self.saved_data_dir}: {project_path}"
                )
                return False

            project_data = self.load_project(project_path)
            if not project_data:
                return False

            original_version = project_data.get("version", "0.1.0")

            if original_version == target_version:
                logging.info(f"Project already at version {target_version}")
                return True

            logging.info(
                f"Migrating project from {original_version} to {target_version}"
            )

            migration_backup = project_path.with_suffix(
                f".{original_version}.backup"
            )
            with logged_file_op("Copy project", migration_backup):
                shutil.copy2(project_path, migration_backup)

            project_data = migrate_project_data(
                project_data, target_version, migration_backup
            )

            self.current_project_data = project_data
            success = self.save_project(project_path)

            if success:
                logging.info(
                    f"Project migrated successfully to {target_version}"
                )
            else:
                logging.error("Migration failed, restoring from backup")
                with logged_file_op("Restore project", project_path):
                    shutil.copy2(migration_backup, project_path)

            return success

        except Exception as e:
            logging.error(f"Project migration failed: {e}")
            return False

    def _recovery_file(self) -> Path:
        return self.base_dir / "saved_data" / ".recovery.gz"

    def enable_auto_recovery(self):
        return load_recovery_file(self._recovery_file())

    @_write_locked
    def save_recovery_state(self):
        return save_recovery_file(
            self._recovery_file(),
            self.current_project_path,
            self.current_project_data,
        )

    def clear_recovery_state(self):
        return clear_recovery_file(self._recovery_file())
