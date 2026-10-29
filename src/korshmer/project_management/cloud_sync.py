import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from korshmer.exceptions import InvalidFileError
from korshmer.project_management.validators.project_validator import (
    is_safe_name,
)
from korshmer.utils.filesystem import sanitize_path

PROJECT_SUFFIXES = (".kproj", ".json")
FOLDER_PROVIDER = "folder"
STATE_FILE_NAME = "sync_state.json"
RESOLUTIONS = ("local", "remote")
MIN_AUTO_SYNC_MINUTES = 1
MAX_AUTO_SYNC_MINUTES = 7 * 24 * 60


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=".sync_", suffix=".tmp"
    )
    os.close(fd)
    try:
        shutil.copyfile(source, tmp_name)
        os.replace(tmp_name, dest)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


class CloudSyncManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.sync_dir = self.base_dir / "saved_data" / "cloud_sync"
        self.projects_dir = self.base_dir / "saved_data" / "projects"
        self.enabled = False
        self.provider = None
        self.remote_dir: Path | None = None
        self.last_sync: str | None = None
        self._lock = threading.RLock()
        self._synced: dict[str, str] = {}
        self._conflicts: dict[str, Path] = {}
        self._auto_thread: threading.Thread | None = None
        self._auto_stop = threading.Event()
        logging.info("Cloud sync manager initialized")

    @property
    def _state_path(self) -> Path:
        return self.sync_dir / STATE_FILE_NAME

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            synced = data["synced"]
            if not isinstance(synced, dict):
                raise TypeError("synced must be an object")
            self._synced = {
                k: v
                for k, v in synced.items()
                if isinstance(k, str) and isinstance(v, str)
            }
            last_sync = data.get("last_sync")
            self.last_sync = last_sync if isinstance(last_sync, str) else None
        except (OSError, ValueError, KeyError, TypeError):
            self._synced = {}
            self.last_sync = None

    def _save_state(self) -> None:
        try:
            self.sync_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"synced": self._synced, "last_sync": self.last_sync},
                indent=2,
            )
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as e:
            logging.error(f"Failed to save cloud sync state: {e}")

    @staticmethod
    def _valid_id(cloud_id) -> bool:
        return (
            isinstance(cloud_id, str)
            and is_safe_name(cloud_id)
            and cloud_id.endswith(PROJECT_SUFFIXES)
            and not cloud_id.startswith(".")
        )

    def _remote_path(self, cloud_id) -> Path | None:
        if self.remote_dir is None or not self._valid_id(cloud_id):
            return None
        return self.remote_dir / cloud_id

    def connect(self, provider: str, _credentials: dict[str, str]) -> bool:
        if provider != FOLDER_PROVIDER:
            logging.warning(f"Cloud sync for {provider} not yet implemented")
            return False
        try:
            raw = _credentials["path"]
            if not isinstance(raw, (str, Path)) or not str(raw).strip():
                raise ValueError("empty path")
            sanitize_path(raw)
            remote = Path(raw).expanduser().resolve()
            remote.mkdir(parents=True, exist_ok=True)
            if not os.access(remote, os.W_OK):
                raise PermissionError(f"Folder is not writable: {remote}")
            sync_root = self.sync_dir.resolve()
            if remote == sync_root or sync_root in remote.parents:
                raise ValueError("Folder cannot be inside the sync state dir")
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            InvalidFileError,
        ) as e:
            logging.error(f"Cloud sync could not use the folder: {e}")
            return False
        with self._lock:
            self.remote_dir = remote
            self.provider = FOLDER_PROVIDER
            self.enabled = True
            self._conflicts = {}
            self._load_state()
        return True

    def disconnect(self) -> bool:
        self.disable_auto_sync()
        with self._lock:
            self.enabled = False
            self.provider = None
            self.remote_dir = None
            self._conflicts = {}
        return True

    def _local_project(self, project_path) -> Path | None:
        try:
            sanitize_path(project_path)
            path = Path(project_path)
        except (TypeError, ValueError, OSError, InvalidFileError) as e:
            logging.error(f"Rejected cloud sync path: {e}")
            return None
        if not self._valid_id(path.name) or not path.is_file():
            logging.error(f"Cloud sync needs a project file: {project_path!r}")
            return None
        return path

    def _mark_synced(self, cloud_id: str, digest: str) -> None:
        self._synced[cloud_id] = digest
        self._conflicts.pop(cloud_id, None)
        self.last_sync = datetime.now().isoformat()
        self._save_state()

    def sync_project(self, project_path: Path) -> bool:
        with self._lock:
            if not self.enabled or self.remote_dir is None:
                logging.warning("Cloud sync is not connected")
                return False
            local = self._local_project(project_path)
            if local is None:
                return False
            cloud_id = local.name
            remote = self.remote_dir / cloud_id
            try:
                local_hash = _file_hash(local)
                if remote.is_file():
                    remote_hash = _file_hash(remote)
                    if remote_hash == local_hash:
                        self._mark_synced(cloud_id, local_hash)
                        return True
                    known = self._synced.get(cloud_id)
                    if known is not None and known == remote_hash:
                        pass
                    elif known is not None and known == local_hash:
                        logging.info(
                            f"Cloud copy of {cloud_id} is newer; "
                            "download it to update"
                        )
                        return False
                    else:
                        self._conflicts[cloud_id] = local
                        logging.warning(
                            f"Cloud sync conflict for {cloud_id}; "
                            "resolve it first"
                        )
                        return False
                _copy_atomic(local, remote)
                self._mark_synced(cloud_id, local_hash)
                return True
            except OSError as e:
                logging.error(f"Cloud sync failed for {cloud_id}: {e}")
                return False

    def download_project(self, _cloud_id: str, _local_path: Path) -> bool:
        with self._lock:
            remote = self._remote_path(_cloud_id)
            if not self.enabled or remote is None:
                logging.warning("Cloud download needs a connection and a name")
                return False
            try:
                sanitize_path(_local_path)
                local = Path(_local_path)
                if not remote.is_file():
                    logging.error(f"Cloud project not found: {_cloud_id}")
                    return False
                digest = _file_hash(remote)
                _copy_atomic(remote, local)
                if local.name == _cloud_id:
                    self._mark_synced(_cloud_id, digest)
                return True
            except (TypeError, ValueError, OSError, InvalidFileError) as e:
                logging.error(f"Cloud download failed for {_cloud_id}: {e}")
                return False

    def list_cloud_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.enabled or self.remote_dir is None:
                return []
            try:
                entries = sorted(self.remote_dir.iterdir())
            except OSError as e:
                logging.error(f"Cloud project listing failed: {e}")
                return []
            projects = []
            for entry in entries:
                if not self._valid_id(entry.name):
                    continue
                try:
                    if not entry.is_file():
                        continue
                    stat = entry.stat()
                except OSError:
                    continue
                projects.append(
                    {
                        "id": entry.name,
                        "name": entry.stem,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                    }
                )
            return projects

    def _local_projects(self) -> list[Path]:
        try:
            return sorted(
                p
                for p in self.projects_dir.iterdir()
                if self._valid_id(p.name) and p.is_file()
            )
        except OSError:
            return []

    def sync_all(self) -> dict[str, bool]:
        results = {}
        for path in self._local_projects():
            results[path.name] = self.sync_project(path)
        return results

    def enable_auto_sync(self, _interval_minutes: int = 30) -> bool:
        if isinstance(_interval_minutes, bool) or not isinstance(
            _interval_minutes, (int, float)
        ):
            return False
        if not math.isfinite(_interval_minutes) or not (
            MIN_AUTO_SYNC_MINUTES <= _interval_minutes <= MAX_AUTO_SYNC_MINUTES
        ):
            return False
        with self._lock:
            if not self.enabled:
                logging.warning("Auto-sync needs a cloud connection")
                return False
        self.disable_auto_sync()
        stop = threading.Event()
        interval = float(_interval_minutes) * 60.0

        def work():
            while not stop.wait(interval):
                try:
                    self.sync_all()
                except Exception as e:
                    logging.error(f"Auto-sync run failed: {e}")

        thread = threading.Thread(target=work, daemon=True)
        with self._lock:
            self._auto_stop = stop
            self._auto_thread = thread
        thread.start()
        return True

    def disable_auto_sync(self) -> bool:
        with self._lock:
            thread = self._auto_thread
            self._auto_stop.set()
            self._auto_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        return True

    @property
    def auto_sync_active(self) -> bool:
        thread = self._auto_thread
        return thread is not None and thread.is_alive()

    def get_sync_status(self) -> dict[str, Any]:
        with self._lock:
            pending_uploads = 0
            pending_downloads = 0
            if self.enabled and self.remote_dir is not None:
                local_ids = set()
                for path in self._local_projects():
                    local_ids.add(path.name)
                    try:
                        if _file_hash(path) != self._synced.get(path.name):
                            pending_uploads += 1
                    except OSError:
                        continue
                for item in self.list_cloud_projects():
                    if item["id"] in local_ids:
                        continue
                    pending_downloads += 1
            return {
                "enabled": self.enabled,
                "provider": self.provider,
                "last_sync": self.last_sync,
                "pending_uploads": pending_uploads,
                "pending_downloads": pending_downloads,
                "conflicts": sorted(str(k) for k in self._conflicts),
            }

    def resolve_conflict(self, file_path: Path, resolution: str) -> bool:
        with self._lock:
            if not self.enabled or resolution not in RESOLUTIONS:
                return False
            try:
                local = Path(file_path)
            except TypeError:
                return False
            cloud_id = local.name
            remote = self._remote_path(cloud_id)
            if remote is None or cloud_id not in self._conflicts:
                return False
            try:
                if resolution == "local":
                    _copy_atomic(local, remote)
                    self._mark_synced(cloud_id, _file_hash(local))
                else:
                    digest = _file_hash(remote)
                    _copy_atomic(remote, local)
                    self._mark_synced(cloud_id, digest)
                return True
            except OSError as e:
                logging.error(f"Conflict resolution failed for {cloud_id}: {e}")
                return False
