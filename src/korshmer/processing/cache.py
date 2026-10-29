import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from threading import Lock
from typing import Any

_logger = logging.getLogger(__name__)


class ProcessingCache:
    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 2048,
        ttl_seconds: int = 86400 * 7,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self._index: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._hit_count = 0
        self._access_count = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        now = time.time()
        try:
            files = list(self.cache_dir.glob("??/*.pkl"))
        except OSError as e:
            _logger.error(f"Failed to scan cache dir {self.cache_dir}: {e}")
            return

        for pkl_path in files:
            try:
                stat = pkl_path.stat()
            except OSError:
                continue
            if now - stat.st_mtime > self.ttl_seconds:
                try:
                    pkl_path.unlink()
                except OSError:
                    pass
                continue
            self._index[pkl_path.stem] = {
                "path": str(pkl_path),
                "size": stat.st_size,
                "created_at": stat.st_mtime,
                "accessed_at": stat.st_mtime,
            }

        self._evict_if_needed()

    def _make_key(self, *parts: Any) -> str:
        hash_input = ""
        for part in parts:
            try:
                if isinstance(part, str):
                    hash_input += part
                else:
                    hash_input += json.dumps(part, sort_keys=True)
            except (TypeError, ValueError):
                hash_input += str(part)

        return hashlib.sha256(hash_input.encode()).hexdigest()

    def make_file_key(
        self, file_path: str, model_version: str, params: dict
    ) -> str:
        try:
            stat = os.stat(file_path)
            hasher = hashlib.sha256()
            hasher.update(f"{stat.st_size}:{stat.st_mtime_ns}:".encode())
            with open(file_path, "rb") as f:
                hasher.update(f.read(65536))
            file_hash = hasher.hexdigest()
        except OSError:
            file_hash = hashlib.sha256(file_path.encode()).hexdigest()

        sorted_params = json.dumps(params, sort_keys=True, default=repr)
        return self._make_key(file_hash, model_version, sorted_params)

    def get(self, key: str) -> Any | None:
        with self._lock:
            self._access_count += 1

            if key not in self._index:
                return None

            entry = self._index[key]
            entry["accessed_at"] = time.time()

        pkl_subdir = self.cache_dir / key[:2]
        pkl_path = pkl_subdir / f"{key}.pkl"

        try:
            if not pkl_path.exists():
                with self._lock:
                    self._index.pop(key, None)
                return None

            created_at = entry.get("created_at", 0)
            if time.time() - created_at > self.ttl_seconds:
                with self._lock:
                    self._index.pop(key, None)
                try:
                    pkl_path.unlink()
                except OSError:
                    pass
                return None

            with open(pkl_path, "rb") as f:
                result = pickle.load(f)

            with self._lock:
                self._hit_count += 1

            return result

        except Exception as e:
            _logger.debug(f"Cache read error for {key}: {e}")
            with self._lock:
                self._index.pop(key, None)
            try:
                pkl_path.unlink()
            except OSError:
                pass
            return None

    def set(self, key: str, value: Any) -> bool:
        pkl_subdir = self.cache_dir / key[:2]
        try:
            pkl_subdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _logger.error(f"Failed to create cache subdir {pkl_subdir}: {e}")
            return False

        pkl_path = pkl_subdir / f"{key}.pkl"

        try:
            payload = pickle.dumps(value)
        except Exception as e:
            _logger.error(f"Failed to serialize cache value for {key}: {e}")
            return False

        tmp_path = pkl_subdir / f"{key}.{os.getpid()}.{id(payload)}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(payload)
            os.replace(tmp_path, pkl_path)
            file_size = len(payload)

        except OSError as e:
            _logger.error(f"Failed to write cache for {key}: {e}")
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return False

        current_time = time.time()
        with self._lock:
            self._index[key] = {
                "path": str(pkl_path),
                "size": file_size,
                "created_at": current_time,
                "accessed_at": current_time,
            }

        self._evict_if_needed()
        return True

    def _evict_if_needed(self):
        with self._lock:
            total_size = sum(entry["size"] for entry in self._index.values())
            max_bytes = self.max_size_mb * 1024 * 1024

            if total_size <= max_bytes:
                return

            sorted_entries = sorted(
                self._index.items(), key=lambda x: x[1].get("accessed_at", 0)
            )

            for key, entry in sorted_entries:
                if total_size <= max_bytes:
                    break

                pkl_path = Path(entry["path"])
                try:
                    pkl_path.unlink()
                except OSError:
                    pass

                total_size -= entry["size"]
                del self._index[key]

    def invalidate(self, key: str):
        with self._lock:
            self._index.pop(key, None)

        try:
            (self.cache_dir / key[:2] / f"{key}.pkl").unlink()
        except OSError:
            pass

    def clear(self):
        with self._lock:
            self._index.clear()

        try:
            import shutil

            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _logger.error(f"Failed to clear cache dir {self.cache_dir}: {e}")

    def stats(self) -> dict:
        with self._lock:
            entry_count = len(self._index)
            total_size_mb = sum(
                entry["size"] for entry in self._index.values()
            ) / (1024 * 1024)
            hit_rate = 0.0
            if self._access_count > 0:
                hit_rate = self._hit_count / self._access_count

        return {
            "entry_count": entry_count,
            "total_size_mb": total_size_mb,
            "hit_rate": hit_rate,
        }


_default_cache: ProcessingCache | None = None


def get_cache(cache_dir: Path | None = None) -> ProcessingCache:
    global _default_cache
    if _default_cache is None:
        if cache_dir is None:
            from korshmer.constants import BASE_DIR

            cache_dir = Path(BASE_DIR) / "cache"
        _default_cache = ProcessingCache(cache_dir)
    return _default_cache


def cached(model_version: str = "1.0"):
    from functools import wraps

    def decorator(fn):
        @wraps(fn)
        def wrapper(file_path: str, *args, **kwargs):
            cache = get_cache()
            key = cache.make_file_key(
                file_path, model_version, {"args": args, "kwargs": kwargs}
            )
            result = cache.get(key)
            if result is not None:
                return result
            result = fn(file_path, *args, **kwargs)
            if result is not None:
                cache.set(key, result)
            return result

        return wrapper

    return decorator
