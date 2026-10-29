import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from korshmer.project_management.file_io_log import logged_file_op
from korshmer.utils.search_input import sanitize_search_query


class AssetManager:
    _instances: dict[str, "AssetManager"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, base_dir: Path) -> "AssetManager":
        key = str(Path(base_dir).resolve())
        instance = cls._instances.get(key)
        if instance is not None:
            return instance
        with cls._instances_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = cls(base_dir)
                cls._instances[key] = instance
        return instance

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.assets_dir = self.base_dir / "saved_data" / "assets"
        self.library_file = self.base_dir / "saved_data" / "asset_library.json"
        self.favorites_file = self.base_dir / "saved_data" / "favorites.json"

        self._ensure_directories()
        self.asset_library: dict[str, dict[str, Any]] = self._load_library()
        self.favorites: set[str] = self._load_favorites()
        self.tags: dict[str, set[str]] = {}

    def _ensure_directories(self):
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _load_library(self) -> dict[str, dict[str, Any]]:
        if self.library_file.exists():
            try:
                with logged_file_op("Load asset library", self.library_file):
                    with open(self.library_file, encoding="utf-8") as f:
                        data = json.load(f)
                if not isinstance(data, dict):
                    raise TypeError("asset library must be an object")
                return {
                    asset_id: info
                    for asset_id, info in data.items()
                    if isinstance(info, dict)
                }
            except Exception as e:
                logging.error(f"Failed to load asset library: {e}")
        return {}

    def _save_library(self):
        try:
            with logged_file_op("Save asset library", self.library_file):
                tmp = str(self.library_file) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(
                        self.asset_library, f, indent=2, ensure_ascii=False
                    )
                os.replace(tmp, self.library_file)
        except Exception as e:
            logging.error(f"Failed to save asset library: {e}")

    def _load_favorites(self) -> set[str]:
        if self.favorites_file.exists():
            try:
                with logged_file_op("Load favorites", self.favorites_file):
                    with open(self.favorites_file, encoding="utf-8") as f:
                        data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError("favorites must be a list")
                return {item for item in data if isinstance(item, str)}
            except Exception as e:
                logging.error(f"Failed to load favorites: {e}")
        return set()

    def _save_favorites(self):
        try:
            with logged_file_op("Save favorites", self.favorites_file):
                tmp = str(self.favorites_file) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(list(self.favorites), f, indent=2)
                os.replace(tmp, self.favorites_file)
        except Exception as e:
            logging.error(f"Failed to save favorites: {e}")

    def add_asset(
        self,
        file_path: Path,
        tags: list[str] = None,
        copy_to_library: bool = True,
    ) -> str | None:
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logging.error(f"Asset file does not exist: {file_path}")
                return None

            asset_id = self._generate_asset_id(file_path)

            if copy_to_library:
                target_path = self.assets_dir / file_path.name
                counter = 1
                while target_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    target_path = self.assets_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                with logged_file_op("Copy asset", target_path):
                    shutil.copy2(file_path, target_path)
                stored_path = target_path
            else:
                stored_path = file_path

            asset_info = {
                "id": asset_id,
                "name": file_path.name,
                "path": str(stored_path),
                "original_path": str(file_path),
                "type": self._detect_asset_type(file_path),
                "size": file_path.stat().st_size,
                "added": datetime.now().isoformat(),
                "tags": tags or [],
                "metadata": self._extract_metadata(file_path),
                "usage_count": 0,
                "used_in_projects": [],
            }

            self.asset_library[asset_id] = asset_info
            self._save_library()

            return asset_id

        except Exception as e:
            logging.error(f"Failed to add asset: {e}")
            return None

    def _generate_asset_id(self, file_path: Path) -> str:
        content = f"{file_path.name}_{file_path.stat().st_size}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _detect_asset_type(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        audio_formats = {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".aac",
            ".m4a",
            ".wma",
            ".aiff",
            ".aif",
        }
        midi_formats = {".mid", ".midi"}
        preset_formats = {".json", ".preset"}

        if suffix in audio_formats:
            return "audio"
        elif suffix in midi_formats:
            return "midi"
        elif suffix in preset_formats:
            return "preset"
        else:
            return "other"

    def _extract_metadata(self, file_path: Path) -> dict[str, Any]:
        metadata = {}

        try:
            import soundfile as sf

            if file_path.suffix.lower() in {
                ".wav",
                ".flac",
                ".ogg",
                ".aiff",
                ".aif",
            }:
                info = sf.info(str(file_path))
                metadata["sample_rate"] = info.samplerate
                metadata["channels"] = info.channels
                metadata["duration"] = info.duration
                metadata["format"] = info.format

        except Exception as e:
            logging.debug(
                "Failed to extract audio metadata for %s: %s", file_path, e
            )

        return metadata

    def remove_asset(self, asset_id: str, delete_file: bool = False) -> bool:
        try:
            if asset_id not in self.asset_library:
                return False

            asset_info = self.asset_library[asset_id]

            if delete_file:
                asset_path = Path(asset_info["path"])
                if asset_path.exists() and asset_path.parent == self.assets_dir:
                    with logged_file_op(
                        "Delete asset", asset_path, report_size=False
                    ):
                        asset_path.unlink()

            del self.asset_library[asset_id]
            self.favorites.discard(asset_id)

            self._save_library()
            self._save_favorites()

            return True

        except Exception as e:
            logging.error(f"Failed to remove asset: {e}")
            return False

    def add_tag(self, asset_id: str, tag: str) -> bool:
        try:
            if asset_id not in self.asset_library:
                return False

            tags = self.asset_library[asset_id].get("tags", [])
            if tag not in tags:
                tags.append(tag)
                self.asset_library[asset_id]["tags"] = tags
                self._save_library()

            return True

        except Exception as e:
            logging.error(f"Failed to add tag: {e}")
            return False

    def remove_tag(self, asset_id: str, tag: str) -> bool:
        try:
            if asset_id not in self.asset_library:
                return False

            tags = self.asset_library[asset_id].get("tags", [])
            if tag in tags:
                tags.remove(tag)
                self.asset_library[asset_id]["tags"] = tags
                self._save_library()

            return True

        except Exception as e:
            logging.error(f"Failed to remove tag: {e}")
            return False

    def add_to_favorites(self, asset_id: str) -> bool:
        if asset_id in self.asset_library:
            self.favorites.add(asset_id)
            self._save_favorites()
            return True
        return False

    def remove_from_favorites(self, asset_id: str) -> bool:
        if asset_id in self.favorites:
            self.favorites.discard(asset_id)
            self._save_favorites()
            return True
        return False

    def search_assets(
        self,
        query: str = "",
        tags: list[str] = None,
        asset_type: str = None,
        favorites_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = sanitize_search_query(query)
        results = []

        for asset_id, asset_info in self.asset_library.items():
            if favorites_only and asset_id not in self.favorites:
                continue

            if asset_type and asset_info.get("type") != asset_type:
                continue

            if tags:
                asset_tags = set(asset_info.get("tags", []))
                if not asset_tags.intersection(tags):
                    continue

            if query:
                query_lower = query.lower()
                name_lower = asset_info.get("name", "").lower()
                tags_str = " ".join(asset_info.get("tags", [])).lower()

                if (
                    query_lower not in name_lower
                    and query_lower not in tags_str
                ):
                    continue

            results.append(asset_info)

        return results

    def get_all_tags(self) -> list[str]:
        all_tags = set()
        for asset_info in self.asset_library.values():
            all_tags.update(asset_info.get("tags", []))
        return sorted(all_tags)

    def track_usage(self, asset_id: str, project_name: str):
        if asset_id in self.asset_library:
            self.asset_library[asset_id]["usage_count"] += 1

            used_in = self.asset_library[asset_id].get("used_in_projects", [])
            if project_name not in used_in:
                used_in.append(project_name)
                self.asset_library[asset_id]["used_in_projects"] = used_in

            self._save_library()

    def find_duplicates(self) -> list[list[str]]:
        size_groups = {}

        for asset_id, asset_info in self.asset_library.items():
            size = asset_info.get("size", 0)
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(asset_id)

        duplicates = []
        for asset_ids in size_groups.values():
            if len(asset_ids) > 1:
                duplicates.append(asset_ids)

        return duplicates

    def bulk_tag(self, asset_ids: list[str], tag: str) -> int:
        count = 0
        for asset_id in asset_ids:
            if self.add_tag(asset_id, tag):
                count += 1
        return count

    def bulk_remove(
        self, asset_ids: list[str], delete_files: bool = False
    ) -> int:
        count = 0
        for asset_id in asset_ids:
            if self.remove_asset(asset_id, delete_files):
                count += 1
        return count

    def get_asset_stats(self) -> dict[str, Any]:
        total_size = sum(
            asset.get("size", 0) for asset in self.asset_library.values()
        )

        type_counts = {}
        for asset in self.asset_library.values():
            asset_type = asset.get("type", "other")
            type_counts[asset_type] = type_counts.get(asset_type, 0) + 1

        return {
            "total_assets": len(self.asset_library),
            "total_size": total_size,
            "favorites_count": len(self.favorites),
            "type_counts": type_counts,
            "total_tags": len(self.get_all_tags()),
        }
