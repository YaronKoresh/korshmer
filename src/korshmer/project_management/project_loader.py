import hashlib
import json
import logging
import shutil
import zipfile
from pathlib import Path

from korshmer.constants import (
    MAX_PROJECT_ARCHIVE_MEMBER_BYTES,
    MAX_PROJECT_ARCHIVE_TOTAL_BYTES,
    MAX_PROJECT_MANIFEST_BYTES,
)
from korshmer.exceptions import InvalidFileError
from korshmer.project_management.file_io_log import logged_file_op
from korshmer.utils.audit_log import OUTCOME_DENIED, audit
from korshmer.utils.secure_random import make_private_dir


def is_safe_archive_member(member_filename, base_extract_path):
    member_path = Path(member_filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        logging.error(f"Unsafe path in project archive: {member_filename}")
        audit(
            "archive.member",
            OUTCOME_DENIED,
            reason="unsafe_path",
            member=member_filename,
        )
        return False
    target_path = (base_extract_path / member_path).resolve()
    if not target_path.is_relative_to(Path(base_extract_path).resolve()):
        logging.error(
            f"Path escapes extraction directory in project archive: "
            f"{member_filename}"
        )
        audit(
            "archive.member",
            OUTCOME_DENIED,
            reason="escapes_extraction_dir",
            member=member_filename,
        )
        return False
    return True


def _deny_archive_size(reason, **details):
    audit("archive.size", OUTCOME_DENIED, reason=reason, **details)
    raise InvalidFileError(f"Project archive is too large ({reason})")


def check_archive_sizes(zf, base_extract_path):
    total = 0
    for member in zf.infolist():
        if member.is_dir() or not is_safe_archive_member(
            member.filename, base_extract_path
        ):
            continue
        limit = (
            MAX_PROJECT_MANIFEST_BYTES
            if member.filename == "manifest.json"
            else MAX_PROJECT_ARCHIVE_MEMBER_BYTES
        )
        if member.file_size > limit:
            _deny_archive_size(
                "member_too_large",
                member=member.filename,
                size=member.file_size,
            )
        total += member.file_size
        if total > MAX_PROJECT_ARCHIVE_TOTAL_BYTES:
            _deny_archive_size("total_too_large", size=total)


def file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_available_name(target_assets, asset_file):
    base_name = asset_file.stem
    suffix = asset_file.suffix
    counter = 1
    while counter < 10000:
        candidate = target_assets / f"{base_name}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
    raise RuntimeError(f"Could not find unique filename for {asset_file.name}")


def _copy_assets(assets_dir, target_assets):
    target_assets.mkdir(parents=True, exist_ok=True)
    restored = {}

    for asset_file in assets_dir.iterdir():
        if not asset_file.is_file():
            continue

        dest_path = target_assets / asset_file.name
        if dest_path.exists():
            try:
                if file_hash(asset_file) == file_hash(dest_path):
                    restored[asset_file.name] = dest_path
                    continue
            except Exception as e:
                logging.warning(
                    f"Failed to compare asset files '{asset_file}' and "
                    f"'{dest_path}': {e}"
                )

            dest_path = _find_available_name(target_assets, asset_file)

        shutil.copy2(asset_file, dest_path)
        restored[asset_file.name] = dest_path
    return restored


def _relink_assets(project_data, restored):
    assets = project_data.get("assets")
    if not isinstance(assets, list):
        return
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        archive_name = asset.pop("archive_name", None)
        if archive_name in restored:
            asset["path"] = str(restored[archive_name])


def load_compressed(project_path, temp_root, assets_target):
    project_path = Path(project_path)
    with logged_file_op("Load project", project_path):
        return _read_archive(project_path, temp_root, assets_target)


def _read_archive(project_path, temp_root, assets_target):
    with zipfile.ZipFile(project_path, "r") as zf:
        check_archive_sizes(zf, Path(temp_root).resolve())
        with zf.open("manifest.json") as f:
            project_data = json.loads(f.read().decode("utf-8"))

        extract_dir = make_private_dir(temp_root, "extract_")
        try:
            base_extract_path = extract_dir.resolve()
            for member in zf.infolist():
                if not is_safe_archive_member(
                    member.filename, base_extract_path
                ):
                    continue
                zf.extract(member, extract_dir)

            assets_dir = extract_dir / "assets"
            if assets_dir.exists():
                restored = _copy_assets(assets_dir, Path(assets_target))
                _relink_assets(project_data, restored)
        finally:
            try:
                shutil.rmtree(extract_dir, ignore_errors=True)
            except Exception as e:
                logging.error(
                    f"Failed to clean up temporary extraction directory "
                    f"'{extract_dir}': {e}"
                )

    return project_data


def load_uncompressed(project_path):
    project_path = Path(project_path)
    with logged_file_op("Load project", project_path):
        with open(project_path, encoding="utf-8") as f:
            project_data = json.load(f)
    return project_data
