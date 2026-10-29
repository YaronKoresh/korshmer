import json
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from korshmer.project_management.file_io_log import logged_file_op
from korshmer.utils.secure_random import make_private_dir


@contextmanager
def _replace_on_success(final_path):
    final_path = Path(final_path)
    handle, name = tempfile.mkstemp(
        dir=final_path.parent, prefix=final_path.name + ".", suffix=".tmp"
    )
    os.close(handle)
    pending = Path(name)
    try:
        yield pending
        os.replace(pending, final_path)
    finally:
        pending.unlink(missing_ok=True)


def save_compressed(project_data, project_path, temp_root):
    project_path = Path(project_path).with_suffix(".kproj")
    temp_dir = make_private_dir(temp_root, "project_")

    try:
        with logged_file_op("Save project", project_path):
            _write_archive(project_data, project_path, temp_dir)
        return project_path

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _archive_name(asset_path, used_names):
    name = asset_path.name
    counter = 1
    while name in used_names:
        name = f"{asset_path.stem}_{counter}{asset_path.suffix}"
        counter += 1
    used_names.add(name)
    return name


def _plan_assets(project_data):
    manifest = dict(project_data)
    assets = project_data.get("assets", [])
    if not isinstance(assets, list):
        return manifest, []
    used_names = set()
    planned = []
    manifest_assets = []
    for asset in assets:
        if not isinstance(asset, dict):
            manifest_assets.append(asset)
            continue
        entry = dict(asset)
        asset_path = Path(asset.get("path", ""))
        if asset_path.exists() and asset_path.is_file():
            arcname = _archive_name(asset_path, used_names)
            entry["archive_name"] = arcname
            planned.append((asset_path, f"assets/{arcname}"))
        manifest_assets.append(entry)
    manifest["assets"] = manifest_assets
    return manifest, planned


def _write_archive(project_data, project_path, temp_dir):
    manifest, planned = _plan_assets(project_data)
    manifest_path = temp_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    assets_dir = temp_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    with (
        _replace_on_success(project_path) as pending,
        zipfile.ZipFile(
            pending, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf,
    ):
        zf.write(manifest_path, "manifest.json")

        for asset_path, arcname in planned:
            zf.write(asset_path, arcname)


def save_uncompressed(project_data, project_path):
    project_path = Path(project_path).with_suffix(".json")

    with logged_file_op("Save project", project_path):
        with _replace_on_success(project_path) as pending:
            with open(pending, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

    return project_path
