from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from typing import Any

from korshmer.exceptions import KorshmerError, ProjectLoadError
from korshmer.utils.track_layout import merge_by_instrument


async def read_project(path: Path) -> dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()

        def _read_zip():
            with zipfile.ZipFile(path, "r") as zf:
                with zf.open("metadata.json") as f:
                    return json.load(f)

        result = await loop.run_in_executor(None, _read_zip)
        if not isinstance(result, dict):
            raise ValueError(
                f"expected a JSON object, got {type(result).__name__}"
            )
        return result
    except Exception as e:
        raise ProjectLoadError(
            f"Failed to load project: {e}", context={"path": str(path)}
        ) from e


async def write_project(path: Path, data: dict[str, Any]) -> None:
    try:
        loop = asyncio.get_event_loop()

        def _write_zip():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.parent / f".{path.name}.tmp"

            try:
                with zipfile.ZipFile(
                    temp_path, "w", zipfile.ZIP_DEFLATED
                ) as zf:
                    zf.writestr("metadata.json", json.dumps(data, indent=2))

                temp_path.replace(path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise

        await loop.run_in_executor(None, _write_zip)
    except Exception as e:
        raise KorshmerError(
            f"Failed to save project: {e}", context={"path": str(path)}
        ) from e


async def read_presets(path: Path) -> dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()

        def _read_json():
            if not path.exists():
                return {}
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        result = await loop.run_in_executor(None, _read_json)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


async def write_presets(path: Path, data: dict[str, Any]) -> None:
    try:
        loop = asyncio.get_event_loop()

        def _write_json():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.parent / f".{path.name}.tmp"

            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                temp_path.replace(path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise

        await loop.run_in_executor(None, _write_json)
    except Exception as e:
        raise KorshmerError(
            f"Failed to save presets: {e}", context={"path": str(path)}
        ) from e


async def export_midi(midi_obj: Any, output_path: Path) -> Path:
    loop = asyncio.get_event_loop()

    def _export():
        midi_obj.instruments = merge_by_instrument(
            midi_obj.instruments, keep_shared_names=True
        )
        midi_obj.write(str(output_path))
        return output_path

    result = await loop.run_in_executor(None, _export)
    return result


async def list_audio_files(
    directory: Path, extensions: tuple = (".wav", ".mp3", ".flac", ".ogg")
) -> list:
    loop = asyncio.get_event_loop()

    def _scan_dir():
        if not directory.exists():
            return []
        files = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
        return sorted(files)

    result = await loop.run_in_executor(None, _scan_dir)
    return result
