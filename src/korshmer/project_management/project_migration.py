from __future__ import annotations

from datetime import datetime
from typing import Any

from korshmer.constants import MAX_EFFECT_CHAIN_DEPTH


def validate_project_effects_chain(project_data: Any) -> None:
    if not isinstance(project_data, dict):
        return
    audio = project_data.get("audio_processing")
    chain = audio.get("effects_chain") if isinstance(audio, dict) else None
    if isinstance(chain, list) and len(chain) > MAX_EFFECT_CHAIN_DEPTH:
        raise ValueError(
            f"Project effects chain depth must be at most "
            f"{MAX_EFFECT_CHAIN_DEPTH} effects, got {len(chain)}"
        )


def needs_migration(current: str, target: str) -> bool:
    try:
        current_parts = [int(x) for x in current.split(".")]
        target_parts = [int(x) for x in target.split(".")]

        while len(current_parts) < len(target_parts):
            current_parts.append(0)
        while len(target_parts) < len(current_parts):
            target_parts.append(0)

        return current_parts < target_parts
    except (ValueError, AttributeError):
        return True


def _migrate_to_0_2_0(data: dict[str, Any]) -> dict[str, Any]:
    data = data.copy()
    if "ui_settings" not in data:
        data["ui_settings"] = {
            "frameless": True,
            "custom_titlebar": True,
            "theme": "dark",
        }
    return data


def _migrate_to_0_3_0(data: dict[str, Any]) -> dict[str, Any]:
    data = data.copy()
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and "bpm" not in metadata:
        data["metadata"] = metadata.copy()
        data["metadata"]["bpm"] = 120
        metadata = data["metadata"]
    if "theory_settings" not in data:
        data["theory_settings"] = {
            "bpm_override": True,
            "global_bpm": (
                metadata.get("bpm", 120) if isinstance(metadata, dict) else 120
            ),
        }
    return data


def _migrate_to_0_4_0(data: dict[str, Any]) -> dict[str, Any]:
    data = data.copy()
    if "audio_processing" not in data:
        data["audio_processing"] = {
            "spectral_analysis": {},
            "stem_separation": {},
            "effects_chain": [],
        }
    return data


def _migrate_to_0_5_0(data: dict[str, Any]) -> dict[str, Any]:
    data = data.copy()
    if "ai_settings" not in data:
        data["ai_settings"] = {
            "audio_to_midi_engine": "basic-pitch",
            "music_generation_model": None,
            "smart_mixing_enabled": False,
        }
    return data


def _migrate_to_0_5_1(data: dict[str, Any]) -> dict[str, Any]:
    data = data.copy()
    if "project_management" not in data:
        data["project_management"] = {
            "version_control_enabled": False,
            "auto_backup_enabled": True,
            "cloud_sync_enabled": False,
            "collaboration_enabled": False,
        }

    if "asset_management" not in data:
        data["asset_management"] = {
            "library_enabled": True,
            "auto_organize": True,
            "favorites": [],
        }

    return data


MIGRATIONS = [
    ("0.2.0", _migrate_to_0_2_0),
    ("0.3.0", _migrate_to_0_3_0),
    ("0.4.0", _migrate_to_0_4_0),
    ("0.5.0", _migrate_to_0_5_0),
    ("0.5.1", _migrate_to_0_5_1),
]


def migrate_project_data(
    project_data: dict[str, Any], target_version: str, backup_path
) -> dict[str, Any]:
    validate_project_effects_chain(project_data)
    original_version = project_data.get("version", "0.1.0")
    current_version = original_version

    for version, migration_func in MIGRATIONS:
        if needs_migration(current_version, version) and not needs_migration(
            target_version, version
        ):
            project_data = migration_func(project_data)
            current_version = version
            if current_version == target_version:
                break

    project_data["version"] = target_version
    history = project_data.get("migration_history")
    project_data["migration_history"] = (
        list(history) if isinstance(history, list) else []
    )
    project_data["migration_history"].append(
        {
            "from_version": original_version,
            "to_version": target_version,
            "timestamp": datetime.now().isoformat(),
            "backup_path": str(backup_path),
        }
    )
    return project_data
