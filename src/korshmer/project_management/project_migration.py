from __future__ import annotations

from collections.abc import Callable
from typing import Any

from korshmer.exceptions import ProjectCorruptedError, ProjectVersionError
from korshmer.project_management.validators.project_validator import (
    BROKEN_SONG_HINT,
    SECTION_TYPES,
    validate_project_structure,
)

FORMAT_VERSION_KEY = "format_version"
FIRST_FORMAT_VERSION = 1
PROJECT_FORMAT_VERSION = 2
CENTRE_PAN = 0.0


def project_format_version(project_data: dict[str, Any]) -> int:
    version = project_data.get(FORMAT_VERSION_KEY, FIRST_FORMAT_VERSION)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < FIRST_FORMAT_VERSION
    ):
        raise ProjectCorruptedError(
            "The song file is damaged: its format version should be a "
            f"whole number, found {version!r}",
            recovery_hint=BROKEN_SONG_HINT,
        )
    return version


def with_current_sections(
    project_data: dict[str, Any], fallback_name: str
) -> dict[str, Any]:
    if project_data.get("name") is None:
        project_data["name"] = fallback_name
    for key, section_type in SECTION_TYPES.items():
        if project_data.get(key) is None:
            project_data[key] = section_type()
    project_data[FORMAT_VERSION_KEY] = PROJECT_FORMAT_VERSION
    return project_data


def _centre_everything(project_data: dict[str, Any]) -> None:
    editor = project_data.get("editor")
    if not isinstance(editor, dict):
        return
    for section in ("tracks", "clips"):
        for entry in editor.get(section) or []:
            entry.setdefault("pan", CENTRE_PAN)


MIGRATIONS: dict[int, Callable[[dict[str, Any]], None]] = {
    FIRST_FORMAT_VERSION: _centre_everything,
}


def opened_project(project_data: Any, fallback_name: str) -> dict[str, Any]:
    if not isinstance(project_data, dict):
        raise ProjectCorruptedError(
            "The song file is damaged: it should hold an object, found "
            f"{type(project_data).__name__}",
            recovery_hint=BROKEN_SONG_HINT,
        )
    version = project_format_version(project_data)
    if version > PROJECT_FORMAT_VERSION:
        raise ProjectVersionError(
            "This song was saved by a newer version of KORSHMER (song "
            f"format {version}); this version opens songs up to format "
            f"{PROJECT_FORMAT_VERSION}.",
            recovery_hint="Update KORSHMER to open this song.",
            context={"found": version, "supported": PROJECT_FORMAT_VERSION},
        )
    validate_project_structure(
        project_data, complete=version >= PROJECT_FORMAT_VERSION
    )
    for step in range(version, PROJECT_FORMAT_VERSION):
        MIGRATIONS[step](project_data)
    return with_current_sections(project_data, fallback_name)
