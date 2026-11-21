import re
from pathlib import Path

from korshmer.constants import MAX_EFFECT_CHAIN_DEPTH, RESERVED_WINDOWS_NAMES
from korshmer.exceptions import ProjectCorruptedError
from korshmer.utils.audit_log import OUTCOME_DENIED, audit

MAX_FILESYSTEM_NAME_LENGTH = 255


FILESYSTEM_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

INVALID_NAME_MESSAGE = (
    'Name cannot be empty, "." or "..", and cannot contain "/", "\\" '
    "or null characters. It also cannot contain any of the characters "
    '< > : " | ? * or control characters (such as tab or newline), '
    "cannot be a reserved name such as CON or NUL, cannot end with a "
    "space or a dot, and cannot be longer than "
    f"{MAX_FILESYSTEM_NAME_LENGTH} characters."
)


def is_safe_name(name) -> bool:
    if not isinstance(name, str):
        return False
    if name.strip() in ("", ".", ".."):
        return False
    return not any(ch in name for ch in ("/", "\\", "\x00"))


def is_valid_new_name(name) -> bool:
    if not is_safe_name(name):
        return False
    if FILESYSTEM_UNSAFE_CHARS_RE.search(name):
        return False
    if name != name.rstrip(" ."):
        return False
    if len(name) > MAX_FILESYSTEM_NAME_LENGTH:
        return False
    base = name.split(".", 1)[0].strip().upper()
    if base in RESERVED_WINDOWS_NAMES:
        return False
    return True


def resolve_inside(base, candidate) -> Path | None:
    try:
        if "\x00" in str(candidate):
            audit(
                "path.resolve_inside",
                OUTCOME_DENIED,
                reason="null_byte",
                base=base,
                path=candidate,
            )
            return None
        base_resolved = Path(base).resolve()
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = Path.cwd() / candidate_path
        resolved = candidate_path.resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        audit(
            "path.resolve_inside",
            OUTCOME_DENIED,
            reason="unresolvable",
            base=base,
            path=candidate,
        )
        return None
    if resolved == base_resolved or not resolved.is_relative_to(base_resolved):
        audit(
            "path.resolve_inside",
            OUTCOME_DENIED,
            reason="outside_base",
            base=base_resolved,
            path=resolved,
        )
        return None
    return resolved


def validate_project_effects_chain(project_data) -> None:
    if not isinstance(project_data, dict):
        return
    audio = project_data.get("audio_processing")
    chain = audio.get("effects_chain") if isinstance(audio, dict) else None
    if isinstance(chain, list) and len(chain) > MAX_EFFECT_CHAIN_DEPTH:
        raise ValueError(
            f"Project effects chain depth must be at most "
            f"{MAX_EFFECT_CHAIN_DEPTH} effects, got {len(chain)}"
        )


SECTION_TYPES: dict[str, type] = {
    "metadata": dict,
    "tracks": list,
    "markers": list,
    "assets": list,
    "settings": dict,
}
TYPE_WORDS = {dict: "an object", list: "a list", str: "text"}
EDITOR_WHOLE_NUMBERS = {
    "tracks": ("id",),
    "notes": ("id", "track_id"),
    "clips": ("id",),
}
EDITOR_NUMBERS = {
    "tracks": (),
    "notes": ("start", "duration"),
    "clips": ("start", "duration", "gain"),
}
EDITOR_ENTRY_WORDS = {"tracks": "track", "notes": "note", "clips": "clip"}
BROKEN_SONG_HINT = "Open one of the backups of this song instead."


def _broken(problem) -> ProjectCorruptedError:
    return ProjectCorruptedError(
        f"The song file is damaged: {problem}", recovery_hint=BROKEN_SONG_HINT
    )


def _require_type(value, expected, label) -> None:
    if not isinstance(value, expected):
        raise _broken(
            f"{label} should be {TYPE_WORDS[expected]}, found "
            f"{type(value).__name__}"
        )


def _reads_as(value, number_type) -> bool:
    try:
        number_type(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _require_numbers(entry, keys, number_type, label) -> None:
    for key in keys:
        if key in entry and not _reads_as(entry[key], number_type):
            raise _broken(
                f"{key} of {label} should be a number, found {entry[key]!r}"
            )


def _validate_editor_entries(editor, section) -> None:
    entries = editor.get(section)
    if entries is None:
        return
    _require_type(entries, list, f"editor {section}")
    word = EDITOR_ENTRY_WORDS[section]
    for position, entry in enumerate(entries, 1):
        label = f"editor {word} {position}"
        _require_type(entry, dict, label)
        _require_numbers(entry, EDITOR_WHOLE_NUMBERS[section], int, label)
        _require_numbers(entry, EDITOR_NUMBERS[section], float, label)
        program = entry.get("program", 0)
        if section == "tracks" and not isinstance(program, (int, float)):
            raise _broken(
                f"program of {label} should be a number, found {program!r}"
            )


def validate_project_structure(project_data, complete=True) -> None:
    _require_type(project_data, dict, "the song")
    if complete or project_data.get("name") is not None:
        _require_type(project_data.get("name"), str, "name")
    for key, expected in SECTION_TYPES.items():
        if project_data.get(key) is None:
            if complete:
                raise _broken(f"the {key} section is missing")
            continue
        _require_type(project_data[key], expected, key)
    editor = project_data.get("editor")
    if editor is None:
        return
    _require_type(editor, dict, "editor")
    for section in EDITOR_ENTRY_WORDS:
        _validate_editor_entries(editor, section)
