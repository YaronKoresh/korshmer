import re
from pathlib import Path

from korshmer.constants import RESERVED_WINDOWS_NAMES
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
