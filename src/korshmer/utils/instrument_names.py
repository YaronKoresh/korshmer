import re

MAX_INSTRUMENT_NAME_LENGTH = 128

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_instrument_name(name) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", name)
    cleaned = _LONE_SURROGATE_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned[:MAX_INSTRUMENT_NAME_LENGTH]
