import logging
import os
from pathlib import PurePath

LOG_CONSOLE_LEVEL_ENV = "KORSHMER_LOG_CONSOLE_LEVEL"
LOG_FILE_LEVEL_ENV = "KORSHMER_LOG_FILE_LEVEL"
LOG_INCLUDE_ENV = "KORSHMER_LOG_INCLUDE"
LOG_EXCLUDE_ENV = "KORSHMER_LOG_EXCLUDE"

_PACKAGE = "korshmer"


def parse_components(spec):
    return tuple(
        part.strip() for part in (spec or "").split(",") if part.strip()
    )


def parse_level(value):
    name = (value or "").strip().upper()
    if not name:
        return None, None
    level = logging.getLevelName(name)
    if isinstance(level, int):
        return level, None
    return None, value.strip()


def record_components(record):
    names = []
    if record.name and record.name != "root":
        names.append(record.name)
    parts = PurePath(record.pathname or "").with_suffix("").parts
    if _PACKAGE in parts:
        index = len(parts) - 1 - parts[::-1].index(_PACKAGE)
        module = list(parts[index:])
        if module[-1] == "__init__":
            module.pop()
        names.append(".".join(module))
    return names


def _matches(names, prefixes):
    return any(
        name == prefix or name.startswith(prefix + ".")
        for name in names
        for prefix in prefixes
    )


class ComponentFilter(logging.Filter):
    def __init__(self, include=(), exclude=()):
        super().__init__()
        self.include = tuple(include)
        self.exclude = tuple(exclude)

    def filter(self, record):
        names = record_components(record)
        if self.include and not _matches(names, self.include):
            return False
        if self.exclude and _matches(names, self.exclude):
            return False
        return True


def configure_log_filters(environ=None, logger=None):
    env = os.environ if environ is None else environ
    root = logger if logger is not None else logging.getLogger()
    include = parse_components(env.get(LOG_INCLUDE_ENV))
    exclude = parse_components(env.get(LOG_EXCLUDE_ENV))
    component_filter = (
        ComponentFilter(include, exclude) if include or exclude else None
    )
    applied = {}
    for env_name, is_file in (
        (LOG_CONSOLE_LEVEL_ENV, False),
        (LOG_FILE_LEVEL_ENV, True),
    ):
        level, invalid = parse_level(env.get(env_name))
        if invalid is not None:
            logging.warning(f"Ignoring invalid {env_name} value: {invalid!r}")
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler) != is_file:
                continue
            if level is not None:
                handler.setLevel(level)
                applied[env_name] = level
            if component_filter is not None:
                handler.addFilter(component_filter)
    if component_filter is not None:
        logging.info(
            f"Log components: include={list(include)} exclude={list(exclude)}"
        )
    return applied, include, exclude
