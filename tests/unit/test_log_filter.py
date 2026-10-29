import logging
import os

import pytest

from korshmer.utils import log_filter


def make_record(name="root", pathname="/app/src/korshmer/audio/engine.py"):
    return logging.LogRecord(name, logging.INFO, pathname, 1, "msg", (), None)


@pytest.fixture
def root():
    logger = logging.Logger("test_root")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.devnull)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    yield logger, console, file_handler
    file_handler.close()


def test_parse_components():
    assert log_filter.parse_components(" a.b , ,c ") == ("a.b", "c")
    assert log_filter.parse_components(None) == ()


def test_parse_level():
    assert log_filter.parse_level("error") == (logging.ERROR, None)
    assert log_filter.parse_level("") == (None, None)
    assert log_filter.parse_level("loud") == (None, "loud")


def test_components_from_logger_name_and_path():
    assert log_filter.record_components(make_record("korshmer.network.x")) == [
        "korshmer.network.x",
        "korshmer.audio.engine",
    ]
    assert log_filter.record_components(make_record()) == [
        "korshmer.audio.engine"
    ]
    assert log_filter.record_components(
        make_record(pathname="/x/korshmer/utils/__init__.py")
    ) == ["korshmer.utils"]


def test_include_and_exclude_match_prefix_only_on_boundary():
    f = log_filter.ComponentFilter(include=["korshmer.audio"])
    assert f.filter(make_record())
    assert not f.filter(make_record(pathname="/s/korshmer/audio_file.py"))
    f = log_filter.ComponentFilter(exclude=["korshmer.audio.engine"])
    assert not f.filter(make_record())
    assert f.filter(make_record(pathname="/s/korshmer/network/a.py"))


def test_configure_sets_handler_levels(root):
    logger, console, file_handler = root
    env = {
        log_filter.LOG_CONSOLE_LEVEL_ENV: "error",
        log_filter.LOG_FILE_LEVEL_ENV: "warning",
    }
    applied, _, _ = log_filter.configure_log_filters(env, logger)
    assert console.level == logging.ERROR
    assert file_handler.level == logging.WARNING
    assert len(applied) == 2


def test_configure_adds_filter_to_both_handlers(root):
    logger, console, file_handler = root
    env = {log_filter.LOG_EXCLUDE_ENV: "korshmer.audio"}
    log_filter.configure_log_filters(env, logger)
    for handler in (console, file_handler):
        assert not handler.filter(make_record())


def test_configure_without_env_changes_nothing(root):
    logger, console, file_handler = root
    log_filter.configure_log_filters({}, logger)
    assert console.level == logging.INFO
    assert file_handler.level == logging.DEBUG
    assert console.filters == [] and file_handler.filters == []


def test_invalid_level_is_ignored(root):
    logger, console, _ = root
    log_filter.configure_log_filters(
        {log_filter.LOG_CONSOLE_LEVEL_ENV: "loud"}, logger
    )
    assert console.level == logging.INFO
