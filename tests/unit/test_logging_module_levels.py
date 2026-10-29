import logging

import pytest

from korshmer.utils import logging_config


@pytest.fixture
def restore_levels():
    names = ["korshmer.test_mod_a", "korshmer.test_mod_b"]
    saved = {n: logging.getLogger(n).level for n in names}
    yield names
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


def test_parse_levels_accepts_names_case_insensitive():
    levels, errors = logging_config.parse_log_levels(
        "korshmer.audio=debug, korshmer.network = WARNING"
    )
    assert levels == {
        "korshmer.audio": logging.DEBUG,
        "korshmer.network": logging.WARNING,
    }
    assert errors == []


def test_parse_levels_reports_bad_entries():
    levels, errors = logging_config.parse_log_levels("a=LOUD,=INFO,b,c=ERROR,")
    assert levels == {"c": logging.ERROR}
    assert errors == ["a=LOUD", "=INFO", "b"]


def test_parse_levels_empty():
    assert logging_config.parse_log_levels("") == ({}, [])
    assert logging_config.parse_log_levels(None) == ({}, [])


def test_configure_sets_logger_levels(restore_levels):
    env = {
        logging_config.LOG_LEVELS_ENV: "korshmer.test_mod_a=ERROR,korshmer.test_mod_b=DEBUG"
    }
    logging_config.configure_module_log_levels(env)
    assert logging.getLogger("korshmer.test_mod_a").level == logging.ERROR
    assert logging.getLogger("korshmer.test_mod_b").level == logging.DEBUG


def test_configure_without_env_changes_nothing(restore_levels):
    before = logging.getLogger("korshmer.test_mod_a").level
    assert logging_config.configure_module_log_levels({}) == {}
    assert logging.getLogger("korshmer.test_mod_a").level == before


def test_invalid_entry_is_logged_and_valid_still_applied(
    restore_levels, caplog
):
    env = {
        logging_config.LOG_LEVELS_ENV: "korshmer.test_mod_a=nope,korshmer.test_mod_b=WARNING"
    }
    with caplog.at_level(logging.WARNING):
        logging_config.configure_module_log_levels(env)
    assert "korshmer.test_mod_a=nope" in caplog.text
    assert logging.getLogger("korshmer.test_mod_b").level == logging.WARNING


def test_module_level_filters_records(restore_levels, caplog):
    logging_config.configure_module_log_levels(
        {logging_config.LOG_LEVELS_ENV: "korshmer.test_mod_a=ERROR"}
    )
    quiet = logging.getLogger("korshmer.test_mod_a")
    with caplog.at_level(logging.DEBUG):
        quiet.warning("hidden")
        quiet.error("shown")
    assert "hidden" not in caplog.text
    assert "shown" in caplog.text
