import json
import logging

import pytest

from korshmer.ui.managers.keyboard_shortcuts import KeyboardShortcutManager


def manager_with(tmp_path, custom):
    (tmp_path / "shortcuts_user.json").write_text(json.dumps(custom))
    manager = KeyboardShortcutManager(data_dir=tmp_path)
    manager.load_user_shortcuts()
    return manager


def test_customising_one_key_keeps_the_other_default_keys(tmp_path):
    manager = manager_with(tmp_path, {"stop": "F4"})
    assert (manager.get("stop"), manager.get("play_pause")) == ("F4", "Space")


def test_a_custom_key_that_another_action_uses_is_ignored(tmp_path):
    manager = manager_with(tmp_path, {"stop": "Space"})
    assert (manager.get("stop"), manager.get("play_pause")) == (
        "Escape",
        "Space",
    )


def test_an_ignored_custom_key_is_reported_with_both_action_names(
    tmp_path, caplog
):
    with caplog.at_level(logging.WARNING):
        manager_with(tmp_path, {"stop": "Space"})
    assert any(
        "stop" in record.message and "play_pause" in record.message
        for record in caplog.records
    )


def test_two_actions_may_swap_their_keys(tmp_path):
    manager = manager_with(tmp_path, {"stop": "Space", "play_pause": "Escape"})
    assert (manager.get("stop"), manager.get("play_pause")) == (
        "Space",
        "Escape",
    )


def test_an_action_can_be_left_without_a_key(tmp_path):
    manager = manager_with(tmp_path, {"tap_tempo": ""})
    assert manager.get("tap_tempo") == ""


def test_without_saved_shortcuts_the_defaults_stay(tmp_path):
    manager = KeyboardShortcutManager(data_dir=tmp_path)
    assert manager.load_user_shortcuts() is False
    assert manager.get("save") == "Ctrl+S"


def test_changed_keys_are_saved_for_the_next_start(tmp_path):
    first = KeyboardShortcutManager(data_dir=tmp_path)
    first.customise({"stop": "F4"})
    second = KeyboardShortcutManager(data_dir=tmp_path)
    second.load_user_shortcuts()
    assert second.get("stop") == "F4"


def test_changing_keys_to_a_clash_changes_nothing_and_names_the_clash(tmp_path):
    manager = KeyboardShortcutManager(data_dir=tmp_path)
    clashes = manager.customise({"stop": "Space"})
    assert manager.get("stop") == "Escape"
    assert [set(clash[:2]) for clash in clashes] == [{"stop", "play_pause"}]


@pytest.mark.parametrize("action", ["find_tool", "home", "stop", "export"])
def test_safeguard_every_bound_action_has_a_default_key(action):
    assert KeyboardShortcutManager().get(action)
