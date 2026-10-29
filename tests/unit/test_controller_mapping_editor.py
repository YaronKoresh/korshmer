import json

import pytest
from PySide6.QtWidgets import QApplication

from korshmer.midi.controller_mapping import (
    ACTION_NAMES,
    CC_ACTIONS,
    ControllerAction,
    ControllerMapping,
)
from korshmer.midi.hotplug import MidiHotplugMonitor
from korshmer.midi.mapping_store import ControllerMappingStore


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    return ControllerMappingStore(tmp_path / "mappings.json")


def test_changed_cc_maps_to_new_action_only_in_that_mapping():
    mapping = ControllerMapping()
    mapping.set_cc_action(20, "volume")
    assert mapping.map_cc(0, 20, 5) == ControllerAction("volume", 0, 5)
    assert ControllerMapping().map_cc(0, 20, 5) is None
    assert 20 not in CC_ACTIONS["generic"]


@pytest.mark.parametrize("cc", [128, -1, "x", None])
def test_out_of_range_cc_is_rejected(cc):
    with pytest.raises(ValueError):
        ControllerMapping().set_cc_action(cc, "volume")


def test_unknown_action_is_rejected():
    mapping = ControllerMapping()
    with pytest.raises(ValueError):
        mapping.set_cc_action(20, "not_an_action")
    assert 20 not in mapping.cc_actions


def test_unmapped_cc_still_returns_none_and_range_error_kept():
    mapping = ControllerMapping()
    assert mapping.map_cc(0, 99, 1) is None
    with pytest.raises(ValueError):
        mapping.map_cc(0, 128, 1)


def test_two_ccs_may_share_an_action():
    mapping = ControllerMapping("pioneer_ddj")
    mapping.set_cc_action(1, "filter")
    assert mapping.cc_actions[1] == mapping.cc_actions[0x17] == "filter"


def test_remove_and_reset():
    mapping = ControllerMapping()
    assert mapping.remove_cc_action(7) is True
    assert mapping.remove_cc_action(7) is False
    assert mapping.map_cc(0, 7, 1) is None
    mapping.set_cc_action(20, "pan")
    mapping.reset_cc_actions()
    assert mapping.cc_actions == CC_ACTIONS["generic"]


def test_replace_is_all_or_nothing():
    mapping = ControllerMapping()
    before = dict(mapping.cc_actions)
    with pytest.raises(ValueError):
        mapping.replace_cc_actions({1: "pan", 200: "pan"})
    assert mapping.cc_actions == before


def test_save_and_load_give_same_table(store):
    mapping = ControllerMapping()
    mapping.set_cc_action(20, "volume")
    mapping.remove_cc_action(1)
    store.save("My Keys", mapping)

    loaded = ControllerMapping()
    assert store.apply("My Keys", loaded) is True
    assert loaded.cc_actions == mapping.cc_actions
    assert store.apply("Other", ControllerMapping()) is False
    assert store.device_names() == ["My Keys"]


def test_damaged_file_means_defaults(store):
    store.path.write_text("{not json", encoding="utf-8")
    mapping = ControllerMapping()
    assert store.apply("x", mapping) is False
    assert mapping.cc_actions == CC_ACTIONS["generic"]
    store.path.write_text(
        json.dumps(
            {"x": {"cc_actions": {"1": "pan", "300": "pan", "2": "bad"}}}
        )
    )
    assert store.apply("x", mapping) is True
    assert mapping.cc_actions == {1: "pan"}


def test_delete(store):
    store.save("a", ControllerMapping())
    assert store.delete("a") is True
    assert store.delete("a") is False


def test_hotplug_loads_saved_mapping(store, qapp):
    saved = ControllerMapping()
    saved.set_cc_action(20, "volume")
    store.save("Unknown Keys", saved)
    devices = ["Unknown Keys"]
    monitor = MidiHotplugMonitor(
        list_devices=lambda: devices, interval=0.01, mapping_store=store
    )
    monitor.poll()
    mapping = monitor.mapping_for("Unknown Keys")
    assert mapping.map_cc(0, 20, 5) == ControllerAction("volume", 0, 5)

    saved.set_cc_action(21, "pan")
    store.save("Unknown Keys", saved)
    monitor.reload_mappings()
    assert mapping.map_cc(0, 21, 5) == ControllerAction("pan", 0, 5)


def test_dialog_edit_save_reopen(store, qapp):
    from korshmer.ui.widgets.dialogs.controller_mapping import (
        ControllerMappingDialog,
    )

    dialog = ControllerMappingDialog(["Unknown Keys"], store)
    dialog.add_row(20, "volume")
    assert dialog.save() is True
    dialog.close()

    again = ControllerMappingDialog(["Unknown Keys"], store)
    assert (20, "volume") in again.rows()
    again.table.setCurrentCell(0, 0)
    assert again.learn_cc(33) is True
    assert again.rows()[0][0] == 33
    again.close()


def test_dialog_rejects_duplicate_cc_and_resets(store, qapp):
    from korshmer.ui.widgets.dialogs.controller_mapping import (
        ControllerMappingDialog,
    )

    dialog = ControllerMappingDialog(["Unknown Keys"], store)
    dialog.add_row(7, "pan")
    assert dialog.save() is False
    assert store.device_names() == []
    dialog.reset_to_defaults()
    assert dict(dialog.rows()) == CC_ACTIONS["generic"]
    dialog.close()


def test_action_names_cover_default_tables():
    for table in CC_ACTIONS.values():
        assert set(table.values()) <= set(ACTION_NAMES)
