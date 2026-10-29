import pytest

from korshmer.midi.controller_mapping import (
    MAX_MACRO_ACTIONS,
    ControllerAction,
    ControllerMapping,
    Macro,
    as_action_list,
)
from korshmer.midi.mapping_store import ControllerMappingStore


def two_step():
    return Macro("duck", ("cc", 20), (("volume", 40), ("pan", None)))


def test_macro_returns_actions_in_order():
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    assert mapping.map_cc(0, 20, 127) == [
        ControllerAction("volume", 0, 40),
        ControllerAction("pan", 0, 127),
    ]


def test_plain_cc_still_single_action_and_none():
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    assert mapping.map_cc(0, 7, 100) == ControllerAction("volume", 0, 100)
    assert mapping.map_cc(0, 99, 1) is None
    assert as_action_list(mapping.map_cc(0, 99, 1)) == []
    assert len(as_action_list(mapping.map_cc(0, 7, 1))) == 1


def test_validate_cc_errors_are_kept():
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    with pytest.raises(ValueError):
        mapping.map_cc(0, 128, 1)


def test_removing_macro_restores_single_or_none():
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    assert mapping.remove_macro("duck") is True
    assert mapping.remove_macro("duck") is False
    assert mapping.map_cc(0, 20, 5) is None
    mapping.assign_macro(Macro("m", ("cc", 7), (("pan", 1),)))
    mapping.remove_macro("m")
    assert mapping.map_cc(0, 7, 5) == ControllerAction("volume", 0, 5)


def test_macro_referring_to_itself_is_rejected():
    with pytest.raises(ValueError):
        Macro("loop", ("cc", 20), (("loop", 1),))


def test_macro_cannot_contain_another_macro_name():
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    with pytest.raises(ValueError):
        Macro("outer", ("cc", 21), (("duck", 1),))


def test_macro_length_is_limited():
    ok = tuple(("pan", 1) for _ in range(MAX_MACRO_ACTIONS))
    Macro("ok", ("cc", 1), ok)
    with pytest.raises(ValueError):
        Macro("long", ("cc", 1), ok + (("pan", 1),))
    with pytest.raises(ValueError):
        Macro("empty", ("cc", 1), ())


@pytest.mark.parametrize(
    "trigger", [("cc", 128), ("note", 16, 60), ("note", 0, 128), ("x", 1), ()]
)
def test_invalid_trigger_is_rejected(trigger):
    with pytest.raises(ValueError):
        Macro("m", trigger, (("pan", 1),))


def test_trigger_cannot_be_shared_by_two_macros():
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    with pytest.raises(ValueError):
        mapping.assign_macro(Macro("other", ("cc", 20), (("pan", 1),)))
    mapping.assign_macro(Macro("duck", ("cc", 21), (("pan", 1),)))
    assert mapping.map_cc(0, 20, 1) is None


def test_note_macro_starts_on_note_on_only():
    mapping = ControllerMapping()
    mapping.assign_macro(
        Macro("n", ("note", 0, 60), (("pan", 5), ("volume", None)))
    )
    assert mapping.map_note(0, 60, 90) == [
        ControllerAction("pan", 0, 5),
        ControllerAction("volume", 0, 90),
    ]
    assert mapping.map_note(0, 60, 0) == ControllerAction("note_off", 0, 60)
    assert mapping.map_note(1, 60, 90) == ControllerAction("note_on", 1, 60)
    assert (
        len(
            mapping.map_message(
                {"type": "note_on", "channel": 0, "note": 60, "velocity": 9}
            )
        )
        == 2
    )


def test_macros_are_saved_and_loaded(tmp_path):
    store = ControllerMappingStore(tmp_path / "m.json")
    mapping = ControllerMapping()
    mapping.assign_macro(two_step())
    store.save("dev", mapping)
    loaded = ControllerMapping()
    store.apply("dev", loaded)
    assert loaded.macros == mapping.macros
    assert loaded.map_cc(0, 20, 127) == mapping.map_cc(0, 20, 127)


def test_damaged_macro_entries_are_skipped():
    mapping = ControllerMapping()
    mapping.load_dict(
        {
            "cc_actions": {"7": "volume"},
            "macros": [
                {
                    "name": "bad",
                    "trigger": ["cc", 999],
                    "actions": [["pan", 1]],
                },
                {
                    "name": "good",
                    "trigger": ["cc", 20],
                    "actions": [["pan", 1]],
                },
                "junk",
            ],
        }
    )
    assert list(mapping.macros) == ["good"]
