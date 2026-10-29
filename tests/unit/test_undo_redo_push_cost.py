from __future__ import annotations

import time

from korshmer.ui.managers.undo_redo import UndoRedoManager


def _action(index, size=20):
    return {
        "type": "notes",
        "description": f"edit {index}",
        "before": [
            {"p": j, "s": j * 0.1, "e": j * 0.1 + 1} for j in range(size)
        ],
    }


def _full_size(manager):
    return sum(
        manager._action_size(a)
        for a in list(manager.undo_stack) + list(manager.redo_stack)
    )


def test_total_memory_matches_the_stacks_after_every_operation():
    manager = UndoRedoManager(max_history=4)
    for index in range(7):
        manager.push(_action(index, size=index + 1))
        assert manager.stats["total_memory"] == _full_size(manager)
    manager.undo()
    manager.undo()
    assert manager.stats["total_memory"] == _full_size(manager)
    manager.redo()
    assert manager.stats["total_memory"] == _full_size(manager)
    manager.push(_action(99))
    assert manager.stats["total_memory"] == _full_size(manager)
    assert manager.can_redo() is False


def test_redo_into_a_full_history_drops_the_oldest_and_keeps_the_total():
    manager = UndoRedoManager(max_history=3)
    for index in range(3):
        manager.push(_action(index))
    manager.undo()
    manager.push(_action(10))
    manager.undo()
    manager.undo()
    manager.redo()
    manager.redo()
    assert manager.stats["total_memory"] == _full_size(manager)
    assert len(manager.undo_stack) == 3


def test_grouped_actions_are_counted_once_when_the_group_ends():
    manager = UndoRedoManager()
    manager.begin_group()
    manager.push(_action(1))
    manager.push(_action(2))
    assert manager.stats["total_memory"] == 0
    manager.end_group("two edits")
    assert manager.stats["total_memory"] == _full_size(manager) > 0
    assert manager.get_undo_description() == "two edits"


def test_zero_history_keeps_nothing_and_counts_nothing():
    manager = UndoRedoManager(max_history=0)
    manager.push(_action(1))
    assert manager.can_undo() is False
    assert manager.stats["total_memory"] == 0
    assert manager.undo() is None


def test_clear_resets_the_running_total():
    manager = UndoRedoManager()
    for index in range(5):
        manager.push(_action(index))
    manager.undo()
    manager.clear()
    assert manager.stats["total_memory"] == 0
    manager.push(_action(1))
    assert manager.stats["total_memory"] == _full_size(manager)


def test_push_time_does_not_grow_with_the_history_length():
    manager = UndoRedoManager(max_history=1000)
    for index in range(100):
        manager.push(_action(index))
    start = time.perf_counter()
    for index in range(100):
        manager.push(_action(index))
    early = time.perf_counter() - start
    for index in range(700):
        manager.push(_action(index))
    start = time.perf_counter()
    for index in range(100):
        manager.push(_action(index))
    late = time.perf_counter() - start
    assert late < early * 4 + 0.05


def test_group_is_recorded_as_one_undoable_action():
    manager = UndoRedoManager()
    manager.begin_group()
    manager.push({"type": "a"})
    manager.push({"type": "b"})
    manager.end_group("two edits")

    assert manager.can_undo() is True
    assert len(manager.get_history()) == 1
    undone = manager.undo()
    assert undone["type"] == "group"
    assert [a["type"] for a in undone["actions"]] == ["a", "b"]
    assert manager.can_undo() is False
    assert manager.redo() is undone


def test_empty_group_records_nothing_and_stops_grouping():
    manager = UndoRedoManager()
    manager.begin_group()
    manager.end_group("nothing")
    assert manager.can_undo() is False
    manager.push({"type": "plain"})
    assert manager.can_undo() is True
    assert manager.grouping is False


def test_end_group_without_begin_records_nothing():
    manager = UndoRedoManager()
    manager.end_group("stray")
    assert manager.can_undo() is False
