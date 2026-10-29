from __future__ import annotations

from korshmer.ui.managers.undo_redo import UndoRedoManager


def test_total_memory_decreases_after_clear():
    mgr = UndoRedoManager(max_history=10)
    for i in range(5):
        mgr.push({"type": "edit", "data": "x" * 1000, "idx": i})
    mem_before = mgr.stats["total_memory"]
    assert mem_before > 0

    mgr.clear()
    assert mgr.stats["total_memory"] == 0


def test_total_memory_decreases_when_redo_cleared():
    mgr = UndoRedoManager(max_history=10)
    for i in range(3):
        mgr.push({"type": "edit", "data": "y" * 500, "idx": i})

    mgr.undo()
    mgr.undo()
    mem_with_redo = mgr.stats["total_memory"]

    mgr.push({"type": "edit", "data": "z"})
    mem_after = mgr.stats["total_memory"]
    assert mem_after < mem_with_redo, (
        "Memory should decrease when redo stack is cleared on new push"
    )


def test_total_memory_decreases_when_deque_overflows():
    mgr = UndoRedoManager(max_history=3)
    for i in range(3):
        mgr.push({"type": "edit", "data": "a" * 2000, "idx": i})
    mem_full = mgr.stats["total_memory"]

    mgr.push({"type": "edit", "data": "b", "idx": 99})
    mem_after_overflow = mgr.stats["total_memory"]
    assert mem_after_overflow < mem_full, (
        "Memory should decrease when oldest action overflows from deque"
    )


def test_action_size_counts_nested_data():
    mgr = UndoRedoManager()
    small = {"type": "edit", "data": "x"}
    big = {"type": "edit", "data": "x" * 10000}

    size_small = mgr._action_size(small)
    size_big = mgr._action_size(big)
    assert size_big > size_small + 5000, (
        "Deep sizing should reflect nested data size"
    )
