import sys
import time
from collections import deque


class UndoRedoManager:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.undo_stack = deque(maxlen=max_history)
        self.redo_stack = []
        self._undo_sizes = deque()
        self._redo_sizes = []
        self.current_group = []
        self.grouping = False
        self.stats = {"undo_count": 0, "redo_count": 0, "total_memory": 0}

    def begin_group(self):
        self.grouping = True
        self.current_group = []

    def end_group(self, description=""):
        actions = self.current_group if self.grouping else []
        self.grouping = False
        self.current_group = []
        if actions:
            self.push(
                {
                    "type": "group",
                    "actions": actions,
                    "description": description,
                    "timestamp": time.time(),
                }
            )

    @staticmethod
    def _action_size(action):
        seen: set[int] = set()

        def _deep(obj):
            oid = id(obj)
            if oid in seen:
                return 0
            seen.add(oid)
            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    total += _deep(k) + _deep(v)
            elif isinstance(obj, (list, tuple, deque)):
                for item in obj:
                    total += _deep(item)
            return total

        try:
            return _deep(action)
        except Exception:
            return 0

    def _add_undo(self, action, size):
        maxlen = self.undo_stack.maxlen
        if maxlen == 0:
            return
        if maxlen is not None and len(self._undo_sizes) >= maxlen:
            self.stats["total_memory"] -= self._undo_sizes.popleft()
        self.undo_stack.append(action)
        self._undo_sizes.append(size)
        self.stats["total_memory"] += size

    def push(self, action):
        if self.grouping:
            self.current_group.append(action)
            return

        self.redo_stack.clear()
        self.stats["total_memory"] -= sum(self._redo_sizes)
        self._redo_sizes.clear()
        self._add_undo(action, self._action_size(action))

    def undo(self):
        if not self.undo_stack:
            return None
        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        self._redo_sizes.append(self._undo_sizes.pop())
        self.stats["undo_count"] += 1
        return action

    def redo(self):
        if not self.redo_stack:
            return None
        action = self.redo_stack.pop()
        size = self._redo_sizes.pop()
        self.stats["total_memory"] -= size
        self._add_undo(action, size)
        self.stats["redo_count"] += 1
        return action

    def can_undo(self):
        return len(self.undo_stack) > 0

    def can_redo(self):
        return len(self.redo_stack) > 0

    def get_undo_description(self):
        if self.undo_stack:
            return self.undo_stack[-1].get("description", "Unknown action")
        return ""

    def get_redo_description(self):
        if self.redo_stack:
            return self.redo_stack[-1].get("description", "Unknown action")
        return ""

    def get_history(self):
        return list(self.undo_stack)

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._undo_sizes.clear()
        self._redo_sizes.clear()
        self.stats = {"undo_count": 0, "redo_count": 0, "total_memory": 0}

    def get_stats(self):
        return {
            **self.stats,
            "undo_available": len(self.undo_stack),
            "redo_available": len(self.redo_stack),
        }
